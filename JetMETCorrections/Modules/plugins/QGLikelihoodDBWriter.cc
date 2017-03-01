// Author: Benedikt Hegner, Tom Cornelis
// Email:  benedikt.hegner@cern.ch, tom.cornelis@cern.ch

#include "TFile.h"
#include "TVector.h"
#include "TList.h"
#include "TKey.h"
#include "TH1.h"
#include <sstream>
#include <stdlib.h>  
#include <vector>
#include <memory>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

class  QGLikelihoodDBWriter : public edm::EDAnalyzer{
 public:
  QGLikelihoodDBWriter(const edm::ParameterSet&);
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override {}
  virtual void endJob() override {}
  ~QGLikelihoodDBWriter(){}

 private:
  bool getVectorFromFile(TFile*, std::vector<float>&, const TString&);
  void tryToMerge(std::map<std::vector<int>, QGLikelihoodCategory>&, std::map<std::vector<int>, TH1*>&, std::vector<int>&, int);
  QGLikelihoodObject::Histogram transformToHistogramObject(TH1*);
  std::string inputRootFile, payloadTag;
};


// Constructor
QGLikelihoodDBWriter::QGLikelihoodDBWriter(const edm::ParameterSet& pSet){
  inputRootFile    = pSet.getParameter<std::string>("src");
  payloadTag       = pSet.getParameter<std::string>("payload");
}


// Get vector from input file (includes translating TVector to std::vector)
bool QGLikelihoodDBWriter::getVectorFromFile(TFile* f, std::vector<float>& vector, const TString& name){
  TVectorT<float> *tVector = nullptr;
  f->GetObject(name, tVector);
  if(!tVector) return false;
  for(int i = 0; i < tVector->GetNoElements(); ++i) vector.push_back((*tVector)[i]);
  return true;
}


// Transform ROOT TH1 to QGLikelihoodObject (same indexing)
QGLikelihoodObject::Histogram QGLikelihoodDBWriter::transformToHistogramObject(TH1* th1){
  QGLikelihoodObject::Histogram histogram(th1->GetNbinsX(), th1->GetXaxis()->GetBinLowEdge(1), th1->GetXaxis()->GetBinUpEdge(th1->GetNbinsX()));
  for(int ibin = 0; ibin <= th1->GetNbinsX() + 1; ++ibin) histogram.setBinContent(ibin, th1->GetBinContent(ibin));
  return histogram;
}


// Try to merge bin with neighbouring bin (index = 2,3,4 for eta,pt,rho)
void QGLikelihoodDBWriter::tryToMerge(std::map<std::vector<int>, QGLikelihoodCategory>& categories, std::map<std::vector<int>, TH1*>& pdfs, std::vector<int>& binNumbers, int index){
  if(!pdfs[binNumbers]) return;
  std::vector<int> neighbour = binNumbers;
  do {
    if(--(neighbour[index]) < 0) return;
  } while (!pdfs[neighbour]);
  if(TString(pdfs[binNumbers]->GetTitle()) != TString(pdfs[neighbour]->GetTitle())) return;
  if(index != 4 && categories[neighbour].RhoMax != categories[binNumbers].RhoMax)   return;
  if(index != 4 && categories[neighbour].RhoMin != categories[binNumbers].RhoMin)   return;
  if(index != 3 && categories[neighbour].PtMax  != categories[binNumbers].PtMax)    return;
  if(index != 3 && categories[neighbour].PtMin  != categories[binNumbers].PtMin)    return;
  if(index != 2 && categories[neighbour].EtaMax != categories[binNumbers].EtaMax)   return;
  if(index != 2 && categories[neighbour].EtaMin != categories[binNumbers].EtaMin)   return;

  if(index == 4) categories[neighbour].RhoMax = categories[binNumbers].RhoMax;
  if(index == 3) categories[neighbour].PtMax  = categories[binNumbers].PtMax;
  if(index == 2) categories[neighbour].EtaMax = categories[binNumbers].EtaMax;
  pdfs.erase(binNumbers);
  categories.erase(binNumbers);
}


// Begin Job
void QGLikelihoodDBWriter::beginJob(){
  QGLikelihoodObject *payload = new QGLikelihoodObject();
  payload->data.clear();

  // Get the ROOT file
  TFile *f = TFile::Open(edm::FileInPath(inputRootFile.c_str()).fullPath().c_str());

  // The ROOT file contains the binning for each variable, needed to set up the binning grid
  std::map<TString, std::vector<float>> gridOfBins;
  for(TString binVariable : {"eta", "pt", "rho"}){
    if(!getVectorFromFile(f, gridOfBins[binVariable], binVariable + "Bins")){
      edm::LogError("NoBins") << "Missing bin information for " << binVariable << " in input file" << std::endl;
      return;
    }
  }

  // Get pdf's from file and associate them to a QGLikelihoodCategory
  // Some pdf's in the ROOT-file are copies from each other, with the same title: those are merged bins in pt and rho
  // Here we do not store the copies, but try to extend the range of the neighbouring category (will result in less comparisons during application phase)
  std::map<std::vector<int>, TH1*> pdfs;
  std::map<std::vector<int>, QGLikelihoodCategory> categories;
  for(TString type : {"gluon","quark"}){
    int qgIndex = (type == "gluon");									// Keep numbering same as in RecoJets/JetAlgorithms/src/QGLikelihoodCalculator.cc
    for(TString likelihoodVar : {"mult","ptD","axis2"}){
      int varIndex = (likelihoodVar == "mult" ? 0 : (likelihoodVar == "ptD" ? 1 : 2));			// Keep order same as in RecoJets/JetProducers/plugins/QGTagger.cc
      for(int i = 0; i < (int)gridOfBins["eta"].size() - 1; ++i){
        for(int j = 0; j < (int)gridOfBins["pt"].size() - 1; ++j){
          for(int k = 0; k < (int)gridOfBins["rho"].size() - 1; ++k){

            QGLikelihoodCategory category;
            category.EtaMin   = gridOfBins["eta"][i];
            category.EtaMax   = gridOfBins["eta"][i+1];
            category.PtMin    = gridOfBins["pt"][j];
            category.PtMax    = gridOfBins["pt"][j+1];
            category.RhoMin   = gridOfBins["rho"][k];
            category.RhoMax   = gridOfBins["rho"][k+1];
            category.QGIndex  = qgIndex;
            category.VarIndex = varIndex;

            TString key = TString::Format(likelihoodVar + "/" + likelihoodVar + "_" + type + "_eta%d_pt%d_rho%d", i, j, k);
            TH1* pdf = (TH1*) f->Get(key);
            if(!pdf){
              edm::LogError("NoPDF") << "Could not found pdf with key  " << key << " in input file" << std::endl;
              return;
            }

            std::vector<int> binNumbers = {qgIndex, varIndex, i,j,k};
            pdfs[binNumbers]       = pdf; 
            categories[binNumbers] = category;

            tryToMerge(categories, pdfs, binNumbers, 4);
          }
          for(int k = 0; k < (int)gridOfBins["rho"].size() - 1; ++k){
            std::vector<int> binNumbers = {qgIndex, varIndex, i,j,k};
            tryToMerge(categories, pdfs, binNumbers, 3);
          }
        }
        for(int j = 0; j < (int)gridOfBins["pt"].size() - 1; ++j){
          for(int k = 0; k < (int)gridOfBins["rho"].size() - 1; ++k){
            std::vector<int> binNumbers = {qgIndex, varIndex, i,j,k};
            tryToMerge(categories, pdfs, binNumbers, 2);
          }
        }
      }
    }
  }


  // Write all categories with their histograms to file
  int i = 0;
  for(auto category : categories){
    QGLikelihoodObject::Entry entry;
    entry.category  = category.second;
    entry.histogram = transformToHistogramObject(pdfs[category.first]);
    entry.mean      = 0; // not used by the algorithm, is an old data member used in the past, but DB objects are not allowed to change
    payload->data.push_back(entry);
    
    char buff[1000];
    sprintf(buff, "%6d) var=%1d\t\tqg=%1d\t\teta={%5.2f,%5.2f}\t\tpt={%8.2f,%8.2f}\t\trho={%6.2f,%8.2f}", i++,
                        category.second.VarIndex, category.second.QGIndex, category.second.EtaMin, category.second.EtaMax, 
                        category.second.PtMin,    category.second.PtMax,   category.second.RhoMin, category.second.RhoMax);
    edm::LogVerbatim("HistName") << buff << std::endl;
  }

  // Define the valid range, if no category is found within these bounds a warning will be thrown
  payload->qgValidRange.EtaMin   = gridOfBins["eta"].front();
  payload->qgValidRange.EtaMax   = gridOfBins["eta"].back();
  payload->qgValidRange.PtMin    = gridOfBins["pt"].front();
  payload->qgValidRange.PtMax    = gridOfBins["pt"].back();
  payload->qgValidRange.RhoMin   = gridOfBins["rho"].front();
  payload->qgValidRange.RhoMax   = gridOfBins["rho"].back();
  payload->qgValidRange.QGIndex  = -1;
  payload->qgValidRange.VarIndex = -1;

  // Now write it into the DB
  edm::LogInfo("UserOutput") << "Opening PoolDBOutputService" << std::endl;

  edm::Service<cond::service::PoolDBOutputService> s;
  if(s.isAvailable()){ 
    edm::LogInfo("UserOutput") <<  "Setting up payload with " << payload->data.size() <<  " entries and tag " << payloadTag << std::endl;
    if (s->isNewTagRequest(payloadTag))	s->createNewIOV<QGLikelihoodObject>(payload, s->beginOfTime(), s->endOfTime(), payloadTag);
    else s->appendSinceTime<QGLikelihoodObject>(payload, 111, payloadTag);
  }
  edm::LogInfo("UserOutput") <<  "Wrote in CondDB QGLikelihood payload label: " << payloadTag << std::endl;
}


DEFINE_FWK_MODULE(QGLikelihoodDBWriter);

