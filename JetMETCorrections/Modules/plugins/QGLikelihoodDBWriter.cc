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
  ~QGLikelihoodDBWriter() {}

 private:
  bool extractString(std::string, std::string&);
  bool getVectorFromFile(TFile*, std::vector<float>&, const TString&);
  QGLikelihoodObject::Histogram transformToHistogramObject(TH1* th1);
  std::string inputRootFile;
  std::string payloadTag;
};

// Constructor
QGLikelihoodDBWriter::QGLikelihoodDBWriter(const edm::ParameterSet& pSet){
  inputRootFile    = pSet.getParameter<std::string>("src");
  payloadTag       = pSet.getParameter<std::string>("payload");
}


// Translates the TVector with the bins to std::vector
bool QGLikelihoodDBWriter::getVectorFromFile(TFile* f, std::vector<float>& v, const TString& name){
  TVectorT<float> *tv = nullptr;
  f->GetObject(name, tv);
  if(!tv) return false;
  for(int i = 0; i < tv->GetNoElements(); ++i) v.push_back((*tv)[i]);
  return true;
}

// Transform ROOT TH1 to QGLikelihoodObject (same indexing)
QGLikelihoodObject::Histogram QGLikelihoodDBWriter::transformToHistogramObject(TH1* th1){
  QGLikelihoodObject::Histogram histogram(th1->GetNbinsX(), th1->GetXaxis()->GetBinLowEdge(1), th1->GetXaxis()->GetBinUpEdge(th1->GetNbinsX()));
  for(int ibin = 0; ibin <= th1->GetNbinsX() + 1; ++ibin){
    histogram.setBinContent(ibin, th1->GetBinContent(ibin));
  }
  return histogram;
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
  // Here we do not store the copies, but extend the range of the category
  std::map<std::vector<int>, TH1*> pdfs;
  std::map<std::vector<int>, QGLikelihoodCategory> categories;
  for(TString likelihoodVar : {"axis2","mult","ptD"}){
    int varIndex = (likelihoodVar == "mult" ? 0 : (likelihoodVar == "axis2" ? 1 : 2));
    for(TString type : {"quark","gluon"}){
      int qgIndex = (type == "quark" ? 0 : 1);
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

            std::vector<int> binNumbers = {qgIndex, varIndex, i,j,k};
            TString key = TString::Format(likelihoodVar + "/" + likelihoodVar + "_" + type + "_eta%d_pt%d_rho%d", i, j, k);
            TH1* pdf = (TH1*) f->Get(key);
            if(!pdf){
              edm::LogError("NoPDF") << "Could not found pdf with key  " << key << " in input file" << std::endl;
              return;
            }

            if(k > 0){											// If copy of neighbour rho category --> extend range of first one
              std::vector<int> neighbour = {qgIndex, varIndex, i,j,k-1};
              if(pdf->GetTitle() == pdfs[neighbour]->GetTitle()){
                categories[neighbour].RhoMax = category.RhoMax;
                continue;                
              }
            }
            pdfs[binNumbers]       = pdf; 
            categories[binNumbers] = category;
          }
          if(j > 0){											// If copy of neighbour pt category --> extend range of first one
            for(int k = 0; k < (int)gridOfBins["rho"].size() - 1; ++k){					// (also working after rho bins were already merged)
              std::vector<int> cat1 = {qgIndex, varIndex, i,j-1,k};
              std::vector<int> cat2 = {qgIndex, varIndex, i,j,k};
              if(!pdfs.count(cat1) || !pdfs.count(cat2))             continue;
              if(pdfs[cat1]->GetTitle() != pdfs[cat2]->GetTitle())   continue;
              std::cout << pdfs[cat1]->GetTitle() << std::endl;
              if(categories[cat1].RhoMin != categories[cat2].RhoMin) continue;
              if(categories[cat1].RhoMax != categories[cat2].RhoMax) continue;
              categories[cat1].PtMax = categories[cat2].PtMax;
              categories.erase(cat2);
              pdfs.erase(cat2);
            }
          }
        }
      }
    }
  }


  // Get the weights from the file
  std::map<std::vector<int>, float> weights;
  for(int i = 0; i < (int)gridOfBins["eta"].size() - 1; ++i){
    for(int j = 0; j < (int)gridOfBins["pt"].size() - 1; ++j){
      for(int k = 0; k < (int)gridOfBins["rho"].size() - 1; ++k){
        std::vector<float> weightsPerBin;
        if(!getVectorFromFile(f, weightsPerBin, TString::Format("weights/Weights_eta%d_pt%d_rho%d", i, j, k))){
          edm::LogError("NoWeights") << "Missing weights for bin eta" << i << "_pt" << j << "_rho" << k << "!" << std::endl;
          weightsPerBin = {1,1,1};
          return;
        }
       
        for(int varIndex = 0; varIndex < 3; ++varIndex){
          std::vector<int> quarkCategory = {0, varIndex, i, j, k}; 
          std::vector<int> gluonCategory = {1, varIndex, i, j, k}; 
          weights[quarkCategory] = weightsPerBin[varIndex];
          weights[gluonCategory] = weightsPerBin[varIndex];
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
    entry.mean      = pdfs[category.first]->GetMean();
    entry.weight    = weights[category.first];
    payload->data.push_back(entry);
    
    char buff[1000];
    sprintf(buff, "%6d) var=%1d, qg=%1d, eta={%6.2f, %6.2f}, pt={%8.2f, %8.2f}, rhoMin={%6.2f, %6.2f}, weight=%6.2f", i++,
                        category.second.VarIndex, category.second.QGIndex, category.second.EtaMin, category.second.EtaMax, 
                        category.second.PtMin,    category.second.PtMax,   category.second.RhoMin, category.second.RhoMax, entry.weight);
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

