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
  std::string inputRootFile;
  std::string payloadTag;
};

// Constructor
QGLikelihoodDBWriter::QGLikelihoodDBWriter(const edm::ParameterSet& pSet){
  inputRootFile    = pSet.getParameter<std::string>("src");
  payloadTag       = pSet.getParameter<std::string>("payload");
}

bool QGLikelihoodDBWriter::extractString(std::string mySubstring, std::string& myString){
  size_t subStringPos = myString.find(mySubstring);
  if(subStringPos != std::string::npos){
    myString = myString.substr(subStringPos + mySubstring.length(), std::string::npos);
    return true;
  } else return false;
}

// Begin Job
void QGLikelihoodDBWriter::beginJob(){

  QGLikelihoodObject *payload = new QGLikelihoodObject();
  payload->data.clear();

  // Get the ROOT file and the vectors with binning information
  TFile *f = TFile::Open(edm::FileInPath(inputRootFile.c_str()).fullPath().c_str());
  TVectorT<float> *etaBins; 	f->GetObject("etaBins", etaBins);
  TVectorT<float> *ptBinsC; 	f->GetObject("ptBinsC", ptBinsC);
  TVectorT<float> *ptBinsF; 	f->GetObject("ptBinsF", ptBinsF);
  TVectorT<float> *rhoBins; 	f->GetObject("rhoBins", rhoBins);

  // Get keys to the histograms
  TList *keys = f->GetListOfKeys();
  if(!keys){
    edm::LogError("NoKeys") << "There are no keys in the input file." << std::endl;
    return;
  }
 
  // Loop over directories/histograms
  TIter nextdir(keys);
  TKey *keydir;
  while((keydir = (TKey*)nextdir())){
    if(!keydir->IsFolder()) continue;
    TDirectory *dir = (TDirectory*)keydir->ReadObj() ;
    TIter nexthist(dir->GetListOfKeys());
    TKey *keyhist;
    while((keyhist = (TKey*)nexthist())){
      std::string histname = keyhist->GetName();
      int varIndex, qgIndex;

      // First check the variable name, and use index in same order as RecoJets/JetProducers/plugins/QGTagger.cc:73
      if(extractString("mult", histname)) varIndex = 0;
      else if(extractString("ptD", histname)) varIndex = 1;
      else if(extractString("axis2", histname)) varIndex = 2;
      else continue;

      // Check quark or gluon
      if(extractString("quark", histname)) qgIndex = 0;
      else if(extractString("gluon", histname)) qgIndex = 1;
      else continue;

      // Get eta, pt and rho ranges
      extractString("eta-", histname);
      int etaBin = std::atoi(histname.substr(0, histname.find("_")).c_str());
      extractString("pt-", histname);
      int ptBin = std::atoi(histname.substr(0, histname.find("_")).c_str());
      extractString("rho-", histname);
      int rhoBin = std::atoi(histname.substr(0, histname.find("_")).c_str());

      float etaMin = (*etaBins)[etaBin];
      float etaMax = (*etaBins)[etaBin+1];
      TVectorT<float> *ptBins = (etaBin == 0? ptBinsC : ptBinsF);
      float ptMin = (*ptBins)[ptBin];
      float ptMax = (*ptBins)[ptBin+1];
      float rhoMin = (*rhoBins)[rhoBin];
      float rhoMax = (*rhoBins)[rhoBin+1];

      // Print out for debugging      
      char buff[1000];
      sprintf(buff, "%50s : var=%1d, qg=%1d, etaMin=%6.2f, etaMax=%6.2f, ptMin=%8.2f, ptMax=%8.2f, rhoMin=%6.2f, rhoMax=%6.2f", keyhist->GetName(), varIndex, qgIndex, etaMin, etaMax, ptMin, ptMax, rhoMin, rhoMax );
      edm::LogVerbatim("HistName") << buff << std::endl;

      // Define category parameters
      QGLikelihoodCategory category;
      category.RhoMin = rhoMin;
      category.RhoMax = rhoMax;
      category.PtMin = ptMin;
      category.PtMax = ptMax;
      category.EtaMin = etaMin;
      category.EtaMax = etaMax;
      category.QGIndex = qgIndex;
      category.VarIndex = varIndex;

      // Get TH1 
      TH1* th1hist = (TH1*) keyhist->ReadObj();

      // Transform ROOT TH1 to QGLikelihoodObject (same indexing)
      QGLikelihoodObject::Histogram histogram(th1hist->GetNbinsX(), th1hist->GetXaxis()->GetBinLowEdge(1), th1hist->GetXaxis()->GetBinUpEdge(th1hist->GetNbinsX()));
      for(int ibin = 0; ibin <= th1hist->GetNbinsX() + 1; ++ibin){
	histogram.setBinContent(ibin, th1hist->GetBinContent(ibin));
      }

      // Add this entry with its category parameters, histogram and mean
      QGLikelihoodObject::Entry entry;
      entry.category = category;
      entry.histogram = histogram; 
      entry.mean = th1hist->GetMean();
      payload->data.push_back(entry);
    }
  }
  // Define the valid range, if no category is found within these bounds a warning will be thrown
  payload->qgValidRange.RhoMin = rhoBins->Min();
  payload->qgValidRange.RhoMax = rhoBins->Max();
  payload->qgValidRange.EtaMin = etaBins->Min();
  payload->qgValidRange.EtaMax = etaBins->Max();
  payload->qgValidRange.PtMin  = ptBinsC->Min();
  payload->qgValidRange.PtMax  = ptBinsC->Max();
  payload->qgValidRange.QGIndex = -1;
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

