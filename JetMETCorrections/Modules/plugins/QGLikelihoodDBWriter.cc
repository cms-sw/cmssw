// Author: Benedikt Hegner, Tom Cornelis
// Email:  benedikt.hegner@cern.ch, tom.cornelis@cern.ch

#include "TFile.h"
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

  // Get the ROOT files and the keys to the histogram
  TFile *f = TFile::Open(edm::FileInPath(inputRootFile.c_str()).fullPath().c_str());
  TList *keys = f->GetListOfKeys();
  if(!keys){
    edm::LogError("NoKeys") << "There are no keys in the input file." << std::endl;
    return;
  }
  
  // Loop over directories/histograms
  TIter nextdir(keys);
  TKey *keydir;
  while((keydir = (TKey*)nextdir())){
    TDirectory *dir = (TDirectory*)keydir->ReadObj() ;
    TIter nexthist(dir->GetListOfKeys());
    TKey *keyhist;
    while((keyhist = (TKey*)nexthist())){

      float ptMin, ptMax, rhoMin, rhoMax, etaMin, etaMax;
      int varIndex, qgIndex;

      std::string histname = keyhist->GetName();
      std::string histname_ = keyhist->GetName();

      // First check the variable name, and use index in same order as RecoJets/JetProducers/plugins/QGTagger.cc:73
      if(extractString("nPFCand_QC_ptCutJet0", histname)) varIndex = 0;
      else if(extractString("ptD_QCJet0", histname)) varIndex = 1;
      else if(extractString("axis2_QCJet0", histname)) varIndex = 2;
      else continue;

      // Check pseudorapidity range
      if(extractString("_F", histname)){ etaMin = 2.5; etaMax = 4.7;}
      else { etaMin = 0.;etaMax = 2.5;}

      // Check quark or gluon
      if(extractString("quark", histname)) qgIndex = 0;
      else if(extractString("gluon", histname)) qgIndex = 1;
      else continue;

      // Access the pt information
      extractString("pt", histname);
      ptMin = std::atof(histname.substr(0, histname.find("_")).c_str());
      extractString("_", histname);
      ptMax = std::atof(histname.substr(0, histname.find("rho")).c_str());

      if(etaMin == 2.5 && ptMin > 128) continue;		//In forward use one bin for 127->2000
      if(etaMin == 2.5 && ptMin == 127) ptMax = 4000;

      // Access the rho information
      extractString("rho", histname);
      rhoMin = std::atof(histname.c_str());
      rhoMax = rhoMin + 1.; // WARNING: Check if this is still valid when changed to fixedGrid rho (try to move it in the name...)

      // Print out for debugging      
      char buff[1000];
      sprintf(buff, "%50s : var=%1d, qg=%1d, etaMin=%6.2f, etaMax=%6.2f, ptMin=%8.2f, ptMax=%8.2f, rhoMin=%6.2f, rhoMax=%6.2f", histname_.c_str(), varIndex, qgIndex, etaMin, etaMax, ptMin, ptMax, rhoMin, rhoMax );
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

      // In the future, this part will (preferably) move to the making of the root files
      if(th1hist->GetEntries()<50 ) 		th1hist->Rebin(5); 	// try to make it more stable
      else if(th1hist->GetEntries()<500 ) 	th1hist->Rebin(2); 	// try to make it more stable
      th1hist->Scale(1./th1hist->Integral("width")); 

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
  payload->qgValidRange.RhoMin = 0;
  payload->qgValidRange.RhoMax = 46;
  payload->qgValidRange.EtaMin = 0;
  payload->qgValidRange.EtaMax = 4.7;
  payload->qgValidRange.PtMin  = 20;
  payload->qgValidRange.PtMax  = 4000;
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

