#ifndef SiPixelHistoricInfoReader_H
#define SiPixelHistoricInfoReader_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TH2F.h"
#include "TObjArray.h"


class SiPixelHistoricInfoReader : public edm::EDAnalyzer {
  typedef std::vector<std::string> vstring; 

public:
  explicit SiPixelHistoricInfoReader(const edm::ParameterSet&);
	  ~SiPixelHistoricInfoReader();

  virtual void beginJob(const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void endJob(); 
  
  std::string getMEregionString(uint32_t) const; 
  void fillDebugHistogram(TString, float, float); 
  
private:
  edm::ParameterSet parameterSet_;
  
  bool firstBeginRun_; 
  bool printDebug_;
  bool normEvents_; 
  bool variable_[10]; 
  std::vector<std::string> variables_; 
  std::vector<uint32_t> allDetIds; 
  std::string outputFile_; 
  TFile* outputFile; 
  TObjArray* AllDetHistograms; 
  TString hisID, title; 
};

#endif
