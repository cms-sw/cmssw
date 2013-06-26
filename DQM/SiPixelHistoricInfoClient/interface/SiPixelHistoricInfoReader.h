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

#include "TString.h"
#include "TObjArray.h"
#include "TFile.h"


class SiPixelHistoricInfoReader : public edm::EDAnalyzer {
  typedef std::vector<std::string> vstring; 

public:
  explicit SiPixelHistoricInfoReader(const edm::ParameterSet&);
	  ~SiPixelHistoricInfoReader();

  virtual void beginJob();
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  virtual void endJob(); 
  
  std::string getMEregionString(uint32_t) const; 
  void plot(); 
  
private:
  edm::ParameterSet parameterSet_;
  
  bool firstBeginRun_; 
  bool printDebug_;

  bool variable_[20]; 
  std::vector<std::string> variables_; 

  std::vector<uint32_t> allDetIds; 
  TString hisID, title; 
  TObjArray* AllDetHistograms; 

  bool normEvents_; 

  bool makePlots_;
  std::string typePlots_;
  std::string outputDir_; 
  std::string outputFile_; 
  TFile* outputDirFile_; 
};

#endif
