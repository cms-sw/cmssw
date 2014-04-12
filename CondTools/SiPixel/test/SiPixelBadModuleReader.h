#ifndef SiPixelBadModuleReader_H
#define SiPixelBadModuleReader_H

// system include files


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH2F.h"




class SiPixelBadModuleReader : public edm::EDAnalyzer {

 public:
  explicit SiPixelBadModuleReader( const edm::ParameterSet& );
  ~SiPixelBadModuleReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );
    
 private:
  uint32_t printdebug_;
  std::string whichRcd;
  TH2F *_TH2F_dead_modules_BPIX_lay1;
  TH2F *_TH2F_dead_modules_BPIX_lay2;
  TH2F *_TH2F_dead_modules_BPIX_lay3;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_minusZ_disk2;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk1;
  TH2F *_TH2F_dead_modules_FPIX_plusZ_disk2;
};
#endif
