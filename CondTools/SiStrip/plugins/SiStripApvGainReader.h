#ifndef SiStripApvGainReader_H
#define SiStripApvGainReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"

class SiStripApvGainReader : public edm::EDAnalyzer {
public:
  explicit SiStripApvGainReader(const edm::ParameterSet&);
  ~SiStripApvGainReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  bool printdebug_;
  std::string formatedOutput_;
  uint32_t gainType_;
  edm::Service<TFileService> fs_;
  TTree* tree_;
  int id_ = 0, detId_ = 0, apvId_ = 0;
  double gain_ = 0;
};
#endif
