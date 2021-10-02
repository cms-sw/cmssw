#ifndef SiPixelLorentzAngleReader_H
#define SiPixelLorentzAngleReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"

//
//
// class decleration
//
class SiPixelLorentzAngleReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelLorentzAngleReader(const edm::ParameterSet&);
  ~SiPixelLorentzAngleReader() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAToken_;
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleSimRcd> siPixelSimLAToken_;

  bool printdebug_;
  TH1F* LorentzAngleBarrel_;
  TH1F* LorentzAngleForward_;
  bool useSimRcd_;
};

#endif
