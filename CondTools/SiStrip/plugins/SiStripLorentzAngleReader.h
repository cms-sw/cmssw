#ifndef SiStripLorentzAngleReader_H
#define SiStripLorentzAngleReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"

class SiStripLorentzAngle;

//
//
// class decleration
//
class SiStripLorentzAngleReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiStripLorentzAngleReader(const edm::ParameterSet&);
  ~SiStripLorentzAngleReader() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  uint32_t printdebug_;
  std::string label_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> laToken_;
};

#endif
