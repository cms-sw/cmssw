#ifndef CalibTracker_SiPixelESProducers_SiPixelFakeLorentzAngleESSource_h
#define CalibTracker_SiPixelESProducers_SiPixelFakeLorentzAngleESSource_h
// -*- C++ -*-
//
// Package:    SiPixelFakeLorentzAngleESSource
// Class:      SiPixelFakeLorentzAngleESSource
//
/**\class SiPixelFakeLorentzAngleESSource SiPixelFakeLorentzAngleESSource.h CalibTracker/SiPixelGainESProducer/src/SiPixelFakeLorentzAngleESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lotte Wilke
//         Created:  Jan. 31st, 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
//
// class decleration
//

class SiPixelFakeLorentzAngleESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeLorentzAngleESSource(const edm::ParameterSet &);
  ~SiPixelFakeLorentzAngleESSource() override = default;
  virtual std::unique_ptr<SiPixelLorentzAngle> produce(const SiPixelLorentzAngleRcd &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

private:
  int HVgroup(int panel, int module);

  // data members
  const edm::FileInPath fp_;
  const edm::FileInPath t_topo_fp_;
  const std::string myLabel_;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;
  Parameters ModuleParameters_;

  float bPixLorentzAnglePerTesla_;
  float fPixLorentzAnglePerTesla_;
};
#endif
