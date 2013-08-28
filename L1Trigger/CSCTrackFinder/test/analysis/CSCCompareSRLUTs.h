#ifndef CSCTrackFinder_CompareSRLUTs_h
#define CSCTrackFinder_CompareSRLUTs_h

/**
 * \author L. Gray 2/26/06
 *
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCSectorReceiverLUT;

class CSCCompareSRLUTs : public edm::EDAnalyzer {
 public:
  explicit CSCCompareSRLUTs(edm::ParameterSet const& conf);
  virtual ~CSCCompareSRLUTs();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  //virtual void endJob();
 private:
  // variables persistent across events should be declared here.
  //
  CSCSectorReceiverLUT* myLUT, *testLUT; // [Endcap][Sector][Subsector][Station]
  bool binary, isTMB07;
  int endcap, sector, station, subsector;
  edm::ParameterSet LUTparam;
};

DEFINE_FWK_MODULE(CSCCompareSRLUTs);

#endif
