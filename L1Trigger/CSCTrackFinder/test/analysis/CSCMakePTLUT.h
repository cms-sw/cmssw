#ifndef CSCTrackFinder_MakePTLUT_h
#define CSCTrackFinder_MakePTLUT_h

/**
 * \author L. Gray 2/26/06
 *
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

class CSCTFPtLUT;

class CSCMakePTLUT : public edm::one::EDAnalyzer<> {
public:
  explicit CSCMakePTLUT(edm::ParameterSet const& conf);
  virtual ~CSCMakePTLUT();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  //

  std::string fileSuffix() const;
  //CSCTFPtLUT* myTF[2][6][2][4]; // [Endcap][Sector][Subsector][Station]
  //bool writeLocalPhi, writeGlobalPhi, writeGlobalEta,
  bool binary;
  int endcap, sector, station;
  edm::ParameterSet LUTparam;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> scalesToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> ptScalesToken_;
};

DEFINE_FWK_MODULE(CSCMakePTLUT);

#endif
