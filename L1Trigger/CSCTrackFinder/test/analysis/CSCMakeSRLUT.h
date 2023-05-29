#ifndef CSCTrackFinder_MakeLUT_h
#define CSCTrackFinder_MakeLUT_h

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

class CSCSectorReceiverLUT;

class CSCMakeSRLUT : public edm::one::EDAnalyzer<> {
public:
  explicit CSCMakeSRLUT(edm::ParameterSet const& conf);
  virtual ~CSCMakeSRLUT();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  //

  std::string fileSuffix() const;
  CSCSectorReceiverLUT* mySR[2][6][2][4];  // [Endcap][Sector][Subsector][Station]
  bool writeLocalPhi, writeGlobalPhi, writeGlobalEta, binary;
  int endcap, sector, station, isTMB07;
  edm::ParameterSet LUTparam;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geomToken_;
};

DEFINE_FWK_MODULE(CSCMakeSRLUT);

#endif
