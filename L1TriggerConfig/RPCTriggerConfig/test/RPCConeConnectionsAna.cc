// -*- C++ -*-
//
// Package:    RPCConeConnectionsAna
// Class:      RPCConeConnectionsAna
//
/**\class RPCConeConnectionsAna RPCConeConnectionsAna.cc L1TriggerConfig/RPCConeConnectionsAna/src/RPCConeConnectionsAna.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Tue Mar 18 15:15:30 CET 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"

//
// class decleration
//

class RPCConeConnectionsAna : public edm::one::EDAnalyzer<> {
public:
  explicit RPCConeConnectionsAna(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  int getDCCNumber(int iTower, int iSec);
  int getDCC(int iSec);
  void printSymetric(RPCDetId det, edm::ESHandle<RPCGeometry> rpcGeom);
  void printRoll(RPCRoll const* roll);
  int m_towerBeg;
  int m_towerEnd;
  int m_sectorBeg;
  int m_sectorEnd;

  // ----------member data ---------------------------
  edm::ESGetToken<L1RPCConeBuilder, L1RPCConeBuilderRcd> m_coneBuilderToken;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> m_rpcGeomToken;
  edm::ESGetToken<L1RPCConeDefinition, L1RPCConeDefinitionRcd> m_coneDefToken;
  edm::ESGetToken<RPCEMap, RPCEMapRcd> m_nmapToken;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCConeConnectionsAna::RPCConeConnectionsAna(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  m_towerBeg = iConfig.getParameter<int>("minTower");
  m_towerEnd = iConfig.getParameter<int>("maxTower");

  m_sectorBeg = iConfig.getParameter<int>("minSector");
  m_sectorEnd = iConfig.getParameter<int>("maxSector");

  m_coneBuilderToken = esConsumes();
  m_rpcGeomToken = esConsumes();
  m_coneDefToken = esConsumes();
  m_nmapToken = esConsumes();
}

//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void RPCConeConnectionsAna::analyze(const edm::Event& iEvent, const edm::EventSetup& evtSetup) {
  std::map<int, int> PACmap;

  edm::ESHandle<L1RPCConeBuilder> coneBuilder = evtSetup.getHandle(m_coneBuilderToken);

  edm::ESHandle<RPCGeometry> rpcGeom = evtSetup.getHandle(m_rpcGeomToken);

  edm::ESHandle<L1RPCConeDefinition> coneDef = evtSetup.getHandle(m_coneDefToken);

  edm::ESHandle<RPCEMap> nmap = evtSetup.getHandle(m_nmapToken);
  const RPCEMap* eMap = nmap.product();
  edm::ESHandle<RPCReadOutMapping> map = eMap->convert();

  for (TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin(); it != rpcGeom->dets().end(); ++it) {
    if (dynamic_cast<const RPCRoll*>(*it) == 0)
      continue;

    RPCRoll const* roll = dynamic_cast<RPCRoll const*>(*it);

    int detId = roll->id().rawId();
    //      if ( detId != 637567014) continue;
    //     if (roll->id().station() != 2 || roll->id().ring() != 2) continue;

    std::pair<L1RPCConeBuilder::TCompressedConVec::const_iterator, L1RPCConeBuilder::TCompressedConVec::const_iterator>
        compressedConnPair = coneBuilder->getCompConVec(detId);
    //iterate over strips
    for (int strip = 0; strip < roll->nstrips(); ++strip) {
      /* old 
          std::pair<L1RPCConeBuilder::TStripConVec::const_iterator, 
                    L1RPCConeBuilder::TStripConVec::const_iterator> 
                    itPair = coneBuilder->getConVec(detId, strip);*/

      //         L1RPCConeBuilder::TStripConVec::const_iterator it = itPair.first;
      //         for (; it!=itPair.second;++it){
      L1RPCConeBuilder::TCompressedConVec::const_iterator itComp = compressedConnPair.first;
      for (; itComp != compressedConnPair.second; ++itComp) {
        int logstrip = itComp->getLogStrip(strip, coneDef->getLPSizeVec());
        if (logstrip == -1)
          continue;

        // iterate over all PACs
        for (int tower = m_towerBeg; tower <= m_towerEnd; ++tower) {
          if (itComp->m_tower != tower)
            continue;

          for (int sector = m_sectorBeg; sector <= m_sectorEnd; ++sector) {
            int dccInputChannel = getDCCNumber(tower, sector);
            int PAC = sector * 12;
            int PACend = PAC + 11;

            for (; PAC <= PACend; ++PAC) {
              if (itComp->m_PAC != PAC)
                continue;
              ++PACmap[PAC];

              LinkBoardElectronicIndex a;
              std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> linkStrip =
                  std::make_pair(a, LinkBoardPackedStrip(0, 0));

              std::pair<int, int> stripInDetUnit(detId, strip);
              std::vector<std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> > aVec =
                  map->rawDataFrame(stripInDetUnit);
              std::vector<std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> >::const_iterator CI;

              for (CI = aVec.begin(); CI != aVec.end(); CI++) {
                if (CI->first.dccInputChannelNum == dccInputChannel)
                  linkStrip = *CI;
              }

              if (linkStrip.second.packedStrip() == -17) {
                std::cout << "BAD: PAC " << PAC << " tower " << tower << " detId " << detId << " strip " << strip
                          << " lp " << (int)itComp->m_logplane << " ls " << (int)logstrip << std::endl;

                std::cout << " Connected to: " << std::endl;
                for (CI = aVec.begin(); CI != aVec.end(); CI++) {
                  std::cout << "     DCC: " << CI->first.dccId << " DCCInChannel: " << CI->first.dccInputChannelNum
                            << std::endl;
                }
                std::cout << " Me thinks it should be: DCC: " << getDCC(sector) << " DCCInChannel: " << dccInputChannel
                          << std::endl;

                printRoll(roll);
                printSymetric(roll->id(), rpcGeom);

              } else {
                /*
                     std::cout<<" OK: PAC "<< PAC  << " tower "  << tower 
                              << " detId " << detId << " strip " << strip 
                              << " lp " << (int)itComp->m_logplane
                              << " ls "  << (int)logstrip
                              <<" "<< RPCDetId(detId) 
                              << std::endl;*/
              }

            }  // PAC iteration

          }  // sector iteration

        }  // tower iteration

      }  // cone connections interation

    }  // strip in roll iteration

  }  // roll iteration

  std::map<int, int>::iterator it = PACmap.begin();
  for (; it != PACmap.end(); ++it) {
    if (it->second != 8) {
      //    std::cout << "PAC " << it->first << " refcon " << it->second << std::endl;
    }
  }
}

int RPCConeConnectionsAna::getDCCNumber(int iTower, int iSec) {
  int tbNumber = 0;
  if (iTower < -12)
    tbNumber = 0;
  else if (-13 < iTower && iTower < -8)
    tbNumber = 1;
  else if (-9 < iTower && iTower < -4)
    tbNumber = 2;
  else if (-5 < iTower && iTower < -1)
    tbNumber = 3;
  else if (-2 < iTower && iTower < 2)
    tbNumber = 4;
  else if (1 < iTower && iTower < 5)
    tbNumber = 5;
  else if (4 < iTower && iTower < 9)
    tbNumber = 6;
  else if (8 < iTower && iTower < 13)
    tbNumber = 7;
  else if (12 < iTower)
    tbNumber = 8;

  int phiFactor = iSec % 4;
  return (tbNumber + phiFactor * 9);  //Count DCC input channel from 1
}

int RPCConeConnectionsAna::getDCC(int iSec) {
  int ret = 0;
  if (iSec >= 0 && iSec <= 3)
    ret = 792;
  else if (iSec >= 4 && iSec <= 7)
    ret = 791;
  else if (iSec >= 8 && iSec <= 11)
    ret = 791;

  else
    throw cms::Exception("blablabla") << "Bad ligsector:" << iSec << std::endl;

  return ret;
}

void RPCConeConnectionsAna::printSymetric(RPCDetId det, edm::ESHandle<RPCGeometry> rpcGeom) {
  RPCDetId detSym;

  if (det.region() == 0) {  // bar
    detSym = RPCDetId(0, -det.ring(), det.station(), det.sector(), det.layer(), det.subsector(), det.roll());
  } else {  // endcap
    detSym = RPCDetId(-det.region(), det.ring(), det.station(), det.sector(), det.layer(), det.subsector(), det.roll());
  }

  for (TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin(); it != rpcGeom->dets().end(); ++it) {
    if (dynamic_cast<const RPCRoll*>(*it) == 0)
      continue;
    RPCRoll const* roll = dynamic_cast<RPCRoll const*>(*it);

    if (roll->id() != detSym)
      continue;
    printRoll(roll);
  }
}

void RPCConeConnectionsAna::printRoll(const RPCRoll* roll) {
  LocalPoint lStripCentre1 = roll->centreOfStrip(1);
  LocalPoint lStripCentreMax = roll->centreOfStrip(roll->nstrips());

  GlobalPoint gStripCentre1 = roll->toGlobal(lStripCentre1);
  GlobalPoint gStripCentreMax = roll->toGlobal(lStripCentreMax);
  float phiRaw1 = gStripCentre1.phi();
  float phiRawMax = gStripCentreMax.phi();
  std::cout << roll->id().rawId() << " " << roll->id() << " - chamber spans in phi between : " << phiRaw1 << " "
            << phiRawMax << std::endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(RPCConeConnectionsAna);
