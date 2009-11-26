/**
 *
 * Analyzer that writes LUTs.
 *
 *\author L. Gray (4/13/06)
 *
 *****************************************************
 * 11/11/09
 * GP: added new switch to use the beam start Pt LUTs
 * if (eta > 2.1) 2 stations tracks have quality 2
 *                3 stations tracks have quality 3
 * NB: no matter if the have ME1
 * 
 * --> by default is set to false
 *****************************************************
 *
 */

#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCMakePTLUT.h"

CSCMakePTLUT::CSCMakePTLUT(edm::ParameterSet const& conf) : myTF( 0 )
{
  //writeLocalPhi = conf.getUntrackedParameter<bool>("WriteLocalPhi",true);
  station = conf.getUntrackedParameter<int>("Station",-1);
  sector = conf.getUntrackedParameter<int>("Sector",-1);
  endcap = conf.getUntrackedParameter<int>("Endcap",-1);
  binary = conf.getUntrackedParameter<bool>("BinaryOutput",true);
  isBeamStart = conf.getUntrackedParameter<bool>("BeamStartConfiguration",false);
  LUTparam = conf.getParameter<edm::ParameterSet>("lutParam");

  //init Track Finder LUTs
  //  myTF = new CSCTFPtLUT(LUTparam);
}

CSCMakePTLUT::~CSCMakePTLUT()
{
  if(myTF)
    {
      delete myTF;
      myTF = NULL;
    }
}

void CSCMakePTLUT::analyze(edm::Event const& e, edm::EventSetup const& iSetup)
{
  edm::ESHandle<CSCGeometry> pDD;

  iSetup.get<MuonGeometryRecord>().get( pDD );
  CSCTriggerGeometry::setGeometry(pDD);
  
  edm::ESHandle< L1MuTriggerScales > scales ;
  iSetup.get< L1MuTriggerScalesRcd >().get( scales ) ;

  edm::ESHandle< L1MuTriggerPtScale > ptScale ;
  iSetup.get< L1MuTriggerPtScaleRcd >().get( ptScale ) ;

  myTF = new CSCTFPtLUT(LUTparam, scales.product(), ptScale.product(), isBeamStart );

  std::string filename = std::string("L1CSCPtLUT") + ((binary) ? std::string(".bin") : std::string(".dat"));
  std::ofstream L1CSCPtLUT(filename.c_str());
  for(int i=0; i < 1<<CSCBitWidths::kPtAddressWidth; ++i)
    {
      unsigned short thedata = myTF->Pt(i).toint();
      if(binary) L1CSCPtLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
      else L1CSCPtLUT << std::dec << thedata << std::endl;
    }
}

std::string CSCMakePTLUT::fileSuffix() const {
  std::string fileName = "";
  fileName += ((binary) ? ".bin" : ".dat");
  return fileName;
}
