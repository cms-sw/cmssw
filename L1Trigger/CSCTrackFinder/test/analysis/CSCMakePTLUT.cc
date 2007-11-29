/**
 *
 * Analyzer that writes LUTs.
 *
 *\author L. Gray (4/13/06)
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

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCMakePTLUT.h"

CSCMakePTLUT::CSCMakePTLUT(edm::ParameterSet const& conf)
{
  //writeLocalPhi = conf.getUntrackedParameter<bool>("WriteLocalPhi",true);
  station = conf.getUntrackedParameter<int>("Station",-1);
  sector = conf.getUntrackedParameter<int>("Sector",-1);
  endcap = conf.getUntrackedParameter<int>("Endcap",-1);
  binary = conf.getUntrackedParameter<bool>("BinaryOutput",true);
  LUTparam = conf.getParameter<edm::ParameterSet>("lutParam");

  //init Track Finder LUTs
  myTF = new CSCTFPtLUT(LUTparam);
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
