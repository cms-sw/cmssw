/**
 *
 * Analyzer to compare one LUT to another and record the differences.
 * author L. Gray 4/13/06
 *
 *
 */

#include <fstream>
#include <iostream>
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
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCCompareSRLUTs.h"

CSCCompareSRLUTs::CSCCompareSRLUTs(edm::ParameterSet const& conf)
{
  station = conf.getUntrackedParameter<int>("Station",1);
  sector = conf.getUntrackedParameter<int>("Sector",1);
  subsector = conf.getUntrackedParameter<int>("SubSector",1);
  endcap = conf.getUntrackedParameter<int>("Endcap",1);
  binary = conf.getUntrackedParameter<bool>("BinaryInput",false);
  isTMB07 = conf.getUntrackedParameter<bool>("isTMB07",false);
  LUTparam = conf.getParameter<edm::ParameterSet>("lutParam");

  myLUT = new CSCSectorReceiverLUT(endcap, sector, subsector, station, edm::ParameterSet(), isTMB07);
  testLUT = new CSCSectorReceiverLUT(endcap, sector, subsector, station, LUTparam, isTMB07);
}

CSCCompareSRLUTs::~CSCCompareSRLUTs()
{
  delete myLUT;
  delete testLUT;
}

void CSCCompareSRLUTs::analyze(edm::Event const& e, edm::EventSetup const& iSetup)
{
  // set geometry pointer
  edm::ESHandle<CSCGeometry> pDD;

  iSetup.get<MuonGeometryRecord>().get( pDD );
  CSCTriggerGeometry::setGeometry(pDD);

  // test local phi
  // should match for all inputs
  for(unsigned int address = 0; address < 1<<CSCBitWidths::kLocalPhiAddressWidth; ++address)
    {
      unsigned short mLUT, tstLUT;
      lclphidat mout, tstout;
      mout = myLUT->localPhi(address);
      tstout = testLUT->localPhi(address);
      mLUT = mout.toint();
      tstLUT = tstout.toint();
      if(mLUT != tstLUT)
	edm::LogInfo("mismatch|LocalPhi") << mLUT<< " != " << tstLUT;
    }

  double ntoteta = 0.0, nmatcheta = 0.0, nwithinoneeta = 0.0;
  double ntotphi = 0.0, nmatchphi = 0.0, nwithinonephi = 0.0;

  //test global phi
  // should match (or be close) for all valid inputs.

  for(int c = CSCTriggerNumbering::minTriggerCscId(); c <= CSCTriggerNumbering::maxTriggerCscId(); ++c)
    {
      gblphidat mgPhi, tstgPhi;

      CSCTriggerGeomManager* geom = CSCTriggerGeometry::get();
      CSCChamber* thechamber = geom->chamber(endcap, station, sector, subsector, c);

      if(thechamber)
        {
          const CSCLayerGeometry* layergeom = thechamber->layer(3)->geometry();
          int nWireGroups = layergeom->numberOfWireGroups();

	  for(int wg = 0; wg < nWireGroups; ++wg)
	    for(int phil = 0; phil < 1<<CSCBitWidths::kLocalPhiDataBitWidth; ++phil)
	      {

                ntotphi += 1.0;

                mgPhi = myLUT->globalPhiME(phil, wg, c);
                tstgPhi = testLUT->globalPhiME(phil, wg, c);
                unsigned short my, test;

                my = mgPhi.toint();
                test = tstgPhi.toint();

                if( my == test )
                  nmatchphi += 1.0;
                else if( my <= test + 5 && my >= test - 5 )
                  nwithinonephi += 1.0;
                else
                  {
		    edm::LogInfo("mismatch:GlobalPhi") << c << ' ' << wg << ' ' << phil << ' ' << 0 << ' ' << my << " != " << test;
                  }

              }
        }
    }

  //test global eta
  // should match (or be close) for all valid inputs.
  for(int c = CSCTriggerNumbering::minTriggerCscId(); c <= CSCTriggerNumbering::maxTriggerCscId(); ++c)
    {
      gbletadat mgEta, tstgEta;

      CSCTriggerGeomManager* geom = CSCTriggerGeometry::get();
      CSCChamber* thechamber = geom->chamber(endcap, station, sector, subsector, c);

      if(thechamber)
	{
	  const CSCLayerGeometry* layergeom = thechamber->layer(3)->geometry();
	  int nWireGroups = layergeom->numberOfWireGroups();

	  for(int wg = 0; wg < nWireGroups; ++wg)
	    for(int phil = 0; phil < 4; ++phil)
	      {

		ntoteta += 1.0;

		mgEta = myLUT->globalEtaME(0, (phil << 8), wg, c);
		tstgEta = testLUT->globalEtaME(0, (phil << 8), wg, c);
		unsigned short my, test;

		my = mgEta.toint();
		test = tstgEta.toint();

		if( my == test )
		  nmatcheta += 1.0;
		else if( my == test + 1 || my == test - 1 )
		  nwithinoneeta += 1.0;
		else
                  {
		    std::cout << mgEta.global_eta << std::endl;
		    edm::LogInfo("mismatch:GlobalEta") << c << ' ' << wg << ' ' << phil << ' ' << 0 << ' ' << my << " != " << test;
                  }

	      }
	}
    }
  edm::LogInfo("MatchingInfo") << "PERCENT MATCHING GLOBAL PHI VALUES: "<< nmatchphi/ntotphi;
  edm::LogInfo("MatchingInfo") << "PERCENT WITHIN .105 DEG: " << nwithinonephi/ntotphi;
  edm::LogInfo("MatchingInfo") << "PERCENT GPHI ACCEPTABLE: " << (nmatchphi + nwithinonephi)/ntotphi;

  edm::LogInfo("MatchingInfo") << "PERCENT MATCHING GLOBAL ETA VALUES: "<< nmatcheta/ntoteta;
  edm::LogInfo("MatchingInfo") << "PERCENT WITHIN ONE ETA UNIT: " << nwithinoneeta/ntoteta;
  edm::LogInfo("MatchingInfo") << "PERCENT GETA ACCEPTABLE: " << (nmatcheta + nwithinoneeta)/ntoteta;

}


