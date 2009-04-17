/*
 *	HaloTrigger.cc
 *	Created by Joseph Gartner on 12/17/08
 *	Please use this code for good, not evil
 */

#include "FWCore/Framework/interface/TriggerNames.h"

#include "HLTriggerOffline/special/src/HaloTrigger.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace edm;
using namespace trigger;

HaloTrigger::HaloTrigger(const ParameterSet& ps)
{
//	dbe = NULL;
//	if( ps.getUntrackedParameter<bool>("DQMStore", false) )
//	{
		dbe = Service<DQMStore>().operator->();
		dbe->setVerbose(0);
//	}

	first = false;
	outFile = ps.getUntrackedParameter<string>("outFile");
	
	HLTriggerTag = ps.getParameter< edm::InputTag >("HLTriggerTag");
	GMTInputTag = ps.getParameter< edm::InputTag >("GMTInputTag");
	lctProducer = ps.getParameter< edm::InputTag >("LCTInputTag");
	cscRecHitLabel = ps.getParameter< edm::InputTag >("RecInputTag");

}

HaloTrigger::~HaloTrigger()
{}

void HaloTrigger::beginJob(const EventSetup& es)
{
	/////////////////////////
	// Initialize Counters //
	/////////////////////////
	hltHaloTriggers = 0;
	hltHaloOver1 = 0;
	hltHaloOver2 = 0;
	hltHaloRing23 = 0;
	CscHalo_Gmt = 0;	
	
	/////////////////////////////
	// DQM Directory Structure //
	/////////////////////////////
	DQMStore * dbe = 0;
	dbe = Service<DQMStore>().operator->();
	if( dbe ){
		dbe->setCurrentFolder("HLTriggerOffline/special");
		
		//////////////////////////////////////////
		// Define Monitor Elements a.k.a. Plots //
		//////////////////////////////////////////
		PlusMe1BeamHaloOcc = dbe->book2D("PlusMe1BeamHaloOcc","Positive Endcap, Me1 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		PlusMe1BeamHaloOccRing1 = dbe->book2D("PlusMe1BeamHaloOccRing1","Positive Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		PlusMe1BeamHaloOccRing2 = dbe->book2D("PlusMe1BeamHaloOccRing2","Positive Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		PlusMe1BeamHaloOccRing2or3 = dbe->book2D("PlusMe1BeamHaloOccRing2or3","Positive Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
		PlusMe1BeamHaloOccRad = dbe->book1D("PlusMe1BeamHaloOccRad","Positive Endcap, Me1 Beam Halo Radial Occupancy",750,0,750);
	
		PlusMe2BeamHaloOcc = dbe->book2D("PlusMe2BeamHaloOcc","Positive Endcap, Me2 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		PlusMe2BeamHaloOccRing1 = dbe->book2D("PlusMe2BeamHaloOccRing1","Positive Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		PlusMe2BeamHaloOccRing2 = dbe->book2D("PlusMe2BeamHaloOccRing2","Positive Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		PlusMe2BeamHaloOccRing2or3 = dbe->book2D("PlusMe2BeamHaloOccRing2or3","Positive Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
	
		PlusMe3BeamHaloOcc = dbe->book2D("PlusMe3BeamHaloOcc","Positive Endcap, Me3 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		PlusMe3BeamHaloOccRing1 = dbe->book2D("PlusMe3BeamHaloOccRing1","Positive Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		PlusMe3BeamHaloOccRing2 = dbe->book2D("PlusMe3BeamHaloOccRing2","Positive Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		PlusMe3BeamHaloOccRing2or3 = dbe->book2D("PlusMe3BeamHaloOccRing2or3","Positive Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
	
		PlusMe4BeamHaloOcc = dbe->book2D("PlusMe4BeamHaloOcc","Positive Endcap, Me4 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		PlusMe4BeamHaloOccRing1 = dbe->book2D("PlusMe4BeamHaloOccRing1","Positive Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		PlusMe4BeamHaloOccRing2 = dbe->book2D("PlusMe4BeamHaloOccRing2","Positive Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		PlusMe4BeamHaloOccRing2or3 = dbe->book2D("PlusMe4BeamHaloOccRing2or3","Positive Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
	
		MinusMe1BeamHaloOcc = dbe->book2D("MinusMe1Me1BeamHaloOcc","Negative Endcap, Me1 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		MinusMe1BeamHaloOccRing1 = dbe->book2D("MinusBeamHaloOccRing1","Negative Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		MinusMe1BeamHaloOccRing2 = dbe->book2D("MinusMe1BeamHaloOccRing2","Negative Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		MinusMe1BeamHaloOccRing2or3 = dbe->book2D("MinusMe1BeamHaloOccRing2or3","Negative Endcap, Me1 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
		MinusMe1BeamHaloOccRad = dbe->book1D("MinusMe1BeamHaloOccRad","Minus Endcap, Me1 Beam Halo Radial Occupancy",750,0,750);
		
		MinusMe2BeamHaloOcc = dbe->book2D("MinusMe2BeamHaloOcc","Negative Endcap, Me2 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		MinusMe2BeamHaloOccRing1 = dbe->book2D("MinusMe2BeamHaloOccRing1","Negative Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		MinusMe2BeamHaloOccRing2 = dbe->book2D("MinusMe2BeamHaloOccRing2","Negative Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		MinusMe2BeamHaloOccRing2or3 = dbe->book2D("MinusMe2BeamHaloOccRing2or3","Negative Endcap, Me2 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
		
		MinusMe3BeamHaloOcc = dbe->book2D("MinusMe3BeamHaloOcc","Negative Endcap, Me3 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		MinusMe3BeamHaloOccRing1 = dbe->book2D("MinusMe3BeamHaloOccRing1","Negative Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring1",1000,-500,500,1000,-500,500);
		MinusMe3BeamHaloOccRing2 = dbe->book2D("MinusMe3BeamHaloOccRing2","Negative Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		MinusMe3BeamHaloOccRing2or3 = dbe->book2D("MinusMe3BeamHaloOccRing2or3","Negative Endcap, Me3 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
		
		MinusMe4BeamHaloOcc = dbe->book2D("MinusMe4BeamHaloOcc","Negative Endcap, Me4 Beam Halo Occupancy",1000,-500,500,1000,-500,500);
		MinusMe4BeamHaloOccRing1 = dbe->book2D("MinusMe4BeamHaloOccRing1","Negative Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring1",1000,-500.,500.,1000,-500.,500.);
		MinusMe4BeamHaloOccRing2 = dbe->book2D("MinusMe4BeamHaloOccRing2","Negative Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring2",1000,-500,500,1000,-500,500);
		MinusMe4BeamHaloOccRing2or3 = dbe->book2D("MinusMe4BeamHaloOccRing2or3","Negative Endcap, Me4 Beam Halo Occupancy, HLT Trig Ring 2 or 3",1000,-500,500,1000,-500,500);
	}
	
	es.get<MuonGeometryRecord>().get(m_cscGeometry);
}

void HaloTrigger::endJob(void)
{		
	dbe->save(outFile);	
	return;
}

void HaloTrigger::analyze(const Event& e, const EventSetup& es)
{		
	if( HLTriggerTag.label() != "null" )
	{
		edm::Handle<TriggerResults> trh;
		e.getByLabel(HLTriggerTag,trh);
		edm::TriggerNames names;
		
		//////////////////////////
		// Initialize HLT Paths //
		//////////////////////////
		if (!first)
		{
			first = true;
			names.init(*trh);	
			Namen = names.triggerNames();
			const unsigned int n(Namen.size());
			std::string searchName_Halo0 ("HLT_CSCBeamHalo");
			std::string searchName_Halo1 ("HLT_CSCBeamHaloOverlapRing1");
			std::string searchName_Halo2 ("HLT_CSCBeamHaloOverlapRing2");
			std::string searchName_Halo3 ("HLT_CSCBeamHaloRing2or3");
			for( unsigned int i=0; i!=n; ++i){
				int compy = Namen[i].compare(searchName_Halo0);
				if( compy == 0) majikNumber0 = i;
				
				compy = Namen[i].compare(searchName_Halo1);
				if( compy == 0) majikNumber1 = i;
				
				compy = Namen[i].compare(searchName_Halo2);
				if( compy == 0) majikNumber2 = i;
				
				compy = Namen[i].compare(searchName_Halo3);
				if( compy == 0) majikNumber3 = i;
			}// loop over all trigger indices searching for HLT of interest
		}
		
		//////////////////////
		// ID Halo Triggers //
		//////////////////////
		bool haloTriggerAny = false;
		bool halo0 = false, halo1 = false, halo2 = false, halo3 = false;
		const unsigned int n(Namen.size());
		for( unsigned int j=0; j!=n; ++j)
		{
			if( (j == majikNumber0) && (trh->accept(j)) )
			{
				halo0 = true;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber1) && (trh->accept(j)) )
			{
				halo1 = true;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber2) && (trh->accept(j)) )
			{
				halo2 = true;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber3) && (trh->accept(j)) )
			{
				halo3 = true;
				haloTriggerAny = true;
			}
		}// loop over all triggers in every event, check if path was run
		
		//////////////////
		// Rec Hit Info //
		//////////////////
		if( haloTriggerAny == true )
		{
			float cscHit[8][5];
			for(int j = 1; j < 9; j++) cscHit[j][1] = 0;
			
			edm::Handle<CSCRecHit2DCollection> cscRecHits;
			e.getByLabel(cscRecHitLabel, cscRecHits);
			CSCRecHit2DCollection::const_iterator hit = cscRecHits->begin();
			for(; hit != cscRecHits->end(); hit++)
			{
				LocalPoint p = hit->localPosition();
				CSCDetId id((hit)->geographicalId().rawId());
				GlobalPoint gP = m_cscGeometry->idToDet(id)->toGlobal(hit->localPosition());
				
				int hitIndex = id.station() + 5*(id.endcap() - 1);
				if( cscHit[hitIndex][1] == 0 )
				{
					cscHit[hitIndex][1] = 1;
					cscHit[hitIndex][2] = gP.x();
					cscHit[hitIndex][3] = gP.y();
					cscHit[hitIndex][4] = gP.z();
					cscHit[hitIndex][5] = sqrt( gP.x()*gP.x() + gP.y()*gP.y() );
					
				}//cscHit[x][1] == 0
			}//2d rec hit loop
			
			/////////////////////
			// Fill Histograms //
			/////////////////////
			if(halo0 == true)
			{
				if(cscHit[1][1] == 1)
				{ 
					PlusMe1BeamHaloOcc->Fill(cscHit[1][2],cscHit[1][3]);
					PlusMe1BeamHaloOccRad->Fill(cscHit[1][5]);
				}
				if(cscHit[2][1] == 1) PlusMe2BeamHaloOcc->Fill(cscHit[2][2],cscHit[2][3]);
				if(cscHit[3][1] == 1) PlusMe3BeamHaloOcc->Fill(cscHit[3][2],cscHit[3][3]);
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOcc->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1)
				{
					MinusMe1BeamHaloOcc->Fill(cscHit[5][2],cscHit[5][3]);
					MinusMe1BeamHaloOccRad->Fill(cscHit[5][5]);
				}
				if(cscHit[6][1] == 1) MinusMe2BeamHaloOcc->Fill(cscHit[6][2],cscHit[6][3]);
				if(cscHit[7][1] == 1) MinusMe3BeamHaloOcc->Fill(cscHit[7][2],cscHit[7][3]);
				if(cscHit[8][1] == 1) MinusMe4BeamHaloOcc->Fill(cscHit[8][2],cscHit[8][3]); 
		
			}//HLT_CSCBeamHalo
			
			if(halo1 == true)
			{
				if(cscHit[1][1] == 1) PlusMe1BeamHaloOccRing1->Fill(cscHit[1][2],cscHit[1][3]);
				if(cscHit[2][1] == 1) PlusMe2BeamHaloOccRing1->Fill(cscHit[2][2],cscHit[2][3]);
				if(cscHit[3][1] == 1) PlusMe3BeamHaloOccRing1->Fill(cscHit[3][2],cscHit[3][3]);
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOccRing1->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1) MinusMe1BeamHaloOccRing1->Fill(cscHit[5][2],cscHit[5][3]);
				if(cscHit[6][1] == 1) MinusMe2BeamHaloOccRing1->Fill(cscHit[6][2],cscHit[6][3]);
				if(cscHit[7][1] == 1) MinusMe3BeamHaloOccRing1->Fill(cscHit[7][2],cscHit[7][3]);
				if(cscHit[8][1] == 1) MinusMe4BeamHaloOccRing1->Fill(cscHit[8][2],cscHit[8][3]);
			}//HLT_CSCBeamHalo
			
			if(halo2 == true)
			{
				if(cscHit[1][1] == 1) PlusMe1BeamHaloOccRing2->Fill(cscHit[1][2],cscHit[1][3]);
				if(cscHit[2][1] == 1) PlusMe2BeamHaloOccRing2->Fill(cscHit[2][2],cscHit[2][3]);
				if(cscHit[3][1] == 1) PlusMe3BeamHaloOccRing2->Fill(cscHit[3][2],cscHit[3][3]);
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOccRing2->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1) MinusMe1BeamHaloOccRing2->Fill(cscHit[5][2],cscHit[5][3]);
				if(cscHit[6][1] == 1) MinusMe2BeamHaloOccRing2->Fill(cscHit[6][2],cscHit[6][3]);
				if(cscHit[7][1] == 1) MinusMe3BeamHaloOccRing2->Fill(cscHit[7][2],cscHit[7][3]);
				if(cscHit[8][1] == 1) MinusMe4BeamHaloOccRing2->Fill(cscHit[8][2],cscHit[8][3]);
			}//HLT_CSCBeamHalo
			
			if(halo3 == true)
			{
				if(cscHit[1][1] == 1) PlusMe1BeamHaloOccRing2or3->Fill(cscHit[1][2],cscHit[1][3]);
				if(cscHit[2][1] == 1) PlusMe2BeamHaloOccRing2or3->Fill(cscHit[2][2],cscHit[2][3]);
				if(cscHit[3][1] == 1) PlusMe3BeamHaloOccRing2or3->Fill(cscHit[3][2],cscHit[3][3]);
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOccRing2or3->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1) MinusMe1BeamHaloOccRing2or3->Fill(cscHit[5][2],cscHit[5][3]);
				if(cscHit[6][1] == 1) MinusMe2BeamHaloOccRing2or3->Fill(cscHit[6][2],cscHit[6][3]);
				if(cscHit[7][1] == 1) MinusMe3BeamHaloOccRing2or3->Fill(cscHit[7][2],cscHit[7][3]);
				if(cscHit[8][1] == 1) MinusMe4BeamHaloOccRing2or3->Fill(cscHit[8][2],cscHit[8][3]);
			}//HLT_CSCBeamHalo
		}//haloTrigger true
	}//HLTriggerTag != null
}
