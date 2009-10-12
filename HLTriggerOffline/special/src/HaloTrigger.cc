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
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

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
	
		MinusMe3BeamHaloOccRing2Unfold = dbe->book2D("MinusMe3BeamHaloOccRing2Unfold","Negative Endcap Me3, Ring 2 Unfolded",120,-1,11,500,-2.4,-0.9);
		MinusMe2BeamHaloOccRing2Unfold = dbe->book2D("MinusMe2BeamHaloOccRing2Unfold","Negative Endcap Me2, Ring 2 Unfolded",120,-1,11,500,-2.4,-0.9);
		PlusMe3BeamHaloOccRing2Unfold = dbe->book2D("PlusMe3BeamHaloOccRing2Unfold","Negative Endcap Me3, Ring 2 Unfolded",120,-1,11,500,0.9,2.4);
		PlusMe2BeamHaloOccRing2Unfold = dbe->book2D("PlusMe2BeamHaloOccRing2Unfold","Negative Endcap Me2, Ring 2 Unfolded",120,-1,11,500,0.9,2.4);
		
		MinusMe3BeamHaloOccRing1Unfold = dbe->book2D("MinusMe3BeamHaloOccRing1Unfold","Negative Endcap Me3, Ring 1 Unfolded",120,-1,11,500,-2.5,-0.9);
		MinusMe2BeamHaloOccRing1Unfold = dbe->book2D("MinusMe2BeamHaloOccRing1Unfold","Negative Endcap Me2, Ring 1 Unfolded",120,-1,11,500,-2.5,-0.9);
		PlusMe3BeamHaloOccRing1Unfold = dbe->book2D("PlusMe3BeamHaloOccRing1Unfold","Negative Endcap Me3, Ring 1 Unfolded",120,-1,11,500,0.9,2.5);
		PlusMe2BeamHaloOccRing1Unfold = dbe->book2D("PlusMe2BeamHaloOccRing1Unfold","Negative Endcap Me2, Ring 1 Unfolded",120,-1,11,500,0.9,2.5);
	
		MinusEff = dbe->bookProfile("MinusEff", "Minus Endcap Efficiency by Vertex",50,0,750,50,0,1);
		MinusEff->setAxisTitle("Radius",1);
		MinusEff->setAxisTitle("Efficiency",2);
		//MinusEffProj3 = dbe->bookProfile("MinusEffProj3", "Minus Endcap Efficiency Projected to statoin 3",50,0,750,50,0,1);
		MinusEffProj3 = dbe->book1D("MinusEffProj3","Minus Endcap Efficiency by projection",50,0,750);
		MinusEffProj3->setBinLabel(1,"0",1);
		MinusEffProj3->setBinLabel(10,"150",1);
		MinusEffProj3->setBinLabel(20,"300",1);
		MinusEffProj3->setBinLabel(30,"450",1);
		MinusEffProj3->setBinLabel(40,"600",1);
		MinusEffProj3->setBinLabel(50,"750",1);
		MinusEffProj3->setAxisTitle("Radius",1);
		MinusEffProj3->setAxisTitle("Efficiency",2);
		MinusEffNum = dbe->book1D("MinusEffNum", "Minus Endcap numerator",50,0,750);
		MinusEffDen = dbe->book1D("MinusEffDen", "Minus Endcap denomenator", 50, 0, 750);
		PlusEff  = dbe->bookProfile("PlusEff",  "Plus Endcap Efficiency by Vertex",50,0,750,50,0,1);
		PlusEff->setAxisTitle("Radius",1);
		PlusEff->setAxisTitle("Efficiency",2);
		//PlusEffProj3  = dbe->bookProfile("PlusEffProj3",  "Plus Endcap Efficiency Projected to station 3",50,0,750,50,0,1);
		PlusEffProj3 = dbe->book1D("PlusEffProj3","Plus Endcap Efficiency by projection",50,0,750);
		PlusEffProj3->setAxisTitle("Radius",1);
		PlusEffProj3->setAxisTitle("Efficiency",2);
		PlusEffProj3->setBinLabel(1,"0",1);
		PlusEffProj3->setBinLabel(10,"150",1);
		PlusEffProj3->setBinLabel(20,"300",1);
		PlusEffProj3->setBinLabel(30,"450",1);
		PlusEffProj3->setBinLabel(40,"600",1);
		PlusEffProj3->setBinLabel(50,"750",1);
		PlusEffNum = dbe->book1D("PlusEffNum", "Plus Endcap numerator",50,0,750);
		PlusEffDen = dbe->book1D("PlusEffDen", "Plus Endcap denomenator", 50, 0, 750);
	}
	
	es.get<MuonGeometryRecord>().get(m_cscGeometry);
	for( int i=0; i<50; ++i)
	{
		numCountPlus[i] = 0;
		denCountPlus[i] = 0;
		numCountMinus[i] = 0;
		denCountMinus[i] = 0;
	}
}

void HaloTrigger::endJob(void)
{
	for(int j=1;j<51;j++)
	{
		if(denCountPlus[j] != 0)
		{
			float fillValuePlus = numCountPlus[j]/denCountPlus[j];
			PlusEffProj3->setBinContent(j,fillValuePlus);
		}
		
		if(denCountMinus[j] != 0)
		{
			float fillValueMinus = numCountMinus[j]/denCountMinus[j];
			MinusEffProj3->setBinContent(j,fillValueMinus);
		}
	}
		


	dbe->save(outFile);	
	return;
}

void HaloTrigger::analyze(const Event& e, const EventSetup& es)
{
	edm::Handle<edm::SimVertexContainer> simvertices_handle;
	e.getByLabel("g4SimHits",simvertices_handle);
    edm::SimVertexContainer const* simvertices = simvertices_handle.product();
 
    edm::Handle<edm::SimTrackContainer> simtracks_handle;
    e.getByLabel("g4SimHits",simtracks_handle);
    edm::SimTrackContainer const* simtracks = simtracks_handle.product();

	float vtxArray[7];
	vtxArray[1] = 0;
	vtxArray[6] = 0;
    edm::SimTrackContainer::const_iterator isimtr;
	for(isimtr=simtracks->begin(); isimtr!=simtracks->end(); isimtr++) {
		if( (*simvertices)[(*isimtr).vertIndex()].position().z() > 2299 ){
			vtxArray[1] = 1;
			vtxArray[2] = (*simvertices)[(*isimtr).vertIndex()].position().x();
			vtxArray[3] = (*simvertices)[(*isimtr).vertIndex()].position().y();
			vtxArray[4] = (*isimtr).momentum().px();
			vtxArray[5] = (*isimtr).momentum().py();
			vtxArray[6] = (*isimtr).momentum().pz();
		}
		
		if( (*simvertices)[(*isimtr).vertIndex()].position().z() < -2299 ){
			vtxArray[1] = -1;
			vtxArray[2] = (*simvertices)[(*isimtr).vertIndex()].position().x();
			vtxArray[3] = (*simvertices)[(*isimtr).vertIndex()].position().y();
			vtxArray[4] = (*isimtr).momentum().px();
			vtxArray[5] = (*isimtr).momentum().py();
			vtxArray[6] = (*isimtr).momentum().pz();
		}
	}

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
		
		float radius = sqrt( vtxArray[2]*vtxArray[2] + vtxArray[3]*vtxArray[3] );
		
		if( vtxArray[1] == 1){
			float projX = ((vtxArray[4]/vtxArray[6])*(-1375)) + vtxArray[2];
			float projY = ((vtxArray[5]/vtxArray[6])*(-1375)) + vtxArray[3];
			float projRad = sqrt(projX*projX + projY*projY);
			int radInd = 0;
			for(int j=0; j<50; j++)
			{ 
				if((projRad >= (j)*15.0) && (projRad < (j+1)*15)) radInd = j;
			}
			PlusEffDen->Fill(radius);
			if(haloTriggerAny == true){
				PlusEff->Fill(radius,1);
				PlusEffNum->Fill(radius);
				if((vtxArray[6] < 10) && (radInd != 0))
				{
					numCountPlus[radInd]+=1.0;
					denCountPlus[radInd]+=1.0;
				}
			} else {
				PlusEff->Fill(radius,0);
				if((vtxArray[6] < 50) && (radInd !=0))
				{
					denCountPlus[radInd]+=1.0;
				}
			}
		}
		
		if( vtxArray[1] == -1){
			float projX = ((vtxArray[4]/vtxArray[6])*(1375)) + vtxArray[2];
			float projY = ((vtxArray[5]/vtxArray[6])*(1375)) + vtxArray[3];
			float projRad = sqrt(projX*projX + projY*projY);
			int radInd = 0;
			for(int j=0; j<50; j++) 
				if((projRad >= (j)*15.0) && (projRad < (j+1)*15)) radInd = j;
			MinusEffDen->Fill(radius);
			if(haloTriggerAny == true){
				MinusEff->Fill(radius,1);
				if((vtxArray[6] > 50) && (radInd != 0)){
					numCountMinus[radInd]+=1.0;
					denCountMinus[radInd]+=1.0;
				}
				MinusEffNum->Fill(radius);
			} else {
				MinusEff->Fill(radius,0);
				if((vtxArray[6] > 10) && (radInd != 0))
				{
					denCountMinus[radInd]+=1.0;
				}
			}
		}
		
		//////////////////
		// Rec Hit Info //
		//////////////////
		if( haloTriggerAny == true )
		{
			float cscHit[9][8];
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
					float joe1 = ((gP.phi()+3.2)*180*100/(3.14));
					int joe2 = int(joe1) % 1000;
					float joe3 = (1.0*joe2)/100;
					cscHit[hitIndex][6] = joe3;
					cscHit[hitIndex][7] = gP.eta();
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
				if(cscHit[2][1] == 1)
				{
					PlusMe2BeamHaloOccRing1->Fill(cscHit[2][2],cscHit[2][3]);
					PlusMe2BeamHaloOccRing1Unfold->Fill(cscHit[2][6],cscHit[2][7]);
				}
				if(cscHit[3][1] == 1)
				{
					PlusMe3BeamHaloOccRing1->Fill(cscHit[3][2],cscHit[3][3]);
					PlusMe3BeamHaloOccRing1Unfold->Fill(cscHit[3][6],cscHit[3][7]);
				}
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOccRing1->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1) MinusMe1BeamHaloOccRing1->Fill(cscHit[5][2],cscHit[5][3]);
				if(cscHit[6][1] == 1)
				{
					MinusMe2BeamHaloOccRing1->Fill(cscHit[6][2],cscHit[6][3]);
					MinusMe2BeamHaloOccRing1Unfold->Fill(cscHit[6][6],cscHit[6][7]);
				}
				if(cscHit[7][1] == 1)
				{
					MinusMe3BeamHaloOccRing1->Fill(cscHit[7][2],cscHit[7][3]);
					MinusMe3BeamHaloOccRing1Unfold->Fill(cscHit[7][6],cscHit[7][7]);
				}
				if(cscHit[8][1] == 1) MinusMe4BeamHaloOccRing1->Fill(cscHit[8][2],cscHit[8][3]);
			}//HLT_CSCBeamHalo
			
			if(halo2 == true)
			{
				if(cscHit[1][1] == 1) PlusMe1BeamHaloOccRing2->Fill(cscHit[1][2],cscHit[1][3]);
				if(cscHit[2][1] == 1) 
				{
					PlusMe2BeamHaloOccRing2->Fill(cscHit[2][2],cscHit[2][3]);
					PlusMe2BeamHaloOccRing2Unfold->Fill(cscHit[2][6],cscHit[2][7]);
				}
				if(cscHit[3][1] == 1)
				{
					PlusMe3BeamHaloOccRing2->Fill(cscHit[3][2],cscHit[3][3]);
					PlusMe3BeamHaloOccRing2Unfold->Fill(cscHit[2][6],cscHit[2][7]);
				}
				if(cscHit[4][1] == 1) PlusMe4BeamHaloOccRing2->Fill(cscHit[4][2],cscHit[4][3]);
				if(cscHit[5][1] == 1) MinusMe1BeamHaloOccRing2->Fill(cscHit[5][2],cscHit[5][3]);
				if(cscHit[6][1] == 1)
				{
					MinusMe2BeamHaloOccRing2->Fill(cscHit[6][2],cscHit[6][3]);
					MinusMe2BeamHaloOccRing2Unfold->Fill(cscHit[6][6],cscHit[6][7]);
				}
				if(cscHit[7][1] == 1)
				{
					MinusMe3BeamHaloOccRing2->Fill(cscHit[7][2],cscHit[7][3]);
					MinusMe3BeamHaloOccRing2Unfold->Fill(cscHit[7][6],cscHit[7][7]);
				}
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
