/*
 *	HaloTrigger.cc
 *	Created by Joseph Gartner on 12/17/08
 *	Please use this code for good, not evil
 */

#include "FWCore/Framework/interface/TriggerNames.h"

#include "HLTriggerOffline/special/src/HaloTrigger.h"

#include "DQMServices/Core/interface/DQMStore.h"

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
	
	bzero(srLUTs_,sizeof(srLUTs_));
  	int endcap=1, sector=1; // assume SR LUTs are all same for every sector in either of endcaps
  	bool TMB07=true; // specific TMB firmware
  	// Create a dumy pset for SR LUTs
  	edm::ParameterSet srLUTset;
  	srLUTset.addUntrackedParameter<bool>("ReadLUTs", false);
  	srLUTset.addUntrackedParameter<bool>("Binary",   false);
  	srLUTset.addUntrackedParameter<std::string>("LUTPath", "./");
  	for(int station=1,fpga=0; station<=4 && fpga<5; station++)
	{
    	if(station==1)
        	for(int subSector=0; subSector<2 && fpga<5; subSector++)
           		srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
     	else
			srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
  	}

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
		TriggerChainEff = dbe->book1D("HltL1CscTrue","Event Count of various triggers",9,-0.5,8.5);
		TriggerChainEff->setAxisTitle("Trigger Type",1);
		TriggerChainEff->setAxisTitle("Efficiency given GMT Halo Trigger",2);
		TriggerChainEff->setBinLabel(2,"HLT_CSCBeamHalo",   1);
		TriggerChainEff->setBinLabel(4,"HLT_CSCBeamHaloOverlapRing1",1);
		TriggerChainEff->setBinLabel(6,"HLT_CSCBeamHaloOverlapRing2", 1);
		TriggerChainEff->setBinLabel(8,"HLT_CSCBeamHaloRing2or3",1);
		
		haloDelEta23 = dbe->book1D("Halo_Eta23","Change in station 3 to station 2 Eta for Halo Muons", 40, -0.20,0.30);
		haloDelPhi23 = dbe->book1D("Halo_Phi23","Change in station 3 to station 2 Phi for Halo Muons", 320, -16.0, 16.0);
	}
}

void HaloTrigger::endJob(void)
{
	float b;

	if(CscHalo_Gmt != 0)
	{
		b =	(hltHaloTriggers*1.0)/(CscHalo_Gmt*1.0);
		TriggerChainEff->setBinContent(2,b);
		TriggerChainEff->setBinLabel(2,"HLT_CSCHalo",1);
		b =	(hltHaloOver1*1.0)/(CscHalo_Gmt*1.0);
		TriggerChainEff->setBinContent(4,b);
		TriggerChainEff->setBinLabel(4,"HLT_CSCHaloOverlap1",1);
		b =	(hltHaloOver2*1.0)/(CscHalo_Gmt*1.0);
		TriggerChainEff->setBinContent(6,b);
		TriggerChainEff->setBinLabel(6,"HLT_CSCHaloOverlap2",1);
		b =	(hltHaloRing23*1.0)/(CscHalo_Gmt*1.0);
		TriggerChainEff->setBinContent(8,b);
		TriggerChainEff->setBinLabel(8,"HLT_CSCHaloRing2or3",1);
	}
		
	dbe->save(outFile);	
	return;
}

void HaloTrigger::analyze(const Event& e, const EventSetup& es)
{	
	bool haloTriggerAny = false;	
	if( HLTriggerTag.label() != "null" )
	{
		edm::Handle<TriggerResults> trh;
		e.getByLabel(HLTriggerTag,trh);
		edm::TriggerNames names;
		
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
		
		const unsigned int n(Namen.size());
		for( unsigned int j=0; j!=n; ++j){
			if( (j == majikNumber0) && (trh->accept(j)) )
			{
				hltHaloTriggers += 1;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber1) && (trh->accept(j)) )
			{
				hltHaloOver1 += 1;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber2) && (trh->accept(j)) )
			{
				hltHaloOver2 += 1;
				haloTriggerAny = true;
			}
			
			if( (j == majikNumber3) && (trh->accept(j)) )
			{
				hltHaloRing23 += 1;
				haloTriggerAny = true;
			}
		}// loop over all triggers in every event, check if path was run
	}//HLTriggerTag != null
	
	if(haloTriggerAny == true)
	{
		double haloVals[4][4];
		for( int i = 0; i < 4; i++)
		{
			haloVals[i][0] = 0;
		}		
		
		if( lctProducer.label() != "null" )
		{
			edm::ESHandle<CSCGeometry> pDD;
			es.get<MuonGeometryRecord>().get( pDD );
			CSCTriggerGeometry::setGeometry(pDD);
			
			edm::Handle<CSCCorrelatedLCTDigiCollection> corrlcts;
			e.getByLabel(lctProducer.label(),lctProducer.instance(),corrlcts);
			for(CSCCorrelatedLCTDigiCollection::DigiRangeIterator csc=corrlcts.product()->begin(); csc!=corrlcts.product()->end(); csc++)
			{
				CSCCorrelatedLCTDigiCollection::Range range1 = corrlcts.product()->get((*csc).first);
				for(CSCCorrelatedLCTDigiCollection::const_iterator lct=range1.first; lct!=range1.second; lct++)
				{
					int endcap  = (*csc).first.endcap()-1;
					int station = (*csc).first.station()-1;
					int sector  = (*csc).first.triggerSector()-1;
					int cscId   = (*csc).first.triggerCscId()-1;
					int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels((*csc).first);
					int fpga    = ( subSector ? subSector-1 : station+1 );

					std::cout << "Station, sector: " << station << ", " << sector << std::endl;

					if( (station == 1) || (station == 2) )
					{
						int modEnd = 0;
						if( endcap == 0 ) modEnd = -1;
						if( endcap == 1 ) modEnd = 1;
						int indexHalo = modEnd + station;
						if(haloVals[indexHalo][0] == 1.0) haloVals[indexHalo][3] = 1.0;
						if(haloVals[indexHalo][0] == 0) haloVals[indexHalo][0] = 1.0;
						haloVals[indexHalo][1] = sector*1.0;
						
						lclphidat lclPhi;
						lclPhi = srLUTs_[fpga]->localPhi(lct->getStrip(), lct->getPattern(), lct->getQuality(), lct->getBend() );
						
						gblphidat gblPhi;
						gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, lct->getKeyWG(), cscId+1);
						
						gbletadat gblEta;
						gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, lct->getKeyWG(), cscId+1);
						
						haloVals[indexHalo][2] = gblEta.global_eta/127. * 1.5 + 0.9;
						haloVals[indexHalo][3] = gblPhi.global_phi/4096 * 62;
					} //station1 or 2
				
					if( (haloVals[0][0] == 1.) && (haloVals[1][0] == 1.) && (haloVals[0][3] != 1.) && (haloVals[1][3] != 1.)  ){
						if( haloVals[0][1] == haloVals[1][1] ){
							double delEta23 = haloVals[1][2] - haloVals[0][2];
							double delPhi23 = haloVals[1][3] - haloVals[0][3];
							haloDelEta23->Fill( delEta23 );
							haloDelPhi23->Fill( delPhi23 );
						}
					}
				
					if( (haloVals[2][0] == 1.) && (haloVals[3][0] == 1.) && (haloVals[2][3] != 1.) && (haloVals[3][3] != 1.)  ){
						if( haloVals[2][1] == haloVals[3][1] ){
							double delEta23 = haloVals[3][2] - haloVals[2][2];
							double delPhi23 = haloVals[3][3] - haloVals[2][3];
							haloDelEta23->Fill( delEta23 );
							haloDelPhi23->Fill( delPhi23 );
						}
					}
				}//LCT Range iter
			}//CSCCorrelatedLCT
		}//if lct!=null
	}//if halo trigger
			
	
	edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
	e.getByLabel(GMTInputTag, gmtrc_handle);
	L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
	
	std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
	std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;	
	
	for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
		std::vector<L1MuRegionalCand>::const_iterator iter1;
		std::vector<L1MuRegionalCand> rmc;
		
		int ihalo = 0;
		
		rmc = igmtrr->getCSCCands();
		for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
			if ( !(*iter1).empty() ) {
				if((*iter1).isFineHalo()) {
					ihalo++;
				}
			}
		}
		if(igmtrr->getBxInEvent()==0 && ihalo>0) CscHalo_Gmt += 1;
	}

}
