/*
 *	HaloTrigger.cc
 *	Created by Joseph Gartner on 12/17/08
 *	Please use this code for good, not evil
 */
 
#include "HLTriggerOffline/special/src/HaloTrigger.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;
using namespace edm;
using namespace trigger;

HaloTrigger::HaloTrigger(const ParameterSet& ps) 
	: HLTriggerTag( ps.getParameter< InputTag >("HLTriggerTag") ),
	  L1GTRR( ps.getParameter< InputTag >("L1GTRR") ),
		GMTInputTag( ps.getParameter< InputTag >("GMTInputTag") )
{
	first = false;
	outFile = ps.getUntrackedParameter<string>("outFile");
}

HaloTrigger::~HaloTrigger()
{}

void HaloTrigger::beginJob(const EventSetup& es)
{
	/////////////////////////////
	// DQM Directory Structure //
	/////////////////////////////
	DQMStore * dbe = 0;
	dbe = Service<DQMStore>().operator->();
	if( dbe ){
		dbe->setCurrentFolder("HLTriggerOffline/special");
		
		/////////////////////////////
		// Define Monitor Elements //
		/////////////////////////////
		halocount = dbe->book1D("HltL1CscTrue","Event Count of various triggers",9,-0.5,8.5);
		halocount->setAxisTitle("Trigger Type",1);
		halocount->setAxisTitle("Count",2);
		halocount->setBinLabel(2,"CSCTF Trigger",   1);
		halocount->setBinLabel(4,"GMT Halo Trigger",1);
		halocount->setBinLabel(6,"GT Halo Trigger", 1);
		halocount->setBinLabel(8,"HLT Halo Trigger",1);
	}
}

void HaloTrigger::endJob(void)
{
	dbe->save(outFile);	
	return;
}

void HaloTrigger::analyze(const Event& e, const EventSetup& es)
{
	int hltHaloTriggerTrue=0, gtHaloTriggerTrue=0, gmtHaloTriggerTrue=0, l1tHaloTriggerTrue=0;
	
	if( HLTriggerTag.label() != "null" )
	{
		// Access HLT trigger results
		// And check for a halo trigger
		
		edm::Handle<TriggerResults> trh;
		e.getByLabel(HLTriggerTag,trh);
		edm::TriggerNames names;
		
		if (!first)
		{
			first = true;
			names.init(*trh);	
			
			Namen = names.triggerNames();
			const unsigned int n(Namen.size());
			
			std::string searchName_Halo ("HLT_L1_CSCMuonHalo");
			for( unsigned int i=0; i!=n; ++i){
				int compy = Namen[i].compare(searchName_Halo);
				if( compy == 0) majikNumber_HLThalo = i;
			}
		}
		
		const unsigned int n(Namen.size());
		for( unsigned int j=0; j!=n; ++j)
			if( (j == majikNumber_HLThalo) && (trh->accept(j)) )
				hltHaloTriggerTrue = 1;
				
	}//HLTriggerTag != null
			
	if( L1GTRR.label() != "null")
	{
		edm::Handle<L1GlobalTriggerReadoutRecord> gtrr;
		e.getByLabel(L1GTRR,gtrr);
		
		DecisionWord gtDecisionWord = gtrr->decisionWord();
		
		if(gtDecisionWord[54]) gtHaloTriggerTrue = 1;

	}// GT Trigger Tag
	
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
		
		if(igmtrr->getBxInEvent()==0 && ihalo>0) gmtHaloTriggerTrue = true;
	}
	
	edm::Handle<std::vector<L1MuRegionalCand> > csctf;
	e.getByLabel("gtDigis","CSC",csctf);
	std::vector<L1MuRegionalCand>::const_iterator tfTrk;
		
	for(tfTrk = csctf->begin(); tfTrk != csctf->end(); tfTrk++)
	{
		int haloTrig = (0x400000&(tfTrk->getDataWord() ) ) >> 22;
		if(haloTrig == 1)
		{
			l1tHaloTriggerTrue = 1;
		}
	}
	
	if( l1tHaloTriggerTrue == 1 ) halocount->Fill(1);
	if( gmtHaloTriggerTrue == 1 ) halocount->Fill(3);
	if( gtHaloTriggerTrue  == 1 ) halocount->Fill(5);
	if( hltHaloTriggerTrue == 1 ) halocount->Fill(7);
}
