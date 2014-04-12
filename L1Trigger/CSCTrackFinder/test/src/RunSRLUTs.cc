#include "L1Trigger/CSCTrackFinder/test/src/RunSRLUTs.h"

#include <vector>
#include <string>
#include <string.h>
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"

namespace csctf_analysis
{
RunSRLUTs::RunSRLUTs()
{
  
	//Set up SRLUTs
    bool TMB07 = true;
    edm::ParameterSet srLUTset;
    srLUTset.addUntrackedParameter<bool>("Binary",false);
    srLUTset.addUntrackedParameter<bool>("ReadLUTs",false);
    srLUTset.addUntrackedParameter<std::string>("LUTPath","./");
    srLUTset.addUntrackedParameter<bool>("UseMiniLUTs",true);
    int endcap=1;
    int sector=1;

  for(int station=1,fpga=0; station<=4 && fpga<5; station++)
  {
  	if(station==1)
			for(int subSector=0; subSector<2 && fpga<5; subSector++)
	  		srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, subSector+1, station, srLUTset, TMB07);
    else
			srLUTs_[fpga++] = new CSCSectorReceiverLUT(endcap, sector, 0, station, srLUTset, TMB07);
  }


}

RunSRLUTs::~RunSRLUTs()
{
}

void RunSRLUTs::run(std::vector<csctf::TrackStub> *stub_list)
{

	//Assigning (global) eta and phi packed
	for(std::vector<csctf::TrackStub>::iterator itr=stub_list->begin(); itr!=stub_list->end(); itr++)
	{
	  if(itr->station() != 5)
	  {
	 	CSCDetId id(itr->getDetId().rawId());
	  	unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();

        	lclphidat lclPhi;
        	try 
		{
        			lclPhi = srLUTs_[fpga]->localPhi(itr->getStrip(), itr->getPattern(), itr->getQuality(), itr->getBend());
        	} 
		catch( cms::Exception &e ) 
		{
        			bzero(&lclPhi,sizeof(lclPhi));
        	}

        	gblphidat gblPhi;
        	try 
		{
        		gblPhi = srLUTs_[fpga]->globalPhiME(lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
		} 
		catch( cms::Exception &e ) 
		{
        		bzero(&gblPhi,sizeof(gblPhi));
        	}

        	gbletadat gblEta;
        	try 
		{
        		gblEta = srLUTs_[fpga]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
        	} 
		catch( cms::Exception &e ) 
		{
        		bzero(&gblEta,sizeof(gblEta));
        	}

	  		itr->setEtaPacked(gblEta.global_eta);
	  		itr->setPhiPacked(gblPhi.global_phi);
	  }
	}
}

void RunSRLUTs::makeTrackStubs(const CSCCorrelatedLCTDigiCollection * inClcts,std::vector<csctf::TrackStub> *outStubVec)
{

	// Making a list of track stubs from lct collection
	CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;
	for(Citer = inClcts->begin(); Citer != inClcts->end(); Citer++)
	{
		CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
		CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

		for(; Diter != Dend; Diter++)
		{
	  		csctf::TrackStub theStub((*Diter),(*Citer).first);
	  		outStubVec->push_back(theStub);
		}
    	}

	run(outStubVec);
}

void RunSRLUTs::makeAssociatedTrackStubs(const L1CSCTrackCollection * inTrackColl,TrackAndAssociatedStubsCollection *outTrkStubCol)
{
  L1CSCTrackCollection::const_iterator l1CSCTrack;
  for (l1CSCTrack = inTrackColl->begin(); l1CSCTrack != inTrackColl->end(); l1CSCTrack++)
  {
	std::vector<csctf::TrackStub> stubList;

	csc::L1Track track = l1CSCTrack->first;
	CSCCorrelatedLCTDigiCollection clctDigiCol = l1CSCTrack->second;

	makeTrackStubs(&clctDigiCol,&stubList);

	//Addding Stubs and Track to l1TrackAndEasyStubs
	TrackAndAssociatedStubs tempTrackAndStubs;
	tempTrackAndStubs.first = track;
	tempTrackAndStubs.second = stubList;
	outTrkStubCol->push_back(tempTrackAndStubs);
  }

}
}
