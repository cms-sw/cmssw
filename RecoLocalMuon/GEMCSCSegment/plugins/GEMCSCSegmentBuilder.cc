/** 
 * \file GEMCSCSegmentBuilder.cc
 *
 */

#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentBuilder.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentAlgorithm.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentBuilderPluginFactory.h>


#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>


GEMCSCSegmentBuilder::GEMCSCSegmentBuilder(const edm::ParameterSet& ps) : gemgeom_(0), cscgeom_(0) 
{

    // Algo name
    std::string algoName = ps.getParameter<std::string>("algo_name");
    edm::LogVerbatim("GEMCSCSegmentBuilder")<< "[GEMCSCSegmentBuilder :: ctor] algorithm name: " << algoName;

    // SegAlgo parameter set 	  
    edm::ParameterSet segAlgoPSet = ps.getParameter<edm::ParameterSet>("algo_psets");
 
    // Ask factory to build this algorithm, giving it appropriate ParameterSet
    algo = GEMCSCSegmentBuilderPluginFactory::get()->create(algoName, segAlgoPSet);

}

GEMCSCSegmentBuilder::~GEMCSCSegmentBuilder() 
{
    delete algo;
    edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: dstor] deleted the algorithm";
}

void GEMCSCSegmentBuilder::LinkGEMRollsToCSCChamberIndex(const GEMGeometry* gemGeo, const CSCGeometry* cscGeo) 
{

  for (TrackingGeometry::DetContainer::const_iterator it=gemGeo->dets().begin();it<gemGeo->dets().end();it++)
    {
      const GEMChamber* ch = dynamic_cast< const GEMChamber* >( *it );
      if(ch != 0 )
	{
	  std::vector< const GEMEtaPartition*> rolls = (ch->etaPartitions());
	  for(std::vector<const GEMEtaPartition*>::const_iterator r = rolls.begin(); r != rolls.end(); ++r)
	    {
	      GEMDetId gemId = (*r)->id();
	      int region=gemId.region();
	      if(region!=0)
		{
		  int station    = gemId.station();
		  int ring       = gemId.ring();
		  int gemchamber = gemId.chamber();
		  int layer      = gemId.layer();
		  int cscring    = ring;
		  int cscstation = station;
		  int cscchamber = gemchamber;
		  int csclayer   = layer;
		  CSCStationIndex ind(region,cscstation,cscring,cscchamber,csclayer);
		  std::set<GEMDetId> myrolls;
		  if (rollstoreCSC.find(ind)!=rollstoreCSC.end()) myrolls=rollstoreCSC[ind];
		  myrolls.insert(gemId);
		  rollstoreCSC[ind]=myrolls;
		}
	    }
	}
    }

  // edm::LogVerbatim to print details of std::map< CSCIndex, std::set<GEMRolls> >
  for(std::map<CSCStationIndex,std::set<GEMDetId> >::iterator mapit = rollstoreCSC.begin();
      mapit != rollstoreCSC.end(); ++mapit) 
    {
      CSCStationIndex    map_first = mapit->first;
      std::set<GEMDetId> map_secnd = mapit->second;
      std::stringstream GEMRollsstream;
      for(std::set<GEMDetId>::iterator setit=map_secnd.begin(); setit!=map_secnd.end(); ++setit) 
	{ 
	  GEMRollsstream<<"[ GEM Id: "<<setit->rawId()<<" ("<<*setit<<")"<<"],"<<std::endl; 
	}
      std::string GEMRollsstr = GEMRollsstream.str();
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: LinkGEMRollsToCSCChamberIndex] CSC Station Index :: ["
       					      <<map_first.region()<<","<<map_first.station()<<","<<map_first.ring()<<","<<map_first.chamber()<<","<<map_first.layer()
       					      <<"] has following GEM rolls: ["<<GEMRollsstr<<"]"<<std::endl;
    }
}

void GEMCSCSegmentBuilder::build(const GEMRecHitCollection* recHits, const CSCSegmentCollection* cscsegments, GEMCSCSegmentCollection& oc) 
{
  	
  edm::LogVerbatim("GEMCSCSegmentBuilder")<< "[GEMCSCSegmentBuilder :: build] Total number of GEM rechits in this event: " << recHits->size()
					  << " Total number of CSC segments in this event: " << cscsegments->size();
  
  // Structure of the build function:
  // --------------------------------  
  // The build function is called once per Event
  // It loops over CSC Segment Collection and
  //   - identifies the CSC Chambers with segments    
  //   - searches for rechits in GEM rolls associared
  //   - creates CSC segment collection with GEM RecHits associated (1)
  //   - creates CSC segment collection without GEM RecHits associated (2)
  // then 
  //   - passes CSC segment collection (1) and GEM rechits to segment builder
  //   - makes a copy of CSC segment collection (2) and push it in the GEMCSC Segment collection


  // GEM Roll can contain more than one rechit     --> therefore create map <detId, vector<GEMRecHit> >
  // CSC Chamber can contain more than one segment --> therefore create map <detId, vector<CSCSegment> >
  std::map<uint32_t, std::vector<GEMRecHit*> >        gemRecHitCollection;
  std::map<uint32_t, std::vector<const CSCSegment*> > cscSegColl_GEM11;
  std::map<uint32_t, std::vector<const CSCSegment*> > cscSegColl_noGEM;



  // === Loop over CSC Segment Collection ===
  // ========================================
  for(CSCSegmentCollection::const_iterator segmIt = cscsegments->begin(); segmIt != cscsegments->end(); ++segmIt) 
    {   
      CSCDetId CSCId = segmIt->cscDetId();
    
      // Search for Matches between GEM Roll and CSC Chamber
      //   - only matches between ME1/1(a&b) and GE1/1
      //   - notation: CSC ring 1 = ME1/1b; CSC ring 4 = ME1/1a
      
      
      // Case A :: ME1/1 Segments can have GEM Rechits associated to them
      // ================================================================
      if(CSCId.station()==1 && (CSCId.ring()==1 || CSCId.ring()==4)) 
	{
	
	  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] Found ME1/1 Segment in "<<CSCId.rawId()<<" = "<<CSCId<<std::endl;

	  // 1) Save the CSC Segment in CSC segment collection
	  // -------------------------------------------------
	  // get CSC segment vector associated to this CSCDetId
	  // if no vector is associated yet to this CSCDetId, create empty vector
	  // then add the segment to the vector
	  // and assign the vector again to the CSCDetId
	  std::vector<const CSCSegment* > cscsegmentvector = cscSegColl_GEM11[CSCId.rawId()];
	  cscsegmentvector.push_back(segmIt->clone());
	  cscSegColl_GEM11[CSCId.rawId()]=cscsegmentvector;
	  
	  // 2) Make a vector of GEM DetIds associated to this CSC chamber of the segment
	  // ----------------------------------------------------------------------------
	  std::vector<GEMDetId> rollsForThisCSCvector;
	  
	  int cscRegion  = CSCId.endcap();
	  int cscStation = CSCId.station();
	  int cscChamber = CSCId.chamber();
	  int gemRegion  = 1; if(cscRegion==2) gemRegion= -1; // CSC Endcaps are numbered 1,2; GEM Endcaps are numbered +1, -1
	  int gemRing    = 1;                                 // this we can hardcode, only GEMs in Ring 1
	  int gemStation = cscStation;
	  
	  int gem1stChamber = cscChamber;
	  // Just adding also neighbouring chambers here is not enough to get the GEM-CSC segment looking at overlapping chambers
	  // Need to disentangle the use of the CSC chamber and GEM roll in the GEMCSCSegFit class
	  // For now just ignore the neighbouring chambers and keep code commented out ... at later point we will include it again
	  // int gem2ndChamber = gem1stChamber+1; if(gem2ndChamber>36) gem2ndChamber-=36; // neighbouring GEM chamber X+1
	  // int gem3rdChamber = gem1stChamber-1; if(gem2ndChamber<1)  gem2ndChamber+=36; // neighbouring GEM chamber X-1
	  
	  std::vector<CSCStationIndex> indexvector;       
	  CSCStationIndex index11(gemRegion,gemStation,gemRing,gem1stChamber,1);  // GEM Chamber Layer 1       
	  CSCStationIndex index12(gemRegion,gemStation,gemRing,gem1stChamber,2);  // GEM Chamber Layer 2
	  indexvector.push_back(index11); indexvector.push_back(index12); 
	  // for now not inserting neighbouring chambers and keep code commented out ... at later point we will include it again
	  // CSCStationIndex index21(gemRegion,gemStation,gemRing,gem2ndChamber,1);         CSCStationIndex index22(gemRegion,gemStation,gemRing,gem2ndChamber,2); 
	  // CSCStationIndex index31(gemRegion,gemStation,gemRing,gem3rdChamber,1);         CSCStationIndex index32(gemRegion,gemStation,gemRing,gem3rdChamber,2); 
	  // indexvector.push_back(index21); indexvector.push_back(index22); 
	  // indexvector.push_back(index31); indexvector.push_back(index32); 
	  
	  for(std::vector<CSCStationIndex>::iterator cscIndexIt=indexvector.begin(); cscIndexIt!=indexvector.end(); ++cscIndexIt) 
	    {
	      std::set<GEMDetId> rollsForThisCSC = rollstoreCSC[*cscIndexIt]; 
	      for (std::set<GEMDetId>::iterator gemRollIt = rollsForThisCSC.begin(); gemRollIt != rollsForThisCSC.end(); ++gemRollIt)
		{
		  rollsForThisCSCvector.push_back(*gemRollIt);
		}
	    }

      
	  // 3) Loop over GEM Rechits
	  // ------------------------
	  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] Start Loop over GEM Rechits :: size = "<<recHits->size()
						  <<" and number of Rolls for this CSC :: "<<rollsForThisCSCvector.size()<<std::endl;
	  for(GEMRecHitCollection::const_iterator hitIt = recHits->begin(); hitIt != recHits->end(); ++hitIt) 
	    {
	      GEMDetId gemIdfromHit = hitIt->gemId();
	      
	      // Loop over GEM rolls being pointed by a CSC segment and look for a match
	      for (std::vector<GEMDetId>::iterator gemRollIt = rollsForThisCSCvector.begin(); gemRollIt != rollsForThisCSCvector.end(); ++gemRollIt) 
		{
		
		  GEMDetId gemIdfromCSC(*gemRollIt);
		  if(gemIdfromHit == gemIdfromCSC.rawId()) 
		    {
		      // get GEM RecHit vector associated to this CSC DetId
		      // if no vector is associated yet to this CSC DetId, create empty vector
		      // then add the rechits to the vector
		      // and assign the vector again to the CSC DetId  
		      std::vector<GEMRecHit* > gemrechitvector = gemRecHitCollection[CSCId.rawId()];
		      // check whether this hit was already filled in the gemrechit vector
		      bool hitfound = false;
		      for(std::vector<GEMRecHit*>::const_iterator it=gemrechitvector.begin(); it!=gemrechitvector.end(); ++it) 
			{
			  if(hitIt->gemId()==(*it)->gemId() && hitIt->localPosition()==(*it)->localPosition()) hitfound=true;
			}
		      if(!hitfound) gemrechitvector.push_back(hitIt->clone());
		      gemRecHitCollection[CSCId.rawId()]=gemrechitvector;
		      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] GEM Rechit in "<<hitIt->gemId()<< "["<<hitIt->gemId().rawId()
							      <<"] added to CSC segment found in "<<CSCId<<" ["<<CSCId.rawId()<<"]"<<std::endl;	      
		    }
		} // end Loop over GEM Rolls 
	    } // end Loop over GEM RecHits
	} // end requirement of CSC segment in ME1/1 


      // Case B :: all other CSC Chambers have no GEM Chamber associated
      // ===============================================================
      else if(!(CSCId.station()==1 && (CSCId.ring()==1 || CSCId.ring()==4))) 
	{

	  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] Found non-ME1/1 Segment in "<<CSCId.rawId()<<" = "<<CSCId<<std::endl;

	  // get CSC segment vector associated to this CSCDetId
	  // if no vector is associated yet to this CSCDetId, create empty vector
	  // then add the segment to the vector
	  // and assign the vector again to the CSCDetId
	  std::vector<const CSCSegment* > cscsegmentvector_noGEM = cscSegColl_noGEM[CSCId.rawId()];
	  cscsegmentvector_noGEM.push_back(segmIt->clone());
	  cscSegColl_noGEM[CSCId.rawId()]=cscsegmentvector_noGEM;  
	}

      else {} // no other option

    } // end Loop over csc segments
    


  // === Now pass CSC Segments and GEM RecHits to the Segment Algorithm ===
  // ======================================================================

  // Here we do the first modification to the code of Raffaella
  //
  // Raffaella's code loops over the CSC Segments and assign GEM Rechits to them
  // Then it loops over the GemRechits and does the segmentbuilding using the
  // GEM Rechits and the assigned CSC segment. Now this relationship is not a one-to-one relationship (bijection)
  // In case there are more than 1 segment in the CSC, we have no control that the right supersegment is built
  //
  // Now we have a particular case where two muons passed through ME1/1,
  // resulting in 4 segments (2 real ones and 2 ghosts)
  // In this case the gemrechits are assigned to all 4 CSC segments
  // but the supersegment is only constructed once, instead of 4 trials,
  // resulting in 2 supersegments
  //
  // This is something I will change in the GEMCSCSegAlgoRR
  // In GEMCSCSegmentBuilder we just give all CSC segments and all GEM Rechits 
  // belonging to one CSC chamber to the GEMCSC Segment Algo

  // case A :: Look for GEM Rechits associated to CSC Segments
  // =========================================================
  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] Run over the gemRecHitCollection (size = "<<gemRecHitCollection.size()<<") and build GEMCSC Segments";
  for(auto gemHitIt=gemRecHitCollection.begin(); gemHitIt != gemRecHitCollection.end(); ++gemHitIt) 
    {

      CSCDetId cscId(gemHitIt->first);
      
      // hits    
      std::vector<const CSCSegment*> cscSegments = cscSegColl_GEM11[cscId.rawId()];
      std::vector<const GEMRecHit*>  gemRecHits;
      // dets
      std::map<uint32_t, const CSCLayer* >        cscLayers;
      std::map<uint32_t, const GEMEtaPartition* > gemEtaPartitions;
   
      // fill dets for CSC
      std::vector<const CSCLayer*> cscLayerVector = cscgeom_->chamber(cscId.rawId())->layers();
      for(std::vector<const CSCLayer*>::const_iterator layIt = cscLayerVector.begin(); layIt != cscLayerVector.end(); ++layIt) 
	{
	  cscLayers[(*layIt)->id()]         = cscgeom_->layer((*layIt)->id());
	}

      // fill dets & hits for GEM
      for(auto rechit = gemHitIt->second.begin(); rechit != gemHitIt->second.end(); ++rechit) 
	{
	  GEMDetId gemid = (*rechit)->gemId();
	  gemRecHits.push_back(*rechit);
	  gemEtaPartitions[gemid.rawId()] = gemgeom_->etaPartition(gemid.rawId());
	}
      
      // LogDebug
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] ask SegmentAlgo to be run :: CSC DetId "<<cscId.rawId()<<" = "<<cscId
					      <<" with "<<cscSegments.size()<<" CSC segments and "<<gemRecHits.size()<<" GEM rechits";

      // Ask the Segment Algorithm to build a GEMCSC Segment    
      std::vector<GEMCSCSegment> segmentvector = algo->run(cscLayers, gemEtaPartitions, cscSegments, gemRecHits);
      
      // --- LogDebug --------------------------------------------------------
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] SegmentAlgo ran, now trying to add the segments from the returned GEMCSCSegment vector (with size = "
					      <<segmentvector.size()<<") to the master collection";
      std::stringstream segmentvectorss; segmentvectorss<<"[GEMCSCSegmentBuilder :: build] :: GEMCSC segmentvector :: elements ["<<std::endl;
      for(std::vector<GEMCSCSegment>::const_iterator segIt = segmentvector.begin(); segIt != segmentvector.end(); ++segIt)
        {
	  segmentvectorss<<"[GEMCSC Segment details: \n"<<*segIt<<"],"<<std::endl;
        }
      segmentvectorss<<"]";
      std::string segmentvectorstr = segmentvectorss.str();
      edm::LogVerbatim("GEMCSCSegmentBuilder") << segmentvectorstr;
      // --- End LogDebug ----------------------------------------------------


      // Add the segments to master collection
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] |--> GEMCSC Segments created, now try to add to the collection :: CSC DetId: "<<cscId.rawId()<<" = "<<cscId
					      <<" added "<<segmentvector.size()<<" GEMCSC Segments";    
      oc.put(cscId, segmentvector.begin(), segmentvector.end());
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] |--> GEMCSC Segments added to the collection";    
    }
  

  // Case B :: push CSC segments without GEM rechits 
  // associated to them into the GEMCSC Segment Collection
  // =====================================================
  // Also for the case where no GEM chamber is associated to the CSC chambers
  // fill also GEMCSC segment, but this is basically a copy of the CSC segment
  // this will allow us to use only the GEMCSC segment collection in STA Mu Reco
  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] Run over the CSC Segment collection without GEM (size = "<<cscSegColl_noGEM.size()
					  <<" CSC chambers with segments) and wrap GEMCSC Segments";
  for(auto cscSegIt=cscSegColl_noGEM.begin(); cscSegIt != cscSegColl_noGEM.end(); ++cscSegIt) 
    {

      CSCDetId cscId(cscSegIt->first);
      
      std::vector<const GEMRecHit*>  gemRecHits_noGEM; // make empty gem rechits vector
      std::vector<const CSCSegment*> cscSegments_noGEM = cscSegColl_noGEM[cscSegIt->first];
      std::vector<GEMCSCSegment> segmentvector;
     
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] |--> Run over the CSC Segment vector without GEM (size = "<<cscSegments_noGEM.size()
					      <<" CSC segments) and wrap GEMCSC Segments"; 
      for (std::vector<const CSCSegment*>::iterator it=cscSegments_noGEM.begin(); it!=cscSegments_noGEM.end(); ++it) 
	{
	  // wrap the CSC segment in the GEMCSC segment class
	  GEMCSCSegment tmp(*it, gemRecHits_noGEM, (*it)->localPosition(), (*it)->localDirection(), (*it)->parametersError(), (*it)->chi2());
	  segmentvector.push_back(tmp);
	}
      
      // add the segments to the master collection
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] |--> CSC Segments wrapped, now try to add to the collection :: CSC DetId: "<<cscId.rawId()<<" = "<<cscId
					      <<" added "<<segmentvector.size()<<" GEMCSC Segments";    
      oc.put(cscId, segmentvector.begin(), segmentvector.end());
      edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] |--> CSC Segments added to the collection";    
    }

  edm::LogVerbatim("GEMCSCSegmentBuilder")<<"[GEMCSCSegmentBuilder :: build] job done !!!";
}


void GEMCSCSegmentBuilder::setGeometry(const GEMGeometry* gemgeom, const CSCGeometry* cscgeom) 
{
	gemgeom_ = gemgeom;
	cscgeom_ = cscgeom;
}


