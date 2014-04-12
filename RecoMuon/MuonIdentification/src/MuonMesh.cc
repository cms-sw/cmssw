#include "RecoMuon/MuonIdentification/interface/MuonMesh.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include <utility>
#include <algorithm>

MuonMesh::MuonMesh(const edm::ParameterSet& parm) : doME1a(parm.getParameter<bool>("ME1a")),
						    doOverlaps(parm.getParameter<bool>("Overlap")),
						    doClustering(parm.getParameter<bool>("Clustering")),
						    OverlapDPhi(parm.getParameter<double>("OverlapDPhi")), 
						    OverlapDTheta(parm.getParameter<double>("OverlapDTheta")), 
						    ClusterDPhi(parm.getParameter<double>("ClusterDPhi")), 
						    ClusterDTheta(parm.getParameter<double>("ClusterDTheta")) {
}

void MuonMesh::fillMesh(std::vector<reco::Muon>* inputMuons) {

  for(std::vector<reco::Muon>::iterator muonIter1 = inputMuons->begin();
      muonIter1 != inputMuons->end();
      ++muonIter1) {
    if(muonIter1->isTrackerMuon()){

      mesh_[&*muonIter1]; // create this entry if it's a tracker muon
      for(std::vector<reco::Muon>::iterator muonIter2 = inputMuons->begin();
	  muonIter2 != inputMuons->end();
	  ++muonIter2) {
	if(muonIter2->isTrackerMuon()) {
	  if(muonIter2 != muonIter1) {
	    
	    // now fill all the edges for muon1 based on overlaps with muon2
	    for(std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = muonIter1->matches().begin();
		chamberIter1 != muonIter1->matches().end();
		++chamberIter1) {	      
	      for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
		  segmentIter1 != chamberIter1->segmentMatches.end();
		  ++segmentIter1) {
		for(std::vector<reco::MuonChamberMatch>::iterator chamberIter2 = muonIter2->matches().begin();
		    chamberIter2 != muonIter2->matches().end();
		    ++chamberIter2) {		  
		  for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter2 = chamberIter2->segmentMatches.begin();
		      segmentIter2 != chamberIter2->segmentMatches.end();
		      ++segmentIter2) {
		    		      

		    //if(segmentIter1->mask & 0x1e0000 && segmentIter2->mask &0x1e0000) {

		      bool addsegment(false);
		      
		      if( segmentIter1->cscSegmentRef.isNonnull() && segmentIter2->cscSegmentRef.isNonnull() ) {
						
			if( doME1a && 
			    isDuplicateOf(segmentIter1->cscSegmentRef,segmentIter2->cscSegmentRef) &&
			    CSCDetId(chamberIter1->id).ring() == 4 && CSCDetId(chamberIter2->id).ring() == 4 &&
			    chamberIter1->id == chamberIter2->id ) {
			  addsegment = true;
			  //std::cout << "\tME1/a sharing detected." << std::endl;
			}
			
			if( doOverlaps &&
			    isDuplicateOf(std::make_pair(CSCDetId(chamberIter1->id),segmentIter1->cscSegmentRef),
					 std::make_pair(CSCDetId(chamberIter2->id),segmentIter2->cscSegmentRef)) ) {
			  addsegment = true;
			  //std::cout << "\tChamber Overlap sharing detected." << std::endl;
			}
			
			if( doClustering &&
			    isClusteredWith(std::make_pair(CSCDetId(chamberIter1->id),segmentIter1->cscSegmentRef),
					   std::make_pair(CSCDetId(chamberIter2->id),segmentIter2->cscSegmentRef)) ) {
			  addsegment = true;
			  //std::cout << "\tCluster sharing detected." << std::endl;
			}
			//std::cout << std::endl;
		      } // has valid csc segment ref
		      
		      if(addsegment) { // add segment if clusters/overlaps/replicant and doesn't already exist
			
			if(find(mesh_[&*muonIter1].begin(),
				mesh_[&*muonIter1].end(),
				std::make_pair(&*muonIter2,
					       std::make_pair(&*chamberIter2,
							      &*segmentIter2)
					       )
				) == mesh_[&*muonIter1].end()) {
			  mesh_[&*muonIter1].push_back(std::make_pair(&*muonIter2,
								      std::make_pair(&*chamberIter2,
										     &*segmentIter2)
								      )
						       );
			} // find
		      } // add segment?
		      //} // both segments won arbitration
		  } // segmentIter 2				  		  
		} // chamberIter2
	      } //segmentIter1
	    } // chamberIter1	
	    
	  } // if different muon	
	} // is tracker muon  
      } // muonIter2
      
    } // is tracker muon
  } // muonIter1

  // special cases

  // one muon: mark all segments belonging to a muon as cleaned as there are no other muons to fight with
  if(mesh_.size() == 1) {
    for(std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = mesh_.begin()->first->matches().begin();
	chamberIter1 != mesh_.begin()->first->matches().end();
	++chamberIter1) {	      
      for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
	  segmentIter1 != chamberIter1->segmentMatches.end();
	  ++segmentIter1) {
	segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);	
      } // segmentIter1
    } // chamberIter1
  } // if only one tracker muon set winner bit boosted arbitration

  // segments that are not shared amongst muons and the have won all segment arbitration flags need to be promoted
  // also promote DT segments
  if(mesh_.size() > 1) {
    for( MeshType::iterator i = mesh_.begin(); i != mesh_.end(); ++i ) {
      for( std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = i->first->matches().begin();
	   chamberIter1 != i->first->matches().end();
	   ++chamberIter1 ) {	
	for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
	    segmentIter1 != chamberIter1->segmentMatches.end();
	    ++segmentIter1) {

	  bool shared(false);
	  
	  for( AssociationType::iterator j = i->second.begin();
	       j != i->second.end();
	       ++j ) {

	    if( segmentIter1->cscSegmentRef.isNonnull() && 
		j->second.second->cscSegmentRef.isNonnull() ) {	       	      
	      if(chamberIter1->id.subdetId() == MuonSubdetId::CSC &&
		 j->second.first->id.subdetId() == MuonSubdetId::CSC ) {
		CSCDetId segIterId(chamberIter1->id), shareId(j->second.first->id);

		if( doOverlaps &&
		    isDuplicateOf(std::make_pair(segIterId,segmentIter1->cscSegmentRef),
				  std::make_pair(shareId,j->second.second->cscSegmentRef)) )
		  shared = true;
		
		if( doME1a && 
		    isDuplicateOf(segmentIter1->cscSegmentRef,j->second.second->cscSegmentRef) &&
		    segIterId.ring() == 4 && shareId.ring() == 4 &&
		    segIterId == segIterId)
		  shared = true;

		if( doClustering && 
		    isClusteredWith(std::make_pair(CSCDetId(chamberIter1->id),segmentIter1->cscSegmentRef),
				    std::make_pair(CSCDetId(j->second.first->id),j->second.second->cscSegmentRef)) )
		  shared = true;
	      } // in CSCs?
	    } // cscSegmentRef non null?
	  } // j
	  
	  // Promote segments which have won all arbitration and are not shared or are DT segments
	  if( ((segmentIter1->mask&0x1e0000) == 0x1e0000 && !shared) ||
	      (chamberIter1->id.subdetId() == MuonSubdetId::DT && (segmentIter1->mask&0x1e0000)) ) 
	      segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);

	} // segmentIter1
      } // chamberIter1
    }// i
  } // if non-trivial case

}

void MuonMesh::pruneMesh() {
  
  for( MeshType::iterator i = mesh_.begin(); i != mesh_.end(); ++i ) {

        for( AssociationType::iterator j = i->second.begin();
	 j != i->second.end();
	 ++j ) {

      for( std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = i->first->matches().begin();
	   chamberIter1 != i->first->matches().end();
	   ++chamberIter1 ) {	
	for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
	    segmentIter1 != chamberIter1->segmentMatches.end();
	    ++segmentIter1) {

	  if(j->second.second->cscSegmentRef.isNonnull() && segmentIter1->cscSegmentRef.isNonnull()) {
	    
	    //UNUSED:	    bool me1a(false), overlap(false), cluster(false);

	    // remove physical overlap duplicates first
	    if( doOverlaps &&
		isDuplicateOf(std::make_pair(CSCDetId(chamberIter1->id),segmentIter1->cscSegmentRef),
			      std::make_pair(CSCDetId(j->second.first->id),j->second.second->cscSegmentRef)) ) {

	      if ( i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) >
		   j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ) {
		
		segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByOvlClean);
		segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);
		
		//UNUSED:		overlap = true;
	      } else if ( i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ==
			  j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ) { // muon with more matched stations wins
		
		if((segmentIter1->mask & 0x1e0000) > (j->second.second->mask & 0x1e0000)) { // segment with better match wins 
		  
		  segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByOvlClean);
		  segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);
		  
		  //UNUSED:		  overlap = true;				   
		} else { // ??
		  // leave this available for later
		}		 
	      } // overlap duplicate resolution
	    } // is overlap duplicate
	    
	    // do ME1/a arbitration second since the tie breaker depends on other stations
	    // Unlike the other cleanings this one removes the bits from segments associated to tracks which
	    // fail cleaning. (Instead of setting bits for the segments which win.)
	    if( doME1a && 
		isDuplicateOf(segmentIter1->cscSegmentRef,j->second.second->cscSegmentRef) &&
		CSCDetId(chamberIter1->id).ring() == 4 && CSCDetId(j->second.first->id).ring() == 4 &&
		chamberIter1->id ==  j->second.first->id ) {	      

	      if( j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) < 
		  i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ) {
				
		for(AssociationType::iterator AsscIter1 = i->second.begin();
		    AsscIter1 != i->second.end();
		    ++AsscIter1) {
		  if(AsscIter1->second.second->cscSegmentRef.isNonnull())
		    if(j->first == AsscIter1->first &&
		       j->second.first == AsscIter1->second.first &&
		       isDuplicateOf(segmentIter1->cscSegmentRef,AsscIter1->second.second->cscSegmentRef)) {
		      AsscIter1->second.second->mask &= ~reco::MuonSegmentMatch::BelongsToTrackByME1aClean;
		    }		
		}  

		//UNUSED:		me1a = true;
	      } else if ( j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) == 
			  i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ) { // muon with best arbitration wins
						
		bool bestArb(true);
		
		for(AssociationType::iterator AsscIter1 = i->second.begin();
		    AsscIter1 != i->second.end();
		    ++AsscIter1) {
		  if(AsscIter1->second.second->cscSegmentRef.isNonnull())
		    if(j->first == AsscIter1->first &&
		       j->second.first == AsscIter1->second.first &&
		       isDuplicateOf(segmentIter1->cscSegmentRef,AsscIter1->second.second->cscSegmentRef) && 
		       (segmentIter1->mask & 0x1e0000) < (AsscIter1->second.second->mask & 0x1e0000))
		      bestArb = false;
		}
		
		if(bestArb) {
		  
		  for(AssociationType::iterator AsscIter1 = i->second.begin();
		      AsscIter1 != i->second.end();
		      ++AsscIter1) {
		    if(AsscIter1->second.second->cscSegmentRef.isNonnull())
		      if(j->first == AsscIter1->first &&
			 j->second.first == AsscIter1->second.first &&
			 isDuplicateOf(segmentIter1->cscSegmentRef,AsscIter1->second.second->cscSegmentRef)) {
			AsscIter1->second.second->mask &= ~reco::MuonSegmentMatch::BelongsToTrackByME1aClean;
		      }
		  }
		  
		}	 
		//UNUSED		me1a = true;				

	      } // ME1/a duplicate resolution	      	      
	    } // is ME1/aduplicate?
	    
	    if(doClustering && 
	       isClusteredWith(std::make_pair(CSCDetId(chamberIter1->id),segmentIter1->cscSegmentRef),
			       std::make_pair(CSCDetId(j->second.first->id),j->second.second->cscSegmentRef))) {

	      if (i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) >
		  j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) ) {

		 segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByClusClean);
		 segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);

		 //UNUSED:		 cluster = true;
	      } else if (i->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000) <
			 j->first->numberOfMatches((reco::Muon::ArbitrationType)0x1e0000)) {

		j->second.second->setMask(reco::MuonSegmentMatch::BelongsToTrackByClusClean);
		j->second.second->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);
		
		//UNUSED:		cluster = true;
	      } else { // muon with more matched stations wins
		 
		 if((segmentIter1->mask & 0x1e0000) > (j->second.second->mask & 0x1e0000)) { // segment with better match wins 
		   
		   segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByClusClean);
		   segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);
		   
		   //UNUSED:		   cluster = true;				   
		 } else if ((segmentIter1->mask & 0x1e0000) < (j->second.second->mask & 0x1e0000)){ //
		   
		   j->second.second->setMask(reco::MuonSegmentMatch::BelongsToTrackByClusClean);
		   j->second.second->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);
		   
		   //UNUSED:		   cluster = true;				   
		 } else {
		 }
	       } // cluster sharing resolution
	      
	    } // is clustered with?
	    
	  } // csc ref nonnull	  
	} // segmentIter1
      } // chamberIter1       
    } // j, associated segments iterator
  } // i, map iterator

  // final step: make sure everything that's won a cleaning flag has the "BelongsToTrackByCleaning" flag

   for( MeshType::iterator i = mesh_.begin(); i != mesh_.end(); ++i ) {
     for( std::vector<reco::MuonChamberMatch>::iterator chamberIter1 = i->first->matches().begin();
	  chamberIter1 != i->first->matches().end();
	  ++chamberIter1 ) {	
       for(std::vector<reco::MuonSegmentMatch>::iterator segmentIter1 = chamberIter1->segmentMatches.begin();
	   segmentIter1 != chamberIter1->segmentMatches.end();
	   ++segmentIter1) {
	 // set cleaning bit if initial no cleaning bit but there are cleaning algorithm bits set.
	 if( !segmentIter1->isMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning) &&
	     segmentIter1->isMask(0xe00000) )
	   segmentIter1->setMask(reco::MuonSegmentMatch::BelongsToTrackByCleaning);	 
       }// segmentIter1
     } // chamberIter1
   } // i
  
}

bool MuonMesh::isDuplicateOf(const CSCSegmentRef& lhs, const CSCSegmentRef& rhs) const // this isDuplicateOf() deals with duplicate segments in ME1/a
{
  bool result(false);
 
  if(!lhs->isME11a_duplicate())
    return result;

  std::vector<CSCSegment> lhs_duplicates = lhs->duplicateSegments();
      
  if(fabs(lhs->localPosition().x()        - rhs->localPosition().x()      ) < 1E-3 &&
     fabs(lhs->localPosition().y()        - rhs->localPosition().y()      ) < 1E-3 &&
     fabs(lhs->localDirection().x()/lhs->localDirection().z()    - rhs->localDirection().x()/rhs->localDirection().z()   ) < 1E-3 &&
     fabs(lhs->localDirection().y()/lhs->localDirection().z()    - rhs->localDirection().y()/rhs->localDirection().z()   ) < 1E-3 &&
     fabs(lhs->localPositionError().xx()  - rhs->localPositionError().xx()   ) < 1E-3 &&
     fabs(lhs->localPositionError().yy()  - rhs->localPositionError().yy()   ) < 1E-3 &&
     fabs(lhs->localDirectionError().xx() - rhs->localDirectionError().xx()) < 1E-3 &&
     fabs(lhs->localDirectionError().yy() - rhs->localDirectionError().yy()) < 1E-3)
    result = true;

  for( std::vector<CSCSegment>::const_iterator segIter1 = lhs_duplicates.begin();
       segIter1 != lhs_duplicates.end();
       ++segIter1 ) { // loop over lhs duplicates

    if(fabs(segIter1->localPosition().x()        - rhs->localPosition().x()      ) < 1E-3 &&
       fabs(segIter1->localPosition().y()        - rhs->localPosition().y()      ) < 1E-3 &&
       fabs(segIter1->localDirection().x()/segIter1->localDirection().z()    - rhs->localDirection().x()/rhs->localDirection().z()   ) < 1E-3 &&
       fabs(segIter1->localDirection().y()/segIter1->localDirection().z()    - rhs->localDirection().y()/rhs->localDirection().z()   ) < 1E-3 &&
       fabs(segIter1->localPositionError().xx()  - rhs->localPositionError().xx()   ) < 1E-3 &&
       fabs(segIter1->localPositionError().yy()  - rhs->localPositionError().yy()   ) < 1E-3 &&
       fabs(segIter1->localDirectionError().xx() - rhs->localDirectionError().xx()) < 1E-3 &&
       fabs(segIter1->localDirectionError().yy() - rhs->localDirectionError().yy()) < 1E-3)
      result = true;
    /*
    if(fabs(segIter1->localPosition().x()        - rhs->localPosition().x()      ) < 2*sqrt(segIter1->localPositionError().xx()) &&
       fabs(segIter1->localPosition().y()        - rhs->localPosition().y()      ) < 2*sqrt(segIter1->localPositionError().yy()) &&
       fabs(segIter1->localDirection().x()/segIter1->localDirection().z()    - rhs->localDirection().x()/rhs->localDirection().z()   ) 
       < 2*std::sqrt(std::max(segIter1->localDirectionError().yy(),rhs->localDirectionError().xx())) &&
       fabs(segIter1->localDirection().y()/segIter1->localDirection().z()    - rhs->localDirection().y()/rhs->localDirection().z()   ) 
       < 2*std::sqrt(std::max(segIter1->localDirectionError().yy(),rhs->localDirectionError().yy())))
      result = true;
    */

  } // loop over duplicates

  return result;
}

bool MuonMesh::isDuplicateOf(const std::pair<CSCDetId,CSCSegmentRef>& rhs, 
			     const std::pair<CSCDetId,CSCSegmentRef>& lhs) const // this isDuplicateOf() deals with "overlapping chambers" duplicates
{
  bool result(false);
  
  // try to implement the simple case first just back-to-back segments without treatment of ME1/a ganging
  // ME1a should be a simple extension of this

  if(rhs.first.endcap() == lhs.first.endcap() &&
     rhs.first.station() == lhs.first.station() &&
     rhs.first.ring() == lhs.first.ring()) { // if same endcap,station,ring (minimal requirement for ovl candidate)
    /*
    std::cout << "Chamber 1: " << rhs.first << std::endl 
	      << "Chamber 2: " << lhs.first << std::endl;

    std::cout << "Same endcap,ring,station." << std::endl;    
    */
    //create neighboring chamber labels, treat ring as (Z mod 36 or 18) + 1 number line: left, right defined accordingly.
    unsigned modulus = ((rhs.first.ring() != 1 || rhs.first.station() == 1) ? 36 : 18);
    int left_neighbor = (((rhs.first.chamber() - 1 + modulus)%modulus == 0 ) ? modulus : (rhs.first.chamber() - 1 + modulus)%modulus ); // + modulus to ensure positivity
    int right_neighbor = (((rhs.first.chamber() + 1)%modulus == 0 ) ? modulus : (rhs.first.chamber() + 1)%modulus );

    if(lhs.first.chamber() == left_neighbor || 
       lhs.first.chamber() == right_neighbor) { // if this is a neighboring chamber then it can be an overlap
      
      std::vector<CSCSegment> thesegments;
      thesegments.push_back(*(lhs.second));
      /*
      if(lhs.second->isME11a_duplicate())
	thesegments.insert(thesegments.begin(),
			   lhs.second->duplicateSegments().begin(),
			   lhs.second->duplicateSegments().end());
      */

      //std::cout << "lhs is in neighoring chamber of rhs." << std::endl;

      // rhs local direction info
      /*
      double rhs_dydz = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).y()/
	                geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).z();
      double rhs_dxdz = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).x()/
	                geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).z();
      double rhs_dydz_err = rhs.second->localDirectionError().yy();
      double rhs_dxdz_err = rhs.second->localDirectionError().xx();
      */
      
      //rhs global position info
      double rhs_phi = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localPosition()).phi();
      double rhs_theta = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localPosition()).theta();
      double rhs_z = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localPosition()).z();

      for(std::vector<CSCSegment>::const_iterator ilhs = thesegments.begin(); ilhs != thesegments.end(); ++ilhs) {
	
	// lhs local direction info
	/*
	double lhs_dydz = geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).y()/
	                  geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).z();
	double lhs_dxdz = geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).x()/
	                  geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).z();
	double lhs_dydz_err = ilhs->localDirectionError().yy();
	double lhs_dxdz_err = ilhs->localDirectionError().xx();
	*/
	
	//lhs global position info
	double lhs_phi = geometry_->chamber(lhs.first)->toGlobal(ilhs->localPosition()).phi();
	double lhs_theta = geometry_->chamber(lhs.first)->toGlobal(ilhs->localPosition()).theta();
	double lhs_z = geometry_->chamber(lhs.first)->toGlobal(ilhs->localPosition()).z();
	/*
	  std::cout << "RHS Segment Parameters:" << std::endl
	  << "\t RHS Phi   : " << rhs_phi << std::endl
	  << "\t RHS Theta : " << rhs_theta << std::endl
	  << "\t RHS dx/dz : " << rhs_dxdz << " +- " << rhs_dxdz_err << std::endl
	  << "\t RHS dy/dz : " << rhs_dydz << " +- " << rhs_dydz_err << std::endl; 
	  
	  std::cout << "LHS Segment Parameters:" << std::endl
	  << "\t LHS Phi   : " << lhs_phi << std::endl
	  << "\t LHS Theta : " << lhs_theta << std::endl
	  << "\t LHS dx/dz : " << lhs_dxdz << " +- " << lhs_dxdz_err << std::endl
	  << "\t LHS dy/dz : " << lhs_dydz << " +- " << lhs_dydz_err << std::endl; 
	*/
	
	double phidiff = (fabs(rhs_phi - lhs_phi) > 2*M_PI ? fabs(rhs_phi - lhs_phi) - 2*M_PI : fabs(rhs_phi - lhs_phi));

	if(phidiff < OverlapDPhi && fabs(rhs_theta - lhs_theta) < OverlapDTheta && 
	   fabs(rhs_z) < fabs(lhs_z) && rhs_z*lhs_z > 0) // phi overlap region is 3.5 degrees and rhs is infront of lhs
	  result = true;
      } // loop over duplicate segments
    }// neighboring chamber
  } // same endcap,station,ring

  return result;
}

bool MuonMesh::isClusteredWith(const std::pair<CSCDetId,CSCSegmentRef>& lhs, 
			       const std::pair<CSCDetId,CSCSegmentRef>& rhs) const
{
  bool result(false);
  
  // try to implement the simple case first just back-to-back segments without treatment of ME1/a ganging
  // ME1a should be a simple extension of this

  //std::cout << lhs.first << ' ' << rhs.first << std::endl;

  if(rhs.first.endcap() == lhs.first.endcap() && lhs.first.station() < rhs.first.station()) {
          
      std::vector<CSCSegment> thesegments;
      thesegments.push_back(*(lhs.second));
      /*
      if(lhs.second->isME11a_duplicate())
	thesegments.insert(thesegments.begin(),
			   lhs.second->duplicateSegments().begin(),
			   lhs.second->duplicateSegments().end());
      */
      //std::cout << "lhs is in neighoring chamber of rhs." << std::endl;

      // rhs local direction info
      /*
      double rhs_dydz = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).y()/
	                geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).z();
      double rhs_dxdz = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).x()/
	                geometry_->chamber(rhs.first)->toGlobal(rhs.second->localDirection()).z();
      double rhs_dydz_err = rhs.second->localDirectionError().yy();
      double rhs_dxdz_err = rhs.second->localDirectionError().xx();
      */
      
      //rhs global position info
      double rhs_phi = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localPosition()).phi();
      double rhs_theta = geometry_->chamber(rhs.first)->toGlobal(rhs.second->localPosition()).theta();

      for(std::vector<CSCSegment>::const_iterator ilhs = thesegments.begin(); ilhs != thesegments.end(); ++ilhs) {
 	
	// lhs local direction info
	/*
	double lhs_dydz = geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).y()/
	                  geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).z();
	double lhs_dxdz = geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).x()/
	                  geometry_->chamber(lhs.first)->toGlobal(ilhs->localDirection()).z();
	double lhs_dydz_err = ilhs->localDirectionError().yy();
	double lhs_dxdz_err = ilhs->localDirectionError().xx();
	*/

	//lhs global position info
	double lhs_phi = geometry_->chamber(lhs.first)->toGlobal(ilhs->localPosition()).phi();
	double lhs_theta = geometry_->chamber(lhs.first)->toGlobal(ilhs->localPosition()).theta();
	/*
	  std::cout << "RHS Segment Parameters:" << std::endl
	  << "\t RHS Phi   : " << rhs_phi << std::endl
	  << "\t RHS Theta : " << rhs_theta << std::endl
	  << "\t RHS dx/dz : " << rhs_dxdz << " +- " << rhs_dxdz_err << std::endl
	  << "\t RHS dy/dz : " << rhs_dydz << " +- " << rhs_dydz_err << std::endl; 
	  
	  std::cout << "LHS Segment Parameters:" << std::endl
	  << "\t LHS Phi   : " << lhs_phi << std::endl
	  << "\t LHS Theta : " << lhs_theta << std::endl
	  << "\t LHS dx/dz : " << lhs_dxdz << " +- " << lhs_dxdz_err << std::endl
	  << "\t LHS dy/dz : " << lhs_dydz << " +- " << lhs_dydz_err << std::endl; 
	*/

	double phidiff = (fabs(rhs_phi - lhs_phi) > 2*M_PI ? fabs(rhs_phi - lhs_phi) - 2*M_PI : fabs(rhs_phi - lhs_phi));
	
	if(phidiff < ClusterDPhi && fabs(rhs_theta - lhs_theta) < ClusterDTheta) // phi overlap region is 37 degrees
	  result = true;
      } // loop over duplicate segments    
  } // same endcap,station,ring

  return result;
}
