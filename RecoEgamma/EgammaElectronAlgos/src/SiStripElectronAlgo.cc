// -*- C++ -*-
//
// Package:     EgammaElectronAlgos
// Class  :     SiStripElectronAlgo
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:12:04 EDT 2006
// $Id: SiStripElectronAlgo.cc,v 1.40 2012/01/16 09:28:17 innocent Exp $
//

// system include files

// user include files
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace {
  struct CompareHits {
    CompareHits( const TrackingRecHitLessFromGlobalPosition& iLess ) :
      less_(iLess) {}
    bool operator()(const TrackingRecHit* iLHS, const TrackingRecHit* iRHS) {
      return less_(*iLHS,*iRHS);
    }
	 
    TrackingRecHitLessFromGlobalPosition less_;
  };
}


//
// constructors and destructor
//
SiStripElectronAlgo::SiStripElectronAlgo(unsigned int maxHitsOnDetId,
					 double originUncertainty,
					 double phiBandWidth,
					 double maxNormResid,
					 unsigned int minHits,
					 double maxReducedChi2)
  : maxHitsOnDetId_(maxHitsOnDetId)
  , originUncertainty_(originUncertainty)
  , phiBandWidth_(phiBandWidth)
  , maxNormResid_(maxNormResid)
  , minHits_(minHits)
  , maxReducedChi2_(maxReducedChi2)
{
}

// SiStripElectronAlgo::SiStripElectronAlgo(const SiStripElectronAlgo& rhs)
// {
//    // do actual copying here;
// }

SiStripElectronAlgo::~SiStripElectronAlgo()
{
}

//
// assignment operators
//
// const SiStripElectronAlgo& SiStripElectronAlgo::operator=(const SiStripElectronAlgo& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectronAlgo temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void SiStripElectronAlgo::prepareEvent(const edm::ESHandle<TrackerGeometry>& tracker,
				       const edm::Handle<SiStripRecHit2DCollection>& rphiHits,
				       const edm::Handle<SiStripRecHit2DCollection>& stereoHits,
				       const edm::Handle<SiStripMatchedRecHit2DCollection>& matchedHits,
				       const edm::ESHandle<MagneticField>& magneticField)
{
  LogDebug("") << " In prepareEvent " ; 

  tracker_p_ = tracker.product();
  rphiHits_p_ = rphiHits.product();
  stereoHits_p_ = stereoHits.product();
  matchedHits_p_ = matchedHits.product();
  magneticField_p_ = magneticField.product();

  rphiHits_hp_ = &rphiHits;
  stereoHits_hp_ = &stereoHits;

  // Keep a table that relates hit pointers to their index (key) in the collections
  rphiKey_.clear();
  stereoKey_.clear();
  // Keep track of which hits have been used already (so a hit is assigned to only one electron)
  hitUsed_.clear();
  matchedHitUsed_.clear();

  unsigned int counter = 0;
  for (SiStripRecHit2DCollection::DataContainer::const_iterator it = rphiHits_p_->data().begin();  it != rphiHits_p_->data().end();  ++it) {
    rphiKey_[&(*it)] = counter;
    hitUsed_[&(*it)] = false;
    counter++;
  }

  counter = 0;
  for (SiStripRecHit2DCollection::DataContainer::const_iterator it = stereoHits_p_->data().begin();  it != stereoHits_p_->data().end();  ++it) {
    stereoKey_[&(*it)] = counter;
    hitUsed_[&(*it)] = false;
    counter++;
  }

  counter = 0;
  for (SiStripMatchedRecHit2DCollection::DataContainer::const_iterator it = matchedHits_p_->data().begin();  it != matchedHits_p_->data().end();  ++it) {
    matchedKey_[&(*it)] = counter;
    matchedHitUsed_[&(*it)] = false;
    counter++;
  }
  
  LogDebug("") << " Leaving prepareEvent " ;
}



// returns true iff an electron was found
// inserts electrons and trackcandiates into electronOut and trackCandidateOut
bool SiStripElectronAlgo::findElectron(reco::SiStripElectronCollection& electronOut,
				       TrackCandidateCollection& trackCandidateOut,
				       const reco::SuperClusterRef& superclusterIn,
				       const TrackerTopology *tTopo)
{
  // Try each of the two charge hypotheses, but only take one
  bool electronSuccess = projectPhiBand(-1., superclusterIn, tTopo);
  bool positronSuccess = projectPhiBand( 1., superclusterIn, tTopo);

  // electron hypothesis did better than electron
  if ((electronSuccess  &&  !positronSuccess)  ||
      (electronSuccess  &&  positronSuccess  &&  redchi2_neg_ <= redchi2_pos_)) {
      
    // Initial uncertainty for tracking
    AlgebraicSymMatrix55 errors;  // makes 5x5 matrix, indexed from (0,0) to (44)
    errors(0,0) = 3.;      // uncertainty**2 in 1/momentum
    errors(1,1) = 0.01;    // uncertainty**2 in lambda (lambda == pi/2 - polar angle theta)
    errors(2,2) = 0.0001;  // uncertainty**2 in phi
    errors(3,3) = 0.01;    // uncertainty**2 in x_transverse (where x is in cm)
    errors(4,4) = 0.01;    // uncertainty**2 in y_transverse (where y is in cm)

#ifdef EDM_ML_DEBUG 
    // JED Debugging possible double hit sort problem
    std::ostringstream debugstr6;
    debugstr6 << " HERE BEFORE SORT electron case " << " \n" ;
    for (std::vector<const TrackingRecHit*>::iterator itHit=outputHits_neg_.begin(); 
	 itHit != outputHits_neg_.end(); ++itHit) {
      debugstr6 <<" HIT "<<((*itHit)->geographicalId()).rawId()<<" \n"
		<<"    Local Position: "<<(*itHit)->localPosition()<<" \n"
		<<"    Global Rho:  "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).perp())
		<<" Phi "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).phi())
		<< " Z "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).z())<< " \n";
    }
    // end of dump 
    
    
    
    // JED call dump 
    debugstr6 << " Calling sort alongMomentum " << " \n"; 
#endif   
 
    std::sort(outputHits_neg_.begin(), outputHits_neg_.end(),
    	      CompareHits(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(tracker_p_)), alongMomentum)));
    
#ifdef EDM_ML_DEBUG 
    debugstr6 << " Done with sort " << " \n";
    
    debugstr6 << " HERE AFTER SORT electron case " << " \n";
    for (std::vector<const TrackingRecHit*>::iterator itHit=outputHits_neg_.begin(); 
	 itHit != outputHits_neg_.end(); ++itHit) {
      debugstr6 <<" HIT "<<((*itHit)->geographicalId()).rawId()<<" \n"
		<<"    Local Position: "<<(*itHit)->localPosition()<<" \n"
		<<"    Global Rho:  "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).perp())
		<<" Phi "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).phi())
		<< " Z "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).z())<< " \n";; 
    }
    // end of dump 

    LogDebug("")<< debugstr6.str();
#endif
    
    //create an OwnVector needed by the classes which will be stored to the Event
    edm::OwnVector<TrackingRecHit> hits;
    for(std::vector<const TrackingRecHit*>::iterator itHit =outputHits_neg_.begin();
	itHit != outputHits_neg_.end();
	++itHit) {
      hits.push_back( (*itHit)->clone());
      if( !(hitUsed_.find(*itHit) != hitUsed_.end()) ) {
	LogDebug("") << " Assert failure " ;
	assert(hitUsed_.find(*itHit) != hitUsed_.end());
      }
      hitUsed_[*itHit] = true;
    }
    
    TrajectoryStateOnSurface state(
				   GlobalTrajectoryParameters(position_neg_, momentum_neg_, -1, magneticField_p_),
				   CurvilinearTrajectoryError(errors),
				   tracker_p_->idToDet(innerhit_neg_->geographicalId())->surface());
    
    
    PTrajectoryStateOnDet const & PTraj = trajectoryStateTransform::persistentState(state, innerhit_neg_->geographicalId().rawId());
    TrajectorySeed trajectorySeed(PTraj, hits, alongMomentum);
    trackCandidateOut.push_back(TrackCandidate(hits, trajectorySeed, PTraj));
    
    electronOut.push_back(reco::SiStripElectron(superclusterIn,
						-1,
						outputRphiHits_neg_,
						outputStereoHits_neg_,
						phiVsRSlope_neg_,
						slope_neg_,
						intercept_neg_ + superclusterIn->position().phi(),
						chi2_neg_,
						ndof_neg_,
						correct_pT_neg_,
						pZ_neg_,
						zVsRSlope_neg_,
						numberOfStereoHits_neg_,
						numberOfBarrelRphiHits_neg_,
						numberOfEndcapZphiHits_neg_));
    
    return true;
  }
  
  // positron hypothesis did better than electron
  if ((!electronSuccess  &&  positronSuccess)  ||
      (electronSuccess  &&  positronSuccess  &&  redchi2_neg_ > redchi2_pos_)) {
      
    // Initial uncertainty for tracking
    AlgebraicSymMatrix55 errors;  // same as abovr
    errors(0,0) = 3.;      // uncertainty**2 in 1/momentum
    errors(1,1) = 0.01;    // uncertainty**2 in lambda (lambda == pi/2 - polar angle theta)
    errors(2,2) = 0.0001;  // uncertainty**2 in phi
    errors(3,3) = 0.01;    // uncertainty**2 in x_transverse (where x is in cm)
    errors(4,4) = 0.01;    // uncertainty**2 in y_transverse (where y is in cm)

#ifdef EDM_ML_DEBUG 
    // JED Debugging possible double hit sort problem
    std::ostringstream debugstr7;
    debugstr7 << " HERE BEFORE SORT Positron case " << " \n";
    for (std::vector<const TrackingRecHit*>::iterator itHit = outputHits_pos_.begin(); 
	 itHit != outputHits_pos_.end(); ++itHit) {
      debugstr7 <<" HIT "<<((*itHit)->geographicalId()).rawId()<<" \n"
		<<"    Local Position: "<<(*itHit)->localPosition()<<" \n"
		<<"    Global Rho:  "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).perp())
		<<" Phi "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).phi())
		<< " Z "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).z())<< " \n";
    }
    // end of dump 
    
    debugstr7 << " Calling sort alongMomentum " << " \n"; 
#endif
    
    std::sort(outputHits_pos_.begin(), outputHits_pos_.end(),
	      CompareHits(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(tracker_p_)), alongMomentum)));
    
#ifdef EDM_ML_DEBUG 
    debugstr7 << " Done with sort " << " \n";
    
    // JED Debugging possible double hit sort problem
    debugstr7 << " HERE AFTER SORT Positron case " << " \n";
    for (std::vector<const TrackingRecHit*>::iterator itHit = outputHits_pos_.begin(); 
	 itHit != outputHits_pos_.end(); ++itHit) {
      debugstr7 <<" HIT "<<((*itHit)->geographicalId()).rawId()<<" \n"
		<<"    Local Position: "<<(*itHit)->localPosition()<<" \n"
		<<"    Global Rho:  "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).perp())
		<<" Phi "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).phi())
		<< " Z "
		<<(((TrackingGeometry*)(tracker_p_))->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition()).z())<< " \n"; 
    }
    // end of dump 
    LogDebug("") << debugstr7.str();
#endif

    //create an OwnVector needed by the classes which will be stored to the Event
    edm::OwnVector<TrackingRecHit> hits;
    for(std::vector<const TrackingRecHit*>::iterator itHit =outputHits_pos_.begin();
	itHit != outputHits_pos_.end();
	++itHit) {
      hits.push_back( (*itHit)->clone());
      assert(hitUsed_.find(*itHit) != hitUsed_.end());
      hitUsed_[*itHit] = true;
    }
    
    TrajectoryStateOnSurface state(
				   GlobalTrajectoryParameters(position_pos_, momentum_pos_, 1, magneticField_p_),
				   CurvilinearTrajectoryError(errors),
				   tracker_p_->idToDet(innerhit_pos_->geographicalId())->surface());
    
    
    PTrajectoryStateOnDet const & PTraj = trajectoryStateTransform::persistentState(state, innerhit_pos_->geographicalId().rawId());
    TrajectorySeed trajectorySeed(PTraj, hits, alongMomentum);
    trackCandidateOut.push_back(TrackCandidate(hits, trajectorySeed, PTraj));
    
    electronOut.push_back(reco::SiStripElectron(superclusterIn,
						1,
						outputRphiHits_pos_,
						outputStereoHits_pos_,
						phiVsRSlope_pos_,
						slope_pos_,
						intercept_pos_ + superclusterIn->position().phi(),
						chi2_pos_,
						ndof_pos_,
						correct_pT_pos_,
						pZ_pos_,
						zVsRSlope_pos_,
						numberOfStereoHits_pos_,
						numberOfBarrelRphiHits_pos_,
						numberOfEndcapZphiHits_pos_));
    
    return true;
  }
  
  return false;
}

// inserts pointers to good hits into hitPointersOut
// selects hits on DetIds that have no more than maxHitsOnDetId_
// selects from stereo if stereo == true, rphi otherwise
// selects from TID or TEC if endcap == true, TIB or TOB otherwise
void SiStripElectronAlgo::coarseHitSelection(std::vector<const SiStripRecHit2D*>& hitPointersOut,
					     const TrackerTopology *tTopo,
					     bool stereo, bool endcap)
{
  // This function is not time-efficient.  If you want to improve the
  // speed of the algorithm, you'll probably want to change this
  // function.  There may be a more efficienct way to extract hits,
  // and it would definitely help to put a geographical cut on the
  // DetIds.  (How does one determine the global position of a given
  // DetId?  Is tracker_p_->idToDet(id)->surface().toGlobal(LocalPosition(0,0,0))
  // expensive?)

  // Loop over the detector ids
  SiStripRecHit2DCollection::const_iterator itdet = (stereo ? stereoHits_p_->begin() : rphiHits_p_->begin());
  SiStripRecHit2DCollection::const_iterator eddet = (stereo ? stereoHits_p_->end()   : rphiHits_p_->end()  );
  for (; itdet != eddet; ++itdet) {
    // Get the hits on this detector id
    SiStripRecHit2DCollection::DetSet hits = *itdet;
    DetId id(hits.detId());

    // Count the number of hits on this detector id
    unsigned int numberOfHits = hits.size();
      
    // Only take the hits if there aren't too many
    // (Would it be better to loop only once, fill a temporary list,
    // and copy that if numberOfHits <= maxHitsOnDetId_?)
    if (numberOfHits <= maxHitsOnDetId_) {
      for (SiStripRecHit2DCollection::DetSet::const_iterator hit = hits.begin();  
           hit != hits.end();  ++hit) {
        // check that hit is valid first !
	if(!(*hit).isValid()) {
	  LogDebug("") << " InValid hit skipped in coarseHitSelection " << std::endl ;
	  continue ;
	}
        std::string theDet = "null";
        int theLayer = -999;
        bool isStereoDet = false ;
        if(tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB) { 
          theDet = "TIB" ;
          theLayer = tTopo->tibLayer(id); 
          if(tTopo->tibStereo(id)==1) { isStereoDet = true ; }
        } else if
          (tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB) { 
          theDet = "TOB" ;
          theLayer = tTopo->tobLayer(id); 
          if(tTopo->tobStereo(id)==1) { isStereoDet = true ; }
        }else if
          (tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TID) { 
          theDet = "TID" ;
          theLayer = tTopo->tidWheel(id);  // or ring  ?
          if(tTopo->tidStereo(id)==1) { isStereoDet = true ; }
        }else if
          (tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TEC) { 
          theDet = "TEC" ;
          theLayer = tTopo->tecWheel(id);  // or ring or petal ?
          if(tTopo->tecStereo(id)==1) { isStereoDet = true ; }
        } else {
          LogDebug("") << " UHOH BIG PROBLEM - Unrecognized SI Layer" ;
          LogDebug("") << " Det "<< theDet << " Lay " << theLayer ;
          assert(1!=1) ;
        }

	if ((endcap  &&  stereo && (theDet=="TID" || theDet== "TEC") && isStereoDet ) ||
            (endcap  &&  !stereo && (theDet=="TID" || theDet== "TEC") && !isStereoDet )  ||
            (!endcap  && stereo && (theDet=="TIB" || theDet=="TOB") &&  isStereoDet )    ||
            (!endcap  &&  !stereo && (theDet=="TIB" || theDet=="TOB" )&& !isStereoDet )
            ) {  
              

	  hitPointersOut.push_back(&(*hit));

	} // end if this is the right subdetector
      } // end loop over hits
    } // end if this detector id doesn't have too many hits on it
  } // end loop over detector ids
}

// select all matched hits for now

void SiStripElectronAlgo::coarseMatchedHitSelection(std::vector<const SiStripMatchedRecHit2D*>& coarseMatchedHitPointersOut)
{
  
  // Loop over the detector ids
  SiStripMatchedRecHit2DCollection::const_iterator itdet = matchedHits_p_->begin(), eddet = matchedHits_p_->end();
  for (; itdet != eddet; ++itdet) {
    
    // Get the hits on this detector id
    SiStripMatchedRecHit2DCollection::DetSet hits = *itdet ;
    
    // Count the number of hits on this detector id
    unsigned int numberOfHits = 0;
    for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = hits.begin();  hit != hits.end();  ++hit) {
      if ( !((hit->geographicalId()).subdetId() == StripSubdetector::TIB) &&
           !( (hit->geographicalId()).subdetId() == StripSubdetector::TOB )) { break;}
      numberOfHits++;
      if (numberOfHits > maxHitsOnDetId_) { break; }
    }
    
    // Only take the hits if there aren't too many
    if (numberOfHits <= maxHitsOnDetId_) {
      for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = hits.begin();  hit != hits.end();  ++hit) {
	if(!(*hit).isValid()) {
	  LogDebug("") << " InValid hit skipped in coarseMatchedHitSelection " << std::endl ;
	  continue ;
	}
        if ( !((hit->geographicalId()).subdetId() == StripSubdetector::TIB) &&
             !( (hit->geographicalId()).subdetId() == StripSubdetector::TOB )) { break;}
        
        coarseMatchedHitPointersOut.push_back(&(*hit));
      } // end loop over hits
      
    } // end if this detector id doesn't have too many hits on it
  } // end loop over detector ids
  

}// end of matchedHitSelection




// projects a phi band of width phiBandWidth_ from supercluster into tracker (given a chargeHypothesis)
// fills *_pos_ or *_neg_ member data with the results
// returns true iff the electron/positron passes cuts
bool SiStripElectronAlgo::projectPhiBand(float chargeHypothesis, const reco::SuperClusterRef& superclusterIn,
					 const TrackerTopology *tTopo)
{
  // This algorithm projects a phi band into the tracker three times:
  // (a) for all stereo hits, (b) for barrel rphi hits, and (c) for
  // endcap zphi hits.  While accumulating stereo hits in step (a),
  // we fit r vs z to a line.  This resolves the ambiguity in z for
  // rphi hits and the ambiguity in r for zphi hits.  We can then cut
  // on the z of rphi hits (a little wider than one strip length),
  // and we can convert the z of zphi hits into r to apply the phi
  // band cut.  (We don't take advantage of the endcap strips'
  // segmentation in r.)
  // 
  // As we project a phi band into the tracker, we count hits within
  // that band and performs a linear fit for phi vs r.  The number of
  // hits and reduced chi^2 from the fit are used to select a good
  // candidate.

  // set limits on Si hits - smallest error that makes sense = 1 micron ?
  static const double rphiHitSmallestError = 0.0001 ;
  static const double stereoHitSmallestError = 0.0001 ;
  //not used since endcap is not implemented  
  // static const double zphiHitSmallestError = 0.0001 ;


  static const double stereoPitchAngle = 0.100 ; // stereo angle in rad
  static const double cos2SiPitchAngle = cos(stereoPitchAngle)*cos(stereoPitchAngle) ;
  static const double sin2SiPitchAngle = sin(stereoPitchAngle)*sin(stereoPitchAngle) ;
  // overall misalignment fudge to be added in quad to position errors.
  // this is a rough approx to values reported in tracking meet 5/16/2007
  static const double rphiErrFudge = 0.0200 ;
  static const double stereoErrFudge = 0.0200 ;

  // max chi2 of a hit on an SiDet relative to the prediction
  static const double chi2HitMax = 25.0 ;

  // Minimum number of hits to consider on a candidate
  static const unsigned int nHitsLeftMinimum = 3  ;

  // Create and fill vectors of pointers to hits
  std::vector<const SiStripRecHit2D*> stereoHits;
  std::vector<const SiStripRecHit2D*> rphiBarrelHits;
  std::vector<const SiStripRecHit2D*> zphiEndcapHits;

  //                                 stereo? endcap?
  coarseHitSelection(stereoHits, tTopo,    true,   false);

  // skip endcap stereo for now
  //  LogDebug("") << " Getting endcap stereo hits " ;
  // coarseHitSelection(stereoHits,     true,   true);

  std::ostringstream debugstr1;
  debugstr1 << " Getting barrel rphi hits " << " \n" ;

  coarseHitSelection(rphiBarrelHits, tTopo, false,  false);

  //  LogDebug("") << " Getting endcap zphi hits " ;
  //  coarseHitSelection(zphiEndcapHits, false,  true);

  debugstr1 << " Getting matched hits "  << " \n" ;
  std::vector<const SiStripMatchedRecHit2D*> matchedHits;
  coarseMatchedHitSelection(matchedHits);


  // Determine how to project from the supercluster into the tracker
  double energy = superclusterIn->energy();
  double pT = energy * superclusterIn->position().rho()/sqrt(superclusterIn->x()*superclusterIn->x() +
							     superclusterIn->y()*superclusterIn->y() +
							     superclusterIn->z()*superclusterIn->z());
  // cf Jackson p. 581-2, a little geometry
  double phiVsRSlope = -3.00e-3 * chargeHypothesis * magneticField_p_->inTesla(GlobalPoint(superclusterIn->x(), superclusterIn->y(), 0.)).z() / pT / 2.;

  // Shorthand for supercluster radius, z
  const double scr = superclusterIn->position().rho();
  const double scz = superclusterIn->position().z();

  // These are used to fit all hits to a line in phi(r)
  std::vector<bool> uselist;
  std::vector<double> rlist, philist, w2list;
  std::vector<int> typelist;  // stereo = 0, rphi barrel = 1, and zphi disk = 2 (only used in this function)
  std::vector<const SiStripRecHit2D*> hitlist;

  std::vector<bool> matcheduselist;
  std::vector<const SiStripMatchedRecHit2D*> matchedhitlist;

  // These are used to fit the stereo hits to a line in z(r), constrained to pass through supercluster
  double zSlopeFitNumer = 0.;
  double zSlopeFitDenom = 0.;

  
  debugstr1 << " There are a total of " << stereoHits.size()  << " stereoHits in this event " << " \n" 
	    << " There are a total of " <<  rphiBarrelHits.size() << " rphiBarrelHits in this event " << " \n"
	    << " There are a total of " <<  zphiEndcapHits.size() << " zphiEndcapHits in this event " << " \n\n";


  LogDebug("") << debugstr1.str() ;
  
  /////////////////


  // Loop over all matched hits
  // make a list of good matched rechits.
  // in the stereo and rphi loops check to see if the hit is associated with a matchedhit
  LogDebug("") << " Loop over matched hits " << " \n";

  for (std::vector<const SiStripMatchedRecHit2D*>::const_iterator hit = matchedHits.begin() ;
       hit != matchedHits.end() ; ++ hit) {
    
    assert(matchedHitUsed_.find(*hit) != matchedHitUsed_.end());

    if (!matchedHitUsed_[*hit]) {

      // Calculate the 3-D position of this hit
      GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double r = sqrt(position.x()*position.x() + position.y()*position.y());
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster
      double z = position.z();

      // Cut a triangle in the z-r plane using the supercluster's eta/dip angle information
      // and the fact that the electron originated *near* the origin
      if ((-originUncertainty_ + (scz + originUncertainty_)*(r/scr)) < z  &&  z < (originUncertainty_ + (scz - originUncertainty_)*(r/scr))) {
	 
	// Cut a narrow band around the supercluster's projection in phi
	if (unwrapPhi((r-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + phiBandWidth_)) {
	 

	  matcheduselist.push_back(true);
          matchedhitlist.push_back(*hit);
	  


	  // Use this hit to fit z(r)
	  zSlopeFitNumer += -(scr - r) * (z - scz);
	  zSlopeFitDenom += (scr - r) * (scr - r);
	    
	} // end cut on phi band
      } // end cut on electron originating *near* the origin
    } // end assign disjoint sets of hits to electrons
  } // end loop over matched hits

  // Calculate the linear fit for z(r)
  double zVsRSlope;
  if (zSlopeFitDenom > 0.) {
    zVsRSlope = zSlopeFitNumer / zSlopeFitDenom;
  }
  else {
    // zVsRSlope assumes electron is from origin if there were no stereo hits
    zVsRSlope = scz/scr;
  }

  //  // Loop over all stereo hits
  LogDebug("") << " Loop over stereo hits" << " \n";

  // check if the stereo hit is matched to one of the matched hit
  unsigned int numberOfStereoHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = stereoHits.begin();  hit != stereoHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());

    // Calculate the 3-D position of this hit
    GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
    double r_stereo = sqrt(position.x()*position.x() + position.y()*position.y());
    double phi_stereo = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster
    //    double z_stereo = position.z();

    // stereo is has a pitch of 100 mrad - so consider both components
    double r_stereo_err = sqrt((*hit)->localPositionError().xx()*cos2SiPitchAngle +
                               (*hit)->localPositionError().yy()*sin2SiPitchAngle ) ; 



    // make sure that the error for this hit is sensible, ie > 1 micron
    // otherwise skip this hit
    if(r_stereo_err > stereoHitSmallestError ) {
      r_stereo_err = sqrt(r_stereo_err*r_stereo_err+stereoErrFudge*stereoErrFudge);
 
      OmniClusterRef const & stereocluster=(*hit)->omniClusterRef();
    
      bool thisHitIsMatched = false ;

      if (!hitUsed_[*hit]) {
 
	unsigned int matcheduselist_size = matcheduselist.size();
	for (unsigned int i = 0;  i < matcheduselist_size;  i++) {
	  if (matcheduselist[i]) {
            OmniClusterRef const &  mystereocluster = matchedhitlist[i]->stereoClusterRef();
	    if( stereocluster == mystereocluster ) {
	      thisHitIsMatched = true ;
	      //    LogDebug("")<< "     This hit is matched " << tracker_p_->idToDet(matchedhitlist[i]->stereoHit()->geographicalId())->surface().toGlobal(matchedhitlist[i]->stereoHit()->localPosition()) << std::endl;
	      //      break ;
	    }
	  } // check if matcheduselist okay 
	}// loop over matched hits 
      
	if(thisHitIsMatched) {
	  // Use this hit to fit phi(r)
	  uselist.push_back(true);
	  rlist.push_back(r_stereo);
	  philist.push_back(phi_stereo);
	  w2list.push_back(1./(r_stereo_err/r_stereo)/(r_stereo_err/r_stereo));  // weight**2 == 1./uncertainty**2
	  typelist.push_back(0);
	  hitlist.push_back(*hit);
	} // thisHitIsMatched
      } //  if(!hitUsed)

    } //  end of check on hit position error size 

  } // end loop over stereo hits
  
  LogDebug("") << " There are " << uselist.size()  << " good hits after stereo loop "  ;
 

  // Loop over barrel rphi hits
  LogDebug("") << " Looping over barrel rphi hits " ;
  unsigned int rphiMatchedCounter = 0 ;
  unsigned int rphiUnMatchedCounter = 0 ;
  unsigned int numberOfBarrelRphiHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = rphiBarrelHits.begin();  hit != rphiBarrelHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());
    // Calculate the 2.5-D position of this hit
    GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
    double r = sqrt(position.x()*position.x() + position.y()*position.y());
    double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster
    double z = position.z();
    double r_mono_err = sqrt((*hit)->localPositionError().xx()) ;

    // only consider hits with errors that make sense
    if( r_mono_err > rphiHitSmallestError) {
      // inflate the reported error
      r_mono_err=sqrt(r_mono_err*r_mono_err+rphiErrFudge*rphiErrFudge);

      OmniClusterRef const & monocluster=(*hit)->omniClusterRef();
    

      if (!hitUsed_[*hit]) {
	if( (tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB  && 
	     (tTopo->tibLayer((*hit)->geographicalId())==1 || tTopo->tibLayer((*hit)->geographicalId())==2)) || 
	    (tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB  
	     && (tTopo->tobLayer((*hit)->geographicalId())==1 || tTopo->tobLayer((*hit)->geographicalId())==2)) ) {
	  bool thisHitIsMatched = false ;
	  unsigned int matcheduselist_size = matcheduselist.size();
	  for (unsigned int i = 0;  i < matcheduselist_size;  i++) {
	    if (matcheduselist[i]) {
              OmniClusterRef const &  mymonocluster = matchedhitlist[i]->monoClusterRef();
	      if( monocluster == mymonocluster ) {
		thisHitIsMatched = true ;
	      } 
	    } // check if matcheduselist okay 
	  }// loop over matched hits 
        
        
	  if( thisHitIsMatched ) {
	    // Use this hit to fit phi(r)
	    uselist.push_back(true);
	    rlist.push_back(r);
	    philist.push_back(phi);
	    w2list.push_back(1./(r_mono_err/r)/(r_mono_err/r));  // weight**2 == 1./uncertainty**2
	    typelist.push_back(1);
	    hitlist.push_back(*hit);
	    rphiMatchedCounter++;
	  } // end of matched hit check

	} else {


	  // The expected z position of this hit, according to the z(r) fit
	  double zFit = zVsRSlope * (r - scr) + scz;
        
	  // Cut on the Z of the strip
	  // TIB strips are 11 cm long, TOB strips are 19 cm long (can I get these from a function?)
	  if ((tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB  && 
	       std::abs(z - zFit) < 12.)  ||
	      (tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB  && 
	       std::abs(z - zFit) < 20.)    ) {
          
	    // Cut a narrow band around the supercluster's projection in phi
	    if (unwrapPhi((r-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + phiBandWidth_)) {
            
	      // Use this hit to fit phi(r)
	      uselist.push_back(true);
	      rlist.push_back(r);
	      philist.push_back(phi);
	      w2list.push_back(1./(r_mono_err/r)/(r_mono_err/r));  // weight**2 == 1./uncertainty**2
	      typelist.push_back(1);
	      hitlist.push_back(*hit);
	      rphiUnMatchedCounter++;
            
	    } // end cut on phi band
	  } // end cut on strip z
	} // loop over TIB/TOB layer 1,2 
      } // end assign disjoint sets of hits to electrons
    } // end of check on rphi hit position error size
  } // end loop over barrel rphi hits

  LogDebug("") << " There are " << rphiMatchedCounter <<" matched rphi hits"; 
  LogDebug("") << " There are " << rphiUnMatchedCounter <<" unmatched rphi hits";
  LogDebug("") << " There are " << uselist.size() << " good stereo+rphi hits " ;






  ////////////////

  // Loop over endcap zphi hits
  LogDebug("") << " Looping over barrel zphi hits " ;


  unsigned int numberOfEndcapZphiHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = zphiEndcapHits.begin();  
       hit != zphiEndcapHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());
    if (!hitUsed_[*hit]) {
      // Calculate the 2.5-D position of this hit
      GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double z = position.z();
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster
      //      double r=sqrt(position.x()*position.x()+position.y()*position.y()) ;

      // The expected r position of this hit, according to the z(r) fit
      double rFit = (z - scz)/zVsRSlope + scr;

      // I don't know the r widths of the endcap strips, otherwise I
      // would apply a cut on r similar to the rphi z cut

      // Cut a narrow band around the supercluster's projection in phi,
      // and be sure the hit isn't in a reflected band (extrapolation of large z differences into negative r)
      if (rFit > 0.  &&
	  unwrapPhi((rFit-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((rFit-scr)*phiVsRSlope + phiBandWidth_)) {

	// Use this hit to fit phi(r)
	uselist.push_back(true);
	rlist.push_back(rFit);
	philist.push_back(phi);
	w2list.push_back(1./(0.05/rFit)/(0.05/rFit));  // weight**2 == 1./uncertainty**2
	typelist.push_back(2);
	hitlist.push_back(*hit);

      } // end cut on phi band
    } // end assign disjoint sets of hits to electrons
  } // end loop over endcap zphi hits
 
  LogDebug("") << " There are " << uselist.size() << " good stereo+rphi+zphi hits " ;
  //////////////////////

#ifdef EDM_ML_DEBUG 
  std::ostringstream debugstr5;
  debugstr5 << " List of hits before uniqification " << " \n" ;
  for (unsigned int i = 0;  i < uselist.size();  i++) {
    if ( uselist[i] ) {
      const SiStripRecHit2D* hit = hitlist[i];
      debugstr5 << " DetID " << ((hit)->geographicalId()).rawId()
		<< " R " << rlist[i] 
		<< " Phi " << philist[i]
		<< " Weight " << w2list[i] 
		<< " PhiPred " << (rlist[i]-scr)*phiVsRSlope  
		<< " Chi2 " << (philist[i]-(rlist[i]-scr)*phiVsRSlope)*(philist[i]-(rlist[i]-scr)*phiVsRSlope)*w2list[i]
		<< " \n" ;
    }
  }
  debugstr5 << " \n\n\n" ;
  
  debugstr5 << " Counting number of unique detectors " << " \n" ;

  debugstr5 << " These are all the detectors with hits " << " \n";
#endif
  // Loop over hits, and find the best hit for each SiDetUnit - keep only those
  // get a list of DetIds in uselist
  std::vector<uint32_t> detIdList ;

  for (unsigned int i = 0;  i < uselist.size();  i++) {
    if (uselist[i]) {
      const SiStripRecHit2D* hit = hitlist[i];
      uint32_t detIDUsed = ((hit)->geographicalId()).rawId() ;
#ifdef EDM_ML_DEBUG 
      debugstr5<< " DetId " << detIDUsed << "\n";
#endif
      detIdList.push_back(detIDUsed) ;
    }
  }

#ifdef EDM_ML_DEBUG 
  debugstr5 << " There are " << detIdList.size() << " hits on detectors \n";
#endif
  // now sort and then uniq this list of detId
  std::sort(detIdList.begin(), detIdList.end());
  detIdList.erase(
		  std::unique(detIdList.begin(), detIdList.end()), detIdList.end());
#ifdef EDM_ML_DEBUG 
  debugstr5 << " There are " << detIdList.size() << " unique detectors \n";
#endif
  //now we have a list of unique detectors used


#ifdef EDM_ML_DEBUG 
  debugstr5 << " Looping over detectors to choose best hit " << " \n";
#endif
  // Loop over detectors ID and hits to create list of hits on same DetId
  for (unsigned int idet = 0 ; idet < detIdList.size() ; idet++ ) {
    for (unsigned int i = 0;  i+1 < uselist.size();  i++) {
      if (uselist[i]) {
	// Get Chi2 of this hit relative to predicted hit
	const SiStripRecHit2D* hit1 = hitlist[i];
	double r_hit1 = rlist[i];
	double phi_hit1 = philist[i];
	double w2_hit1 = w2list[i] ;
	double phi_pred1 = (r_hit1-scr)*phiVsRSlope ; 
	double chi1 = (phi_hit1-phi_pred1)*(phi_hit1-phi_pred1)*w2_hit1;
	if(detIdList[idet]== ((hit1)->geographicalId()).rawId()) {
	  for (unsigned int j = i+1;  j < uselist.size();  j++) {
	    if (uselist[j]) {
	      const SiStripRecHit2D* hit2 = hitlist[j];
	      if(detIdList[idet]== ((hit2)->geographicalId()).rawId()) {
#ifdef EDM_ML_DEBUG 
		debugstr5 << " Found 2 hits on same Si Detector " 
			  << ((hit2)->geographicalId()).rawId() << "\n";
#endif
		// Get Chi2 of this hit relative to predicted hit
		double r_hit2 = rlist[j];
		double phi_hit2 = philist[j];
		double w2_hit2 = w2list[j] ;
		double phi_pred2 = (r_hit2-scr)*phiVsRSlope ; 
		double chi2 = (phi_hit2-phi_pred2)*(phi_hit2-phi_pred2)*w2_hit2;
#ifdef EDM_ML_DEBUG 
		debugstr5 << " Chi1 " << chi1 << " Chi2 " << chi2 <<"\n";
#endif
		if( chi1< chi2 ){
		  uselist[j] = 0;
		}else{
		  uselist[i] = 0;
		}

	      } // end of Det check
	    } // end of j- usehit check
	  } // end of j hit list loop
	  
	} // end of detector check
      } // end of uselist check
    } // end of i-hit list loop


  } // end of DetId loop


  
  // now let's through out hits with a predicted chi > chi2HitMax 
  for ( unsigned int i = 0;  i < uselist.size();  i++ ) { 
    if ( uselist[i] ) { 
      double localchi2 = (philist[i]-(rlist[i]-scr)*phiVsRSlope)*(philist[i]-(rlist[i]-scr)*phiVsRSlope)*w2list[i] ;
      if(localchi2 > chi2HitMax ) {
#ifdef EDM_ML_DEBUG 
        const SiStripRecHit2D* hit = hitlist[i];
 	debugstr5 << " Throwing out "
		  <<" DetID " << ((hit)->geographicalId()).rawId()
		  << " R " << rlist[i] 
		  << " Phi " << philist[i]
		  << " Weight " << w2list[i] 
		  << " PhiPred " << (rlist[i]-scr)*phiVsRSlope  
		  << " Chi2 " << (philist[i]-(rlist[i]-scr)*phiVsRSlope)*(philist[i]-(rlist[i]-scr)*phiVsRSlope)*w2list[i]
		  << " \n" ;
#endif
	uselist[i]=0 ;
      }
    }
  }

#ifdef EDM_ML_DEBUG 
  debugstr5 << " List of hits after uniqification " << " \n" ;
  for (unsigned int i = 0;  i < uselist.size();  i++) {
    if ( uselist[i] ) {
      const SiStripRecHit2D* hit = hitlist[i];
      debugstr5 << " DetID " << ((hit)->geographicalId()).rawId()
		<< " R " << rlist[i] 
		<< " Phi " << philist[i]
		<< " Weight " << w2list[i] 
		<< " PhiPred " << (rlist[i]-scr)*phiVsRSlope  
		<< " Chi2 " << (philist[i]-(rlist[i]-scr)*phiVsRSlope)*(philist[i]-(rlist[i]-scr)*phiVsRSlope)*w2list[i]
		<< " \n" ;
    }
  }
  debugstr5 << " \n\n\n" ;
#endif

  // need to check that there are more than 2 hits left here!
  unsigned int nHitsLeft =0;
  for (unsigned int i = 0;  i < uselist.size();  i++) {
    if ( uselist[i] ) {
      nHitsLeft++;
    }
  }
  if(nHitsLeft < nHitsLeftMinimum ) {
#ifdef EDM_ML_DEBUG 
    debugstr5 << " Too few hits "<< nHitsLeft 
	      << " left - return false " << " \n";
#endif
    return false ;
  }
#ifdef EDM_ML_DEBUG 
  LogDebug("") << debugstr5.str();
#endif
  /////////////////////
  
  // Calculate a linear phi(r) fit and drop hits until the biggest contributor to chi^2 is less than maxNormResid_
  bool done = false;
  double intercept = 0 ;
  double slope = 0 ;
  double chi2 = 0;

  std::ostringstream debugstr4;
  debugstr4 <<" Calc of phi(r) "<<" \n";

  while (!done) {
    // Use an iterative update of <psi>, <r> etc instead in an
    // attempt to minize divisions of  large numbers
    // The linear fit  psi = slope*r + intercept
    //             slope is (<psi*r>-<psi>*<r>) / (<r^2>-<r>^2>)
    //             intercept is <psi> - slope*<r>
    
    double phiBar   = 0.;
    double phiBarOld   = 0.;
    double rBar  = 0.;
    double rBarOld  = 0.;
    double r2Bar = 0.;
    double r2BarOld = 0.;
    double phirBar = 0.;
    double phirBarOld = 0.;
    double totalWeight = 0.;
    double totalWeightOld = 0.; 
    unsigned int uselist_size = uselist.size();
    for (unsigned int i = 0;  i < uselist_size;  i++) {
      if (uselist[i]) {
        double r = rlist[i];
        double phi = philist[i];
        double weight2 = w2list[i];
	
        totalWeight = totalWeightOld + weight2 ;
	
        //weight2 is 1/sigma^2. Typically sigma is 100micron pitch
        // over root(12) = 30 microns so weight2 is about 10^5-10^6
	
        double totalWeightRatio = totalWeightOld/totalWeight ;
        double localWeightRatio = weight2/totalWeight ;
	
        phiBar = phiBarOld*totalWeightRatio + phi*localWeightRatio ; 
        rBar = rBarOld*totalWeightRatio + r*localWeightRatio ; 
        r2Bar= r2BarOld*totalWeightRatio + r*r*localWeightRatio ; 
        phirBar = phirBarOld*totalWeightRatio + phi*r*localWeightRatio ;
	
	totalWeightOld = totalWeight ;
        phiBarOld = phiBar ;
        rBarOld = rBar ;
        r2BarOld = r2Bar ;
        phirBarOld = phirBar ;  
#ifdef EDM_ML_DEBUG 
	debugstr4 << " totalWeight " << totalWeight 
                  << " totalWeightRatio " << totalWeightRatio
                  << " localWeightRatio "<< localWeightRatio
                  << " phiBar " << phiBar
                  << " rBar " << rBar 
                  << " r2Bar " << r2Bar
                  << " phirBar " << phirBar 
                  << " \n ";
#endif

      } // end of use hit loop
    } // end of hit loop to calculate a linear fit
    slope = (phirBar-phiBar*rBar)/(r2Bar-rBar*rBar);
    intercept = phiBar-slope*rBar ;

    debugstr4 << " end of loop slope " << slope 
              << " intercept " << intercept << " \n";

    // Calculate chi^2
    chi2 = 0.;
    double biggest_normresid = -1.;
    unsigned int biggest_index = 0;
    for (unsigned int i = 0;  i < uselist_size;  i++) {
      if (uselist[i]) {
	double r = rlist[i];
	double phi = philist[i];
	double weight2 = w2list[i];

	double normresid = weight2 * (phi - intercept - slope*r)*(phi - intercept - slope*r);
	chi2 += normresid;

	if (normresid > biggest_normresid) {
	  biggest_normresid = normresid;
	  biggest_index = i;
	}
      }
    } // end loop over hits to calculate chi^2 and find its biggest contributer

    if (biggest_normresid > maxNormResid_) {
#ifdef EDM_ML_DEBUG
      debugstr4 << "Dropping hit from fit due to Chi2 " << " \n" ;
      const SiStripRecHit2D* hit = hitlist[biggest_index];
      debugstr4 << " DetID " << ((hit)->geographicalId()).rawId()
		<< " R " << rlist[biggest_index]
		<< " Phi " << philist[biggest_index]
		<< " Weight " << w2list[biggest_index]
		<< " PhiPred " << (rlist[biggest_index]-scr)*phiVsRSlope  
		<< " Chi2 " << (philist[biggest_index]-(rlist[biggest_index]-scr)*phiVsRSlope)*(philist[biggest_index]-(rlist[biggest_index]-scr)*phiVsRSlope)*w2list[biggest_index] 
		<< " normresid " <<  biggest_normresid
		<< " \n\n";
#endif
      uselist[biggest_index] = false;
    }
    else {
      done = true;
    }
  } // end loop over trial fits
#ifdef EDM_ML_DEBUG 
  debugstr4 <<" Fit " 
            << " Intercept  " << intercept
            << " slope " << slope 
            << " chi2 " << chi2
            << " \n" ;

  LogDebug("") <<   debugstr4.str() ;
#endif

  // Now we have intercept, slope, and chi2; uselist to tell us which hits are used, and hitlist for the hits

  // Identify the innermost hit
  const SiStripRecHit2D* innerhit = (SiStripRecHit2D*)(0);
  double innerhitRadius = -1.;  // meaningless until innerhit is defined

  // Copy hits into an OwnVector, which we put in the TrackCandidate
  std::vector<const TrackingRecHit*> outputHits;
  // Reference rphi and stereo hits into RefVectors, which we put in the SiStripElectron
  std::vector<SiStripRecHit2D> outputRphiHits;
  std::vector<SiStripRecHit2D> outputStereoHits;

  typedef edm::Ref<SiStripRecHit2DCollection,SiStripRecHit2D> SiStripRecHit2DRef;


  for (unsigned int i = 0;  i < uselist.size();  i++) {
    if (uselist[i]) {
      double r = rlist[i];
      const SiStripRecHit2D* hit = hitlist[i];

      // Keep track of the innermost hit
      if (innerhit == (SiStripRecHit2D*)(0)  ||  r < innerhitRadius) {
	innerhit = hit;
	innerhitRadius = r;
      }
	 
      if (typelist[i] == 0) {
	numberOfStereoHits++;

	// Copy this hit for the TrajectorySeed
	outputHits.push_back(hit);
	outputStereoHits.push_back(*hit);
      }
      else if (typelist[i] == 1) {
	numberOfBarrelRphiHits++;

	// Copy this hit for the TrajectorySeed
	outputHits.push_back(hit);
	outputRphiHits.push_back(*hit);
      }
      else if (typelist[i] == 2) {
	numberOfEndcapZphiHits++;

	// Copy this hit for the TrajectorySeed
	outputHits.push_back(hit);
	outputRphiHits.push_back(*hit);
      }
    }
  } // end loop over all hits, after having culled the ones with big residuals

  unsigned int totalNumberOfHits = numberOfStereoHits + numberOfBarrelRphiHits + numberOfEndcapZphiHits;
  double reducedChi2 = (totalNumberOfHits > 2 ? chi2 / (totalNumberOfHits - 2) : 1e10);

  // Select this candidate if it passes minHits_ and maxReducedChi2_ cuts
  if (totalNumberOfHits >= minHits_  &&  reducedChi2 <= maxReducedChi2_) {
    // GlobalTrajectoryParameters evaluated at the position of the innerhit
    GlobalPoint position = tracker_p_->idToDet(innerhit->geographicalId())->surface().toGlobal(innerhit->localPosition());

    // Use our phi(r) linear fit to correct pT (pT is inversely proportional to slope)
    // (By applying a correction instead of going back to first
    // principles, any correction to the phiVsRSlope definition
    // above will be automatically propagated here.)
    double correct_pT = pT * phiVsRSlope / slope;

    // Our phi(r) linear fit returns phi relative to the supercluster phi for a given radius
    double phi = intercept + slope*sqrt(position.x()*position.x() + position.y()*position.y()) + superclusterIn->position().phi();

    // Our z(r) linear fit gives us a better measurement of eta/dip angle
    double pZ = correct_pT * zVsRSlope;

    // Now we can construct a trajectory momentum out of linear fits to hits
    GlobalVector momentum = GlobalVector(correct_pT*cos(phi), correct_pT*sin(phi), pZ);

    if (chargeHypothesis > 0.) {
      redchi2_pos_ = chi2 / double(totalNumberOfHits - 2);
      position_pos_ = position;
      momentum_pos_ = momentum;
      innerhit_pos_ = innerhit;
      outputHits_pos_ = outputHits;
      outputRphiHits_pos_ = outputRphiHits;
      outputStereoHits_pos_ = outputStereoHits;
      phiVsRSlope_pos_ = phiVsRSlope;
      slope_pos_ = slope;
      intercept_pos_ = intercept;
      chi2_pos_ = chi2;
      ndof_pos_ = totalNumberOfHits - 2;
      correct_pT_pos_ = correct_pT;
      pZ_pos_ = pZ;
      zVsRSlope_pos_ = zVsRSlope;
      numberOfStereoHits_pos_ = numberOfStereoHits;
      numberOfBarrelRphiHits_pos_ = numberOfBarrelRphiHits;
      numberOfEndcapZphiHits_pos_ = numberOfEndcapZphiHits;
    }
    else {
      redchi2_neg_ = chi2 / double(totalNumberOfHits - 2);
      position_neg_ = position;
      momentum_neg_ = momentum;
      innerhit_neg_ = innerhit;
      outputHits_neg_ = outputHits;
      outputRphiHits_neg_ = outputRphiHits;
      outputStereoHits_neg_ = outputStereoHits;
      phiVsRSlope_neg_ = phiVsRSlope;
      slope_neg_ = slope;
      intercept_neg_ = intercept;
      chi2_neg_ = chi2;
      ndof_neg_ = totalNumberOfHits - 2;
      correct_pT_neg_ = correct_pT;
      pZ_neg_ = pZ;
      zVsRSlope_neg_ = zVsRSlope;
      numberOfStereoHits_neg_ = numberOfStereoHits;
      numberOfBarrelRphiHits_neg_ = numberOfBarrelRphiHits;
      numberOfEndcapZphiHits_neg_ = numberOfEndcapZphiHits;
    }

    LogDebug("") << " return from projectFindBand with True \n" << std::endl ;
    return true;
  } // end if this is a good electron candidate

  // Signal for a failed electron candidate
  LogDebug("") << " return from projectFindBand with False \n" << std::endl ;
  return false;
}

//
// const member functions
//

//
// static member functions
//
 
