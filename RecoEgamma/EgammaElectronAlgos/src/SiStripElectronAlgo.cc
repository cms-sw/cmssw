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
// $Id: SiStripElectronAlgo.cc,v 1.13 2006/09/19 19:28:36 rahatlou Exp $
//

// system include files

// user include files
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxFittedHel.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxHit.hh"
#include "RecoTracker/RoadSearchHelixMaker/interface/DcxTrackCandidatesToTracks.hh"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"

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
				       const edm::ESHandle<MagneticField>& magneticField)
{
  tracker_p_ = tracker.product();
  rphiHits_p_ = rphiHits.product();
  stereoHits_p_ = stereoHits.product();
  magneticField_p_ = magneticField.product();

  rphiHits_hp_ = &rphiHits;
  stereoHits_hp_ = &stereoHits;

  // Keep a table that relates hit pointers to their index (key) in the collections
  rphiKey_.clear();
  stereoKey_.clear();
  // Keep track of which hits have been used already (so a hit is assigned to only one electron)
  hitUsed_.clear();

  unsigned int counter = 0;
  for (SiStripRecHit2DCollection::const_iterator it = rphiHits_p_->begin();  it != rphiHits_p_->end();  ++it) {
    rphiKey_[&(*it)] = counter;
    hitUsed_[&(*it)] = false;
    counter++;
  }

  counter = 0;
  for (SiStripRecHit2DCollection::const_iterator it = stereoHits_p_->begin();  it != stereoHits_p_->end();  ++it) {
    stereoKey_[&(*it)] = counter;
    hitUsed_[&(*it)] = false;
    counter++;
  }

}

// returns true iff an electron was found
// inserts electrons and trackcandiates into electronOut and trackCandidateOut
bool SiStripElectronAlgo::findElectron(reco::SiStripElectronCollection& electronOut,
				       TrackCandidateCollection& trackCandidateOut,
				       const reco::SuperClusterRef& superclusterIn)
{
  // Try each of the two charge hypotheses, but only take one
  bool electronSuccess = projectPhiBand(-1., superclusterIn);
  bool positronSuccess = projectPhiBand( 1., superclusterIn);

  // electron hypothesis did better than electron
  if ((electronSuccess  &&  !positronSuccess)  ||
      (electronSuccess  &&  positronSuccess  &&  redchi2_neg_ <= redchi2_pos_)) {
      
    // Initial uncertainty for tracking
    AlgebraicSymMatrix errors(5,1);  // makes identity 5x5 matrix, indexed from (1,1) to (5,5)
    errors(1,1) = 3.;      // uncertainty**2 in 1/momentum
    errors(2,2) = 0.01;    // uncertainty**2 in lambda (lambda == pi/2 - polar angle theta)
    errors(3,3) = 0.0001;  // uncertainty**2 in phi
    errors(4,4) = 0.01;    // uncertainty**2 in x_transverse (where x is in cm)
    errors(5,5) = 0.01;    // uncertainty**2 in y_transverse (where y is in cm)
      
    std::sort(outputHits_neg_.begin(), outputHits_neg_.end(),
	      CompareHits(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(tracker_p_)), alongMomentum)));

    //create an OwnVector needed by the classes which will be stored to the Event
    edm::OwnVector<TrackingRecHit> hits;
    for(std::vector<const TrackingRecHit*>::iterator itHit =outputHits_neg_.begin();
	itHit != outputHits_neg_.end();
	++itHit) {
      hits.push_back( (*itHit)->clone());
      assert(hitUsed_.find(*itHit) != hitUsed_.end());
      hitUsed_[*itHit] = true;
    }

    TrajectoryStateOnSurface state(
				   GlobalTrajectoryParameters(position_neg_, momentum_neg_, -1, magneticField_p_),
				   CurvilinearTrajectoryError(errors),
				   tracker_p_->idToDet(innerhit_neg_->geographicalId())->surface());

    TrajectoryStateTransform transformer;
    PTrajectoryStateOnDet* PTraj = transformer.persistentState(state, innerhit_neg_->geographicalId().rawId());
    TrajectorySeed trajectorySeed(*PTraj, hits, alongMomentum);
    trackCandidateOut.push_back(TrackCandidate(hits, trajectorySeed, *PTraj));

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
    AlgebraicSymMatrix errors(5,1);  // makes identity 5x5 matrix, indexed from (1,1) to (5,5)
    errors(1,1) = 3.;      // uncertainty**2 in 1/momentum
    errors(2,2) = 0.01;    // uncertainty**2 in lambda (lambda == pi/2 - polar angle theta)
    errors(3,3) = 0.0001;  // uncertainty**2 in phi
    errors(4,4) = 0.01;    // uncertainty**2 in x_transverse (where x is in cm)
    errors(5,5) = 0.01;    // uncertainty**2 in y_transverse (where y is in cm)

    std::sort(outputHits_pos_.begin(), outputHits_pos_.end(),
	      CompareHits(TrackingRecHitLessFromGlobalPosition(((TrackingGeometry*)(tracker_p_)), alongMomentum)));

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

    TrajectoryStateTransform transformer;
    PTrajectoryStateOnDet* PTraj = transformer.persistentState(state, innerhit_pos_->geographicalId().rawId());
    TrajectorySeed trajectorySeed(*PTraj, hits, alongMomentum);
    trackCandidateOut.push_back(TrackCandidate(hits, trajectorySeed, *PTraj));

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
  const std::vector<DetId> ids = (stereo ? stereoHits_p_->ids() : rphiHits_p_->ids());
  for (std::vector<DetId>::const_iterator id = ids.begin();  id != ids.end();  ++id) {

    // Get the hits on this detector id
    SiStripRecHit2DCollection::range hits = (stereo ? stereoHits_p_->get(*id) : rphiHits_p_->get(*id));

    // Count the number of hits on this detector id
    unsigned int numberOfHits = 0;
    for (SiStripRecHit2DCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
      numberOfHits++;
      if (numberOfHits > maxHitsOnDetId_) { break; }
    }
      
    // Only take the hits if there aren't too many
    // (Would it be better to loop only once, fill a temporary list,
    // and copy that if numberOfHits <= maxHitsOnDetId_?)
    if (numberOfHits <= maxHitsOnDetId_) {
      for (SiStripRecHit2DCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	if ((endcap  &&  (tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TID  ||
			  tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TEC    ))    ||
	    (!endcap  &&  (tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB  ||
			   tracker_p_->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB    ))      ) {

	  hitPointersOut.push_back(&(*hit));

	} // end if this is the right subdetector
      } // end loop over hits
    } // end if this detector id doesn't have too many hits on it
  } // end loop over detector ids
}

// projects a phi band of width phiBandWidth_ from supercluster into tracker (given a chargeHypothesis)
// fills *_pos_ or *_neg_ member data with the results
// returns true iff the electron/positron passes cuts
bool SiStripElectronAlgo::projectPhiBand(float chargeHypothesis, const reco::SuperClusterRef& superclusterIn)
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

  // Create and fill vectors of pointers to hits
  std::vector<const SiStripRecHit2D*> stereoHits;
  std::vector<const SiStripRecHit2D*> rphiBarrelHits;
  std::vector<const SiStripRecHit2D*> zphiEndcapHits;
  //                                 stereo? endcap?
  coarseHitSelection(stereoHits,     true,   false);
  coarseHitSelection(stereoHits,     true,   true);
  coarseHitSelection(rphiBarrelHits, false,  false);
  coarseHitSelection(zphiEndcapHits, false,  true);

  // Determine how to project from the supercluster into the tracker
  double energy = superclusterIn->energy();
  double pT = energy * superclusterIn->rho()/sqrt(superclusterIn->x()*superclusterIn->x() +
						  superclusterIn->y()*superclusterIn->y() +
						  superclusterIn->z()*superclusterIn->z());
  // This comes from Jackson p. 581-2, a little geometry, and a FUDGE FACTOR of two in the denominator
  // Why is that factor of two correct?  (It's not confusion about radius vs. diameter in the definition of curvature.)
  double phiVsRSlope = -3.00e-3 * chargeHypothesis * magneticField_p_->inTesla(GlobalPoint(superclusterIn->x(), superclusterIn->y(), 0.)).z() / pT / 2.;

  // Shorthand for supercluster radius, z
  const double scr = superclusterIn->rho();
  const double scz = superclusterIn->position().z();

  // These are used to fit all hits to a line in phi(r)
  std::vector<bool> uselist;
  std::vector<double> rlist, philist, w2list;
  std::vector<int> typelist;  // stereo = 0, rphi barrel = 1, and zphi disk = 2 (only used in this function)
  std::vector<const SiStripRecHit2D*> hitlist;

  // These are used to fit the stereo hits to a line in z(r), constrained to pass through supercluster
  double zSlopeFitNumer = 0.;
  double zSlopeFitDenom = 0.;

  // Loop over all stereo hits
  unsigned int numberOfStereoHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = stereoHits.begin();  hit != stereoHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());
    if (!hitUsed_[*hit]) {
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
	 
	  // Use this hit to fit phi(r)
	  uselist.push_back(true);
	  rlist.push_back(r);
	  philist.push_back(phi);
	  w2list.push_back(1./(0.05/r)/(0.05/r));  // weight**2 == 1./uncertainty**2
	  typelist.push_back(0);
	  hitlist.push_back(*hit);

	  // Use this hit to fit z(r)
	  zSlopeFitNumer += -(scr - r) * (z - scz);
	  zSlopeFitDenom += (scr - r) * (scr - r);
	    
	} // end cut on phi band
      } // end cut on electron originating *near* the origin
    } // end assign disjoint sets of hits to electrons
  } // end loop over stereo hits

  // Calculate the linear fit for z(r)
  double zVsRSlope;
  if (zSlopeFitDenom > 0.) {
    zVsRSlope = zSlopeFitNumer / zSlopeFitDenom;
  }
  else {
    // zVsRSlope assumes electron is from origin if there were no stereo hits
    zVsRSlope = scz/scr;
  }

  // Loop over barrel rphi hits
  unsigned int numberOfBarrelRphiHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = rphiBarrelHits.begin();  hit != rphiBarrelHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());
    if (!hitUsed_[*hit]) {
      // Calculate the 2.5-D position of this hit
      GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double r = sqrt(position.x()*position.x() + position.y()*position.y());
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster

      // This is the center of the strip
      double z = position.z();
      // The expected z position of this hit, according to the z(r) fit
      double zFit = zVsRSlope * (r - scr) + scz;

      // Cut on the Z of the strip
      // TIB strips are 11 cm long, TOB strips are 19 cm long (can I get these from a function?)
      if ((tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TIB  &&  fabs(z - zFit) < 12.)  ||
	  (tracker_p_->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetEnumerators::TOB  &&  fabs(z - zFit) < 20.)    ) {
	 
	// Cut a narrow band around the supercluster's projection in phi
	if (unwrapPhi((r-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + phiBandWidth_)) {

	  // Use this hit to fit phi(r)
	  uselist.push_back(true);
	  rlist.push_back(r);
	  philist.push_back(phi);
	  w2list.push_back(1./(0.05/r)/(0.05/r));  // weight**2 == 1./uncertainty**2
	  typelist.push_back(1);
	  hitlist.push_back(*hit);

	} // end cut on phi band
      } // end cut on strip z
    } // end assign disjoint sets of hits to electrons
  } // end loop over barrel rphi hits
   
  // Loop over endcap zphi hits
  unsigned int numberOfEndcapZphiHits = 0;
  for (std::vector<const SiStripRecHit2D*>::const_iterator hit = zphiEndcapHits.begin();  hit != zphiEndcapHits.end();  ++hit) {
    assert(hitUsed_.find(*hit) != hitUsed_.end());
    if (!hitUsed_[*hit]) {
      // Calculate the 2.5-D position of this hit
      GlobalPoint position = tracker_p_->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double z = position.z();
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster

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

  // Calculate a linear phi(r) fit and drop hits until the biggest contributor to chi^2 is less than maxNormResid_
  bool done = false;
  double intercept, slope, chi2;
  while (!done) {
    // The linear fit
    double sum_w2   = 0.;
    double sum_w2x  = 0.;
    double sum_w2x2 = 0.;
    double sum_w2y  = 0.;
    double sum_w2y2 = 0.;
    double sum_w2xy = 0.;
    unsigned int uselist_size = uselist.size();
    for (unsigned int i = 0;  i < uselist_size;  i++) {
      if (uselist[i]) {
	double r = rlist[i];
	double phi = philist[i];
	double weight2 = w2list[i];

	sum_w2   += weight2;
	sum_w2x  += weight2 * r;
	sum_w2x2 += weight2 * r*r;
	sum_w2y  += weight2 * phi;
	sum_w2y2 += weight2 * phi*phi;
	sum_w2xy += weight2 * r*phi;
      }
    } // end loop over hits to calculate a linear fit
    double delta = sum_w2 * sum_w2x2 - (sum_w2x)*(sum_w2x);
    intercept = (sum_w2x2 * sum_w2y - sum_w2x * sum_w2xy)/delta;
    slope = (sum_w2 * sum_w2xy - sum_w2x * sum_w2y)/delta;

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
      uselist[biggest_index] = false;
    }
    else {
      done = true;
    }
  } // end loop over trial fits

  // Now we have intercept, slope, and chi2; uselist to tell us which hits are used, and hitlist for the hits

  // Identify the innermost hit
  const SiStripRecHit2D* innerhit = (SiStripRecHit2D*)(0);
  double innerhitRadius = -1.;  // meaningless until innerhit is defined

  // Copy hits into an OwnVector, which we put in the TrackCandidate
  std::vector<const TrackingRecHit*> outputHits;
  // Reference rphi and stereo hits into RefVectors, which we put in the SiStripElectron
  edm::RefVector<SiStripRecHit2DCollection> outputRphiHits;
  edm::RefVector<SiStripRecHit2DCollection> outputStereoHits;

  typedef edm::Ref<SiStripRecHit2DCollection> SiStripRecHit2DRef;


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
	outputStereoHits.push_back(SiStripRecHit2DRef(*stereoHits_hp_, stereoKey_[hit]));
      }
      else if (typelist[i] == 1) {
	numberOfBarrelRphiHits++;

	// Copy this hit for the TrajectorySeed
	outputHits.push_back(hit);
	outputRphiHits.push_back(SiStripRecHit2DRef(*rphiHits_hp_, rphiKey_[hit]));
      }
      else if (typelist[i] == 2) {
	numberOfEndcapZphiHits++;

	// Copy this hit for the TrajectorySeed
	outputHits.push_back(hit);
	outputRphiHits.push_back(SiStripRecHit2DRef(*rphiHits_hp_, rphiKey_[hit]));
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

    return true;
  } // end if this is a good electron candidate

  // Signal for a failed electron candidate
  return false;
}

//
// const member functions
//

//
// static member functions
//
