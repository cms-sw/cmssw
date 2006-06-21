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
// $Id: SiStripElectronAlgo.cc,v 1.3 2006/06/21 17:01:03 pivarski Exp $
//

// system include files

// user include files
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripElectronAlgo::SiStripElectronAlgo(unsigned int maxHitsOnDetId,
					 double originUncertainty,
					 double phiBandWidth,
					 unsigned int minHits,
					 double maxReducedChi2)
   : maxHitsOnDetId_(maxHitsOnDetId)
   , originUncertainty_(originUncertainty)
   , phiBandWidth_(phiBandWidth)
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

void SiStripElectronAlgo::prepareEvent(const TrackerGeometry* tracker,
				       const SiStripRecHit2DLocalPosCollection* rphiHits,
				       const SiStripRecHit2DLocalPosCollection* stereoHits,
				       const MagneticField* magneticField)
{
   tracker_p = tracker;
   rphiHits_p = rphiHits;
   stereoHits_p = stereoHits;
   magneticField_p = magneticField;
}

// returns number of electrons found (0, 1, or 2),
// inserts electrons and clouds into electronOut and cloudOut
int SiStripElectronAlgo::findElectron(reco::SiStripElectronCollection& electronOut,
				      TrackCandidateCollection& trackCandidateOut,
				      const reco::SuperClusterRef& superclusterIn)
{
   // Try each of the two charge hypotheses, possibly yielding two
   // clouds for the same supercluster.
   int numberOfCandidates = 0;

   if (projectPhiBand(electronOut, trackCandidateOut, -1., superclusterIn)) {
      numberOfCandidates++;
      edm::LogInfo("SiStripElectronAlgo") << "looks like an electron." << std::endl;
   }

   if (projectPhiBand(electronOut, trackCandidateOut,  1., superclusterIn)) {
      numberOfCandidates++;
      if (numberOfCandidates == 2) { std::cout << "also "; }
      edm::LogInfo("SiStripElectronAlgo") << "looks like an positron." << std::endl;
   }

   return numberOfCandidates;
}

// inserts pointers to good hits into hitPointersOut
// selects hits on DetIds that have no more than maxHitsOnDetId_
// selects from stereo if stereo == true, rphi otherwise
// selects from TID or TEC if endcap == true, TIB or TOB otherwise
void SiStripElectronAlgo::coarseHitSelection(std::vector<const SiStripRecHit2DLocalPos*>& hitPointersOut,
					     bool stereo, bool endcap)
{
   // This function is not time-efficient.  If you want to improve the
   // speed of the algorithm, you'll probably want to change this
   // function.  There may be a more efficienct way to extract hits,
   // and it would definitely help to put a geographical cut on the
   // DetIds.  (How does one determine the global position of a given
   // DetId?  Is tracker_p->idToDet(id)->surface().toGlobal(LocalPosition(0,0,0))
   // expensive?)

   // Loop over the detector ids
   const std::vector<DetId> ids = (stereo ? stereoHits_p->ids() : rphiHits_p->ids());
   for (std::vector<DetId>::const_iterator id = ids.begin();  id != ids.end();  ++id) {

      // Get the hits on this detector id
      SiStripRecHit2DLocalPosCollection::range hits = (stereo ? stereoHits_p->get(*id) : rphiHits_p->get(*id));

      // Count the number of hits on this detector id
      unsigned int numberOfHits = 0;
      for (SiStripRecHit2DLocalPosCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	 numberOfHits++;
	 if (numberOfHits > maxHitsOnDetId_) { break; }
      }
      
      // Only take the hits if there aren't too many
      // (Would it be better to loop only once, fill a temporary list,
      // and copy that if numberOfHits <= maxHitsOnDetId_?)
      if (numberOfHits <= maxHitsOnDetId_) {
	 for (SiStripRecHit2DLocalPosCollection::const_iterator hit = hits.first;  hit != hits.second;  ++hit) {
	    if ((endcap  &&  (tracker_p->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TID  ||
			      tracker_p->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TEC    ))    ||
		(!endcap  &&  (tracker_p->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TIB  ||
			       tracker_p->idToDetUnit(hit->geographicalId())->type().subDetector() == GeomDetType::TOB    ))      ) {

	       hitPointersOut.push_back(&(*hit));

	    } // end if this is the right subdetector
	 } // end loop over hits
      } // end if this detector id doesn't have too many hits on it
   } // end loop over detector ids
}

// projects a phi band of width phiBandWidth_ from supercluster into tracker (given a chargeHypothesis)
// copies and inserts passing hits into a TrackCandidate, which it puts into trackCandidateOut if passes cuts
// returns true iff the electron/positron passes cuts
bool SiStripElectronAlgo::projectPhiBand(reco::SiStripElectronCollection& electronOut,
					 TrackCandidateCollection& trackCandidateOut,
					 float chargeHypothesis,
					 const reco::SuperClusterRef& superclusterIn)
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
   std::vector<const SiStripRecHit2DLocalPos*> stereoHits;
   std::vector<const SiStripRecHit2DLocalPos*> rphiBarrelHits;
   std::vector<const SiStripRecHit2DLocalPos*> zphiEndcapHits;
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
   double phiVsRSlope = -3.00e-3 * chargeHypothesis * magneticField_p->inTesla(GlobalPoint(superclusterIn->x(), superclusterIn->y(), 0.)).z() / pT / 2.;

   // Shorthand for supercluster radius, z
   const double scr = superclusterIn->rho();
   const double scz = superclusterIn->position().z();

   // Identify the innermost hit
   const SiStripRecHit2DLocalPos* innerhit = (SiStripRecHit2DLocalPos*)(0);
   double innerhitRadius = -1.;  // meaningless until innerhit is defined

   // Collect all hits to pass to RoadSearchHelixMaker
   edm::OwnVector<TrackingRecHit> outputHits;

   // These are used to fit all hits to a line in phi(r)
   double sum_w2   = 0.;
   double sum_w2x  = 0.;
   double sum_w2x2 = 0.;
   double sum_w2y  = 0.;
   double sum_w2y2 = 0.;
   double sum_w2xy = 0.;
   std::vector<double> xlist, ylist, w2list;

   // These are used to fit the stereo hits to a line in z(r), constrained to pass through supercluster
   double zSlopeFitNumer = 0.;
   double zSlopeFitDenom = 0.;

   // Loop over all stereo hits
   unsigned int numberOfStereoHits = 0;
   for (std::vector<const SiStripRecHit2DLocalPos*>::const_iterator hit = stereoHits.begin();  hit != stereoHits.end();  ++hit) {
      
      // Calculate the 3-D position of this hit
      GlobalPoint position = tracker_p->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double r = sqrt(position.x()*position.x() + position.y()*position.y());
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster
      double z = position.z();

      // Cut a triangle in the z-r plane using the supercluster's eta/dip angle information
      // and the fact that the electron originated *near* the origin
      if ((-originUncertainty_ + (scz + originUncertainty_)*(r/scr)) < z  &&  z < (originUncertainty_ + (scz - originUncertainty_)*(r/scr))) {
	 
	 // Cut a narrow band around the supercluster's projection in phi
	 if (unwrapPhi((r-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + phiBandWidth_)) {
	 
	    numberOfStereoHits++;

	    // Use this hit to fit phi(r)
	    double weight2 = 1./(0.05/r)/(0.05/r);  // weight**2 == 1./uncertainty**2
	    sum_w2   += weight2;
	    sum_w2x  += weight2 * r;
	    sum_w2x2 += weight2 * r*r;
	    sum_w2y  += weight2 * phi;
	    sum_w2y2 += weight2 * phi*phi;
	    sum_w2xy += weight2 * r*phi;
	    xlist.push_back(r);
	    ylist.push_back(phi);
	    w2list.push_back(weight2);

	    // Use this hit to fit z(r)
	    zSlopeFitNumer += -(scr - r) * (z - scz);
	    zSlopeFitDenom += (scr - r) * (scr - r);
	    
	    // Keep track of the innermost hit
	    if (innerhit == (SiStripRecHit2DLocalPos*)(0)  ||  r < innerhitRadius) {
	       innerhit = *hit;
	       innerhitRadius = r;
	    }

	    // Copy this hit for the TrajectorySeed
	    outputHits.push_back((*hit)->clone());

	 } // end cut on phi band
      } // end cut on electron originating *near* the origin
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
   for (std::vector<const SiStripRecHit2DLocalPos*>::const_iterator hit = rphiBarrelHits.begin();  hit != rphiBarrelHits.end();  ++hit) {
      
      // Calculate the 2.5-D position of this hit
      GlobalPoint position = tracker_p->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
      double r = sqrt(position.x()*position.x() + position.y()*position.y());
      double phi = unwrapPhi(position.phi() - superclusterIn->position().phi());  // phi is relative to supercluster

      // This is the center of the strip
      double z = position.z();
      // The expected z position of this hit, according to the z(r) fit
      double zFit = zVsRSlope * (r - scr) + scz;

      // Cut on the Z of the strip
      // TIB strips are 11 cm long, TOB strips are 19 cm long (can I get these from a function?)
      if ((tracker_p->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetType::TIB  &&  fabs(z - zFit) < 12.)  ||
	  (tracker_p->idToDetUnit((*hit)->geographicalId())->type().subDetector() == GeomDetType::TOB  &&  fabs(z - zFit) < 20.)    ) {
	 
	 // Cut a narrow band around the supercluster's projection in phi
	 if (unwrapPhi((r-scr)*phiVsRSlope - phiBandWidth_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + phiBandWidth_)) {

	    numberOfBarrelRphiHits++;

	    // Use this hit to fit phi(r)
	    double weight2 = 1./(0.05/r)/(0.05/r);  // weight**2 == 1./uncertainty**2
	    sum_w2   += weight2;
	    sum_w2x  += weight2 * r;
	    sum_w2x2 += weight2 * r*r;
	    sum_w2y  += weight2 * phi;
	    sum_w2y2 += weight2 * phi*phi;
	    sum_w2xy += weight2 * r*phi;
	    xlist.push_back(r);
	    ylist.push_back(phi);
	    w2list.push_back(weight2);

	    // Keep track of the innermost hit
	    if (innerhit == (SiStripRecHit2DLocalPos*)(0)  ||  r < innerhitRadius) {
	       innerhit = *hit;
	       innerhitRadius = r;
	    }

	    // Copy this hit for the TrajectorySeed
	    outputHits.push_back((*hit)->clone());

	 } // end cut on phi band
      } // end cut on strip z
   } // end loop over barrel rphi hits

   // Loop over endcap zphi hits
   unsigned int numberOfEndcapZphiHits = 0;
   for (std::vector<const SiStripRecHit2DLocalPos*>::const_iterator hit = zphiEndcapHits.begin();  hit != zphiEndcapHits.end();  ++hit) {
      
      // Calculate the 2.5-D position of this hit
      GlobalPoint position = tracker_p->idToDet((*hit)->geographicalId())->surface().toGlobal((*hit)->localPosition());
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

	 numberOfEndcapZphiHits++;

	 // Use this hit to fit phi(r)
	 double weight2 = 1./(0.05/rFit)/(0.05/rFit);  // weight**2 == 1./uncertainty**2
	 sum_w2   += weight2;
	 sum_w2x  += weight2 * rFit;
	 sum_w2x2 += weight2 * rFit*rFit;
	 sum_w2y  += weight2 * phi;
	 sum_w2y2 += weight2 * phi*phi;
	 sum_w2xy += weight2 * rFit*phi;
	 xlist.push_back(rFit);
	 ylist.push_back(phi);
	 w2list.push_back(weight2);

	 // Keep track of the innermost hit
	 if (innerhit == (SiStripRecHit2DLocalPos*)(0)  ||  rFit < innerhitRadius) {
	    innerhit = *hit;
	    innerhitRadius = rFit;
	 }

	 // Copy this hit for the TrajectorySeed
	 outputHits.push_back((*hit)->clone());

      } // end cut on phi band
   } // end loop over endcap zphi hits

   // Calculate the linear fit for phi(r)
   double delta = sum_w2 * sum_w2x2 - (sum_w2x)*(sum_w2x);
   double intercept = (sum_w2x2 * sum_w2y - sum_w2x * sum_w2xy)/delta;
   double slope = (sum_w2 * sum_w2xy - sum_w2x * sum_w2y)/delta;

   // Calculate chi^2
   double chi2 = 0.;
   for (unsigned int i = 0;  i < xlist.size();  i++) {
      chi2 += w2list[i] * (ylist[i] - intercept - slope*xlist[i])*(ylist[i] - intercept - slope*xlist[i]);
   }
   // The reduced chi^2 should have a large (rejectable) value if there are no degrees of freedom
   // (with a minHits_ cut above 2, this will never happen...)
   double reducedChi2 = (xlist.size() > 2 ? chi2 / (xlist.size() - 2) : 1e10);

   unsigned int totalNumberOfHits = numberOfStereoHits + numberOfBarrelRphiHits + numberOfEndcapZphiHits;

   edm::LogInfo("SiStripElectronAlgo") << "found " << totalNumberOfHits
				       << " hits in the phi band with a chi^2 of " << chi2 << " ("
				       << (chargeHypothesis > 0. ? "positron" : "electron") << " hypothesis)" << std::endl;
   edm::LogInfo("SiStripElectronAlgo") << "fit phi(r) = " << intercept << " + " << slope << "*r" << std::endl;

   // Select this candidate if it passes minHits_ and maxReducedChi2_ cuts
   if (totalNumberOfHits >= minHits_  &&  reducedChi2 <= maxReducedChi2_) {

      // GlobalTrajectoryParameters evaluated at the position of the innerhit
      GlobalPoint position = tracker_p->idToDet(innerhit->geographicalId())->surface().toGlobal(innerhit->localPosition());
      
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

      // Initial uncertainty for tracking
      AlgebraicSymMatrix errors(5,1);  // makes identity 5x5 matrix, indexed from (1,1) to (5,5)
      errors(1,1) = 3.;      // uncertainty**2 in 1/momentum
      errors(2,2) = 0.01;    // uncertainty**2 in lambda (lambda == pi/2 - polar angle theta)
      errors(3,3) = 0.0001;  // uncertainty**2 in phi
      errors(4,4) = 0.01;    // uncertainty**2 in x_transverse (where x is in cm)
      errors(5,5) = 0.01;    // uncertainty**2 in y_transverse (where y is in cm)

      TrajectoryStateOnSurface state(
	 GlobalTrajectoryParameters(position, momentum, TrackCharge(chargeHypothesis), magneticField_p),
	 CurvilinearTrajectoryError(errors),
	 tracker_p->idToDet(innerhit->geographicalId())->surface());

      TrajectoryStateTransform transformer;
      PTrajectoryStateOnDet* PTraj = transformer.persistentState(state, innerhit->geographicalId().rawId());
      TrajectorySeed trajectorySeed(*PTraj, outputHits, alongMomentum);

//      trackCandidateOut.push_back(TrackCandidate(outputHits));
      trackCandidateOut.push_back(TrackCandidate(outputHits, trajectorySeed, *PTraj));

      electronOut.push_back(reco::SiStripElectron(superclusterIn,
						  (chargeHypothesis > 0. ? 1 : -1),
						  phiVsRSlope,
						  slope,
						  intercept,
						  chi2,
						  (xlist.size() - 2),
						  correct_pT,
						  pZ,
						  zVsRSlope,
						  numberOfStereoHits,
						  numberOfBarrelRphiHits,
						  numberOfEndcapZphiHits));
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
