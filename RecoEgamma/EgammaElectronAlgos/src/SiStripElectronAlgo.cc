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
// $Id: SiStripElectronAlgo.cc,v 1.1 2006/05/27 04:31:16 pivarski Exp $
//

// system include files

// user include files
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"


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
					 double wedgePhiWidth,
					 double originUncertainty,
					 double deltaPhi,
					 unsigned int numHitsMin,
					 unsigned int numHitsMax)
   : maxHitsOnDetId_(maxHitsOnDetId)
   , wedgePhiWidth_(wedgePhiWidth)
   , originUncertainty_(originUncertainty)
   , deltaPhi_(deltaPhi)
   , numHitsMin_(numHitsMin)
   , numHitsMax_(numHitsMax)
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

void SiStripElectronAlgo::getPoints(std::vector<GlobalPoint>& rphiBarrelPoints,
				    std::vector<GlobalPoint>& stereoBarrelPoints,
				    std::vector<GlobalPoint>& endcapPoints)
{
   const std::vector<DetId> rphiDetIds = rphiHits_p->ids();
   const std::vector<DetId> stereoDetIds = stereoHits_p->ids();

   // We want to fill rphiBarrelPoints, stereoBarrelPoints, and
   // endcapPoints with the positions of hits, IF there are no more
   // than maxHitsOnDetId_ on the DetId in question.  Therefore, we
   // fill a temp list for each DetId, and only add the temp list to
   // the output lists if temp.size() <= maxHitsOnDetId_.

   for (std::vector<DetId>::const_iterator id = rphiDetIds.begin();  id != rphiDetIds.end();  ++id) {
      std::vector<GlobalPoint> temp;
      bool isBarrel = true; // only well-defined if temp.size() > 0, but it's also only used if temp.size() > 0 (it's temporary anyway: see below)

      SiStripRecHit2DLocalPosCollection::range detHits = rphiHits_p->get(*id);
      for (SiStripRecHit2DLocalPosCollection::const_iterator detHitIter = detHits.first;  detHitIter != detHits.second;  ++detHitIter) {

	 if (temp.size() > maxHitsOnDetId_) { break; } // save time

	 GlobalPoint point = tracker_p->idToDet(detHitIter->geographicalId())->surface().toGlobal(detHitIter->localPosition());
	 temp.push_back(point);

	 // Replace this with something that actually queries the hit; don't rely on geometric positions!!!
	 isBarrel = ((point.perp2() < 55.*55.  &&  fabs(point.z()) < 67.)  ||  (point.perp2() >= 55.*55.  &&  fabs(point.z()) < 115.));
	 // For a given DetId, all the hits are barrel or all the hits are endcap.  They can't be mixed.

      }

      if (temp.size() <= maxHitsOnDetId_) {
	 if (isBarrel) {
	    for (std::vector<GlobalPoint>::const_iterator tempIter = temp.begin();  tempIter != temp.end();  ++tempIter) {
	       rphiBarrelPoints.push_back(*tempIter);
	    }
	 }
	 else { // endcap
	    for (std::vector<GlobalPoint>::const_iterator tempIter = temp.begin();  tempIter != temp.end();  ++tempIter) {
	       endcapPoints.push_back(*tempIter);
	    }
	 }
      }
   } // end loop over DetIds

   for (std::vector<DetId>::const_iterator id = stereoDetIds.begin();  id != stereoDetIds.end();  ++id) {
      std::vector<GlobalPoint> temp;
      bool isBarrel = true; // only well-defined if temp.size() > 0, but it's also only used if temp.size() > 0 (it's temporary anyway: see below)

      SiStripRecHit2DLocalPosCollection::range detHits = stereoHits_p->get(*id);
      for (SiStripRecHit2DLocalPosCollection::const_iterator detHitIter = detHits.first;  detHitIter != detHits.second;  ++detHitIter) {

	 if (temp.size() > maxHitsOnDetId_) { break; } // save time

	 GlobalPoint point = tracker_p->idToDet(detHitIter->geographicalId())->surface().toGlobal(detHitIter->localPosition());
	 temp.push_back(point);

	 // Replace this with something that actually queries the hit; don't rely on geometric positions!!!
	 isBarrel = ((point.perp2() < 55.*55.  &&  fabs(point.z()) < 67.)  ||  (point.perp2() >= 55.*55.  &&  fabs(point.z()) < 115.));
	 // For a given DetId, all the hits are barrel or all the hits are endcap.  They can't be mixed.

      }

      if (temp.size() <= maxHitsOnDetId_) {
	 if (isBarrel) {
	    for (std::vector<GlobalPoint>::const_iterator tempIter = temp.begin();  tempIter != temp.end();  ++tempIter) {
	       stereoBarrelPoints.push_back(*tempIter);
	    }
	 }
	 else { // endcap
	    for (std::vector<GlobalPoint>::const_iterator tempIter = temp.begin();  tempIter != temp.end();  ++tempIter) {
	       endcapPoints.push_back(*tempIter);
	    }
	 }
      }
   } // end loop over DetIds
}

bool SiStripElectronAlgo::bandHitCounting(reco::SiStripElectronCandidateCollection& electronOut, const reco::SuperCluster& supercluster)
{
   std::vector<GlobalPoint> rphiBarrelPoints;
   std::vector<GlobalPoint> stereoBarrelPoints;
   std::vector<GlobalPoint> endcapPoints;
   getPoints(rphiBarrelPoints, stereoBarrelPoints, endcapPoints);

   // Primitive energy correction; correct energy only matters at low
   // energy, where the helix curvature is significant
   double energy = supercluster.energy() / 0.9;
   double pT = energy * supercluster.rho()/sqrt(supercluster.x()*supercluster.x() +
						supercluster.y()*supercluster.y() +
						supercluster.z()*supercluster.z());
   // This comes from Jackson p. 581-2, a little geometry, and a FUDGE FACTOR of 2. in the denominator
   // Why is that factor of two correct?  (It's not confusion about radius vs. diameter in the definition of curvature.)
   double phiVsRSlope = 3.00e-3 * magneticField_p->inTesla(GlobalPoint(supercluster.x(), supercluster.y(), 0.)).z() / pT / 2.;

   // This will be the output electron if we decide we like this electron
   reco::SiStripElectronCandidate candidate;
   std::vector<GlobalPoint> rphiBarrelWedge;
   std::vector<GlobalPoint> rphiBarrelBand;
   std::vector<GlobalPoint> stereoBarrelWedge;
   std::vector<GlobalPoint> stereoBarrelBand;
   std::vector<GlobalPoint> endcapWedge;
   std::vector<GlobalPoint> endcapBand;

   for (std::vector<GlobalPoint>::const_iterator rphiIter = rphiBarrelPoints.begin();  rphiIter != rphiBarrelPoints.end();  ++rphiIter) {

      // radius of the hit (cylindrical)
      double r = sqrt(rphiIter->x()*rphiIter->x() + rphiIter->y()*rphiIter->y());

      // interpolation between supercluster and origin
      double zHitExpected = supercluster.position().z() / supercluster.rho() * r;

      // phi of the hit with the supercluster's phi as zero
      double phi = unwrapPhi(rphiIter->phi() - supercluster.position().phi());
      // z of the hit
      double z = rphiIter->z();

      // supercluster radius
      double scr = supercluster.rho();

      if (fabs(phi) < wedgePhiWidth_) {

	 // in the TIB (within 55 cm), wafers are 11 cm long
	 // in the TOB (beyond 55 cm), wafers are 19 cm long
	 // Replace this with something like RecoTracking's RoadMap
	 if ((r < 55.  &&  fabs(z - zHitExpected) < 12.)  ||
	     (r >= 55.  &&  fabs(z - zHitExpected) < 20.)) {

	    rphiBarrelWedge.push_back(*rphiIter);

	    // cut a narrow (deltaPhi_) band around the line in a phi vs. r plot
	    if (unwrapPhi((r-scr)*phiVsRSlope - deltaPhi_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + deltaPhi_)) {

	       rphiBarrelBand.push_back(*rphiIter);

	    }
	 
	 } // end z wedge cut
      } // end phi wedge cut
   } // end loop over rphi barrel hits

   // Fit the stereo hits to a line in z(r), constrained to pass through supercluster
   double zSlopeFitNumer = 0.;
   double zSlopeFitDenom = 0.;

   for (std::vector<GlobalPoint>::const_iterator stereoIter = stereoBarrelPoints.begin();  stereoIter != stereoBarrelPoints.end();  ++stereoIter) {

      // radius of the hit (cylindrical)
      double r = sqrt(stereoIter->x()*stereoIter->x() + stereoIter->y()*stereoIter->y());

      // phi of the hit with the supercluster's phi as zero
      double phi = unwrapPhi(stereoIter->phi() - supercluster.position().phi());
      // z of the hit
      double z = stereoIter->z();

      // supercluster radius, z
      double scr = supercluster.rho();
      double scz = supercluster.position().z();

      if (fabs(phi) < wedgePhiWidth_) {

	 if ((-originUncertainty_ + (scz + originUncertainty_)*(r/scr)) < z  &&
	     z < (originUncertainty_ + (scz - originUncertainty_)*(r/scr))) {

	    stereoBarrelWedge.push_back(*stereoIter);

	    // cut a narrow (deltaPhi_) band around the line in a phi vs. r plot
	    if (unwrapPhi((r-scr)*phiVsRSlope - deltaPhi_) < phi  &&  phi < unwrapPhi((r-scr)*phiVsRSlope + deltaPhi_)) {

	       stereoBarrelBand.push_back(*stereoIter);

	       zSlopeFitNumer += -(scr - r) * (z - scz);
	       zSlopeFitDenom += (scr - r) * (scr - r);
	    }
	 
	 } // end z wedge cut
      } // end phi wedge cut
   } // end loop over stereo barrel hits

   if (zSlopeFitDenom > 0.) {
      double zVsRSlope = zSlopeFitNumer / zSlopeFitDenom;

      for (std::vector<GlobalPoint>::const_iterator endcapIter = endcapPoints.begin();  endcapIter != endcapPoints.end();  ++endcapIter) {      

	 // supercluster radius, z
	 // double scr = supercluster.rho();
	 double scz = supercluster.position().z();

	 // phi of the hit with the supercluster's phi as zero
	 double phi = unwrapPhi(endcapIter->phi() - supercluster.position().phi());
	 // z of the hit
	 double z = endcapIter->z();

	 // radius of the hit, determined from z and fit to stereo hits
	 // double rFromFit = (z - scz)/zVsRSlope + scr;

	 if (fabs(phi) < wedgePhiWidth_) {
	    
	    if (fabs(z - scz/2.) < fabs(scz/2.)) {  // be sure the endcap hit is on the same
						    // side of the detector as the supercluster
	       
	       endcapWedge.push_back(*endcapIter);

	       if (unwrapPhi((z-scz)*phiVsRSlope/zVsRSlope - deltaPhi_) < phi  &&  phi < unwrapPhi((z-scz)*phiVsRSlope/zVsRSlope + deltaPhi_)) {

		  endcapBand.push_back(*endcapIter);

	       }

	    } // end z wedge cut
	 } // end phi wedge cut
      } // end loop over endcap hits
   }
   else {  // no stereo hits means no hope of finding a track in the endcap
   }

   unsigned int numhits = rphiBarrelBand.size() + stereoBarrelBand.size() + endcapBand.size();

   if (numhits >= numHitsMin_  &&  numhits <= numHitsMax_) {
      electronOut.push_back(candidate);
      return true;
   }
   else {
      return false;
   }
}

//
// const member functions
//

//
// static member functions
//
