#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <set>
#include <utility>
#include <cmath>
#include <iostream>

using namespace std;

namespace TMTT {

//--- Stub window sizes copied from L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py

std::vector< double >                DegradeBend::barrelCut_ = 
  {0, 2.0, 2.0, 3.5, 4.5, 5.5, 6.5};
std::vector< std::vector< double > > DegradeBend::ringCut_   = 
  {{0},
   {0, 1, 1.5, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4, 2.5, 3, 3.5, 4.5, 5.5},
   {0, 1, 1.5, 1.5, 2, 2, 2, 2.5, 3, 3, 3, 2, 3, 4, 5, 5.5},
   {0, 1.5, 1.5, 2, 2, 2.5, 2.5, 2.5, 3.5, 2.5, 5, 5.5, 6},
   {0, 1.0, 1.5, 1.5, 2, 2, 2, 2, 3, 3, 6, 6, 6.5},
   {0, 1.0, 1.5, 1.5, 1.5, 2, 2, 2, 3, 3, 6, 6, 6.5}};
std::vector< std::vector< double > > DegradeBend::tiltedCut_ = 
  {{0},
   {0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2., 2., 1.5, 1.5, 1., 1.},
   {0, 3., 3., 3., 3., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2, 2},
   {0, 4.5, 4.5, 4, 4, 4, 4, 3.5, 3.5, 3.5, 3, 3, 3}};
std::vector< double >              DegradeBend::barrelNTilt_ = 
  {0., 12., 12., 12., 0., 0., 0.};

//--- Given the original bend, flag indicating if this is a PS or 2S module, & detector identifier,
//--- this return the degraded stub bend, a boolean indicatng if stub bend was outside the assumed window
//--- size programmed below, and an integer indicating how many values of the original bend
//--- were grouped together into this single value of the degraded bend.

void DegradeBend::degrade(float bend, bool psModule, const DetId& stDetId, float windowFEnew,
			  float& degradedBend, bool& reject, unsigned int& numInGroup) const {

  // Get degraded bend value.
  unsigned int windowHalfStrips;
  this->work(bend, psModule, stDetId, windowFEnew,
	     degradedBend, reject, numInGroup, windowHalfStrips);

  // Check for mistakes.
  this->sanityChecks(psModule, stDetId, windowFEnew, degradedBend, numInGroup, windowHalfStrips);
}

//--- Does the actual work of degrading the bend.

void DegradeBend::work(float bend, bool psModule, const DetId& stDetId, float windowFEnew,
		       float& degradedBend, bool& reject, unsigned int&  numInGroup, unsigned int& windowHalfStrips) const {

  // Calculate stub window size in half-strip units used to produce stubs.
  // This code should be kept identical to that in  
  // L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc

  unsigned int window = 0;

  if (stDetId.subdetId()==StripSubdetector::TOB)
    {
      int layer  = theTrackerTopo_->layer(stDetId);
      int ladder = theTrackerTopo_->tobRod(stDetId);
      int type   = 2*theTrackerTopo_->tobSide(stDetId)-3; // -1 for tilted-, 1 for tilted+, 3 for flat
      double corr=0;

      if (type<3) // Only for tilted modules
	{
	  corr   = (barrelNTilt_.at(layer)+1)/2.;
	  ladder = corr-(corr-ladder)*type; // Corrected ring number, bet 0 and barrelNTilt.at(layer), in ascending |z|
	  window = 2*(tiltedCut_.at(layer)).at(ladder);
	}
      else // Classic barrel window otherwise
	{
	  window = 2*barrelCut_.at( layer );
	}
    }
  else if (stDetId.subdetId()==StripSubdetector::TID)
    {
      window = 2*(ringCut_.at( theTrackerTopo_->tidWheel(stDetId))).at(theTrackerTopo_->tidRing(stDetId));
    }

  // Compare this with the possibly tighter window provided by the user, converting to half-strip units.
  unsigned int newWindow = (unsigned int)(2*windowFEnew);
  if (window > newWindow) window = newWindow;

  // This is the window size measured in half-strips.
  windowHalfStrips = window;

  // Number of degraded bend values should correspond to 3 bits (PS modules) or 4 bits (2S modules),
  // so measuring everything in half-strip units, max integer "window" size that can be encoded without
  // compression given by 2*window+1 <= pow(2,B), where B is number of bits.
  // Hence no compression required if window cut is abs(b) < 3 (PS) or 7 (2S). Must introduce one merge for
  // each 1 unit increase in "window" beyond this.

  // Bend is measured with granularity of 0.5 strips.
  // Convert it to integer measured in half-strip units for this calculation!

  int b = std::round(2*bend);

  if (abs(b) <= window) {
    reject = false;
    float degradedB;
    unsigned int numBends = 2*window + 1;
    unsigned int numAllowed = (psModule)  ?  pow(2, bitsPS_)  :  pow(2, bits2S_);
    // Existance of bend = 0 means can only use an odd number of groups.
    numAllowed -= 1;
    if (numBends <= numAllowed) { 
      // Can output uncompressed bend info.
      numInGroup = 1;
      degradedB = float(b);
    } else {
      unsigned int inSmallGroup = numBends/numAllowed;
      unsigned int numLargeGroups = numBends%numAllowed;
      unsigned int inLargeGroup = inSmallGroup + 1;
      unsigned int numSmallGroups = numAllowed - numLargeGroups;
      vector<unsigned int> groups;
      for (unsigned int i = 0; i < numLargeGroups/2; i++) groups.push_back(inLargeGroup);
      for (unsigned int i = 0; i < numSmallGroups/2; i++) groups.push_back(inSmallGroup);
      // Only one of numLargeGroups & numSmallGroups can be odd, since numAllowed is odd.
      // And whichever one is odd is associated to a group with an odd number of elements since numBends is odd,
      if (numLargeGroups%2 == 1 && inLargeGroup%2 == 1) {
	groups.push_back(inLargeGroup);
      } else if (numSmallGroups%2 == 1 && inSmallGroup%2 == 1) {
	groups.push_back(inSmallGroup);
      } else {
	throw cms::Exception("DegradeBend: logic error with odd numbers");
      }
      for (unsigned int i = 0; i < numSmallGroups/2; i++) groups.push_back(inSmallGroup);
      for (unsigned int i = 0; i < numLargeGroups/2; i++) groups.push_back(inLargeGroup);	  

      degradedB = 999;
      int iUp = -int(window) - 1;
      for (unsigned int& inGroup: groups) {
	iUp += inGroup;
	int iDown = iUp - inGroup + 1;
	if (b <= iUp && b >= iDown) {
	  numInGroup = inGroup;
	  degradedB = 0.5*(iUp + iDown); 
	}
      }
      if (degradedB == 999) throw cms::Exception("DegradeResolution: Logic error in loop over groups");
    }

    // This is degraded bend in full strip units (neglecting bend sign).
    degradedBend = float(degradedB)/2.;
      
  } else {
    // This shouldn't happen. If it does, the the window sizes assumed in this code are tighter than the ones
    // actually used to produce the stubs.
    reject = true;
    numInGroup = 0;
    degradedBend = 99999;
  }
}

//--- Check for mistakes.

void DegradeBend::sanityChecks(bool psModule, const DetId& stDetId, float windowFEnew, float degradedBend, unsigned int numInGroup, unsigned int windowHalfStrips) const {

  pair<unsigned int, bool> p(windowHalfStrips, psModule);

  // Map notes if this (window size, psModule) combination has already been checked.
  static map<pair<unsigned int, bool>, bool> checked;

  if (checked.find(p) == checked.end()) {
    bool wasDegraded = false; // Was any stub bend encoding required for this window size?
    set<float> degradedBendTmpValues;
    set<float> bendTmpMatches;

    // Loop over all bend values allowed within the window size.
    for (int bendHalfStrips = -int(windowHalfStrips); bendHalfStrips <= int(windowHalfStrips); bendHalfStrips++) {
      float bendTmp = float(bendHalfStrips)/2.;
      float degradedBendTmp;
      bool rejectTmp;
      unsigned int numInGroupTmp = 0;
      unsigned int windowHalfStripsTmp = 0;
      this->work(bendTmp, psModule, stDetId, windowFEnew,
		 degradedBendTmp, rejectTmp, numInGroupTmp, windowHalfStripsTmp);
      if (numInGroupTmp > 1) wasDegraded = true;
      degradedBendTmpValues.insert(degradedBendTmp);
      if (degradedBend == degradedBendTmp) bendTmpMatches.insert(bendTmp); // Gives same degraded bend as original problem.

      // Sanity checks.
      if (rejectTmp) throw cms::Exception("DegradeBend: `rejected' flag set, despite bend being within window")<<" fabs("<<bendTmp<<") <= "<<float(windowHalfStrips)/2.<<endl;
      if (4*fabs(bendTmp - degradedBendTmp) > std::round(numInGroupTmp - 1)) throw cms::Exception("DegradeBend: degraded bend differs by more than expected from input bend.")<<" bendTmp="<<bendTmp<<" degradedBendTmp="<<degradedBendTmp<<" numInGroupTmp="<<numInGroupTmp<<endl;
    }

    // Sanity checks.
    unsigned int numRedValues = degradedBendTmpValues.size();
    unsigned int maxAllowed = (psModule)  ?  pow(2, bitsPS_)  :  pow(2, bits2S_); 
    if (wasDegraded) {
      if (numRedValues != maxAllowed - 1) throw cms::Exception("DegradeBend: Bend encoding using wrong number of bits")<<numRedValues<<" > "<<maxAllowed<<endl;
    } else {
      if (numRedValues > maxAllowed) throw cms::Exception("DegradeBend: Bend encoding using too many bits")<<numRedValues<<" > "<<maxAllowed<<endl;
    }
    if (bendTmpMatches.size() != numInGroup) throw cms::Exception("DegradeBend: number of bend values in group inconsistent.")<<bendTmpMatches.size()<<" is not equal to "<<numInGroup<<endl;
  }
}

}
