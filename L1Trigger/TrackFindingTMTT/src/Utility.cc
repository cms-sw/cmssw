#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/DeadModuleDB.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace TMTT {

//=== Count number of tracker layers a given list of stubs are in.
//=== By default, consider both PS+2S modules, but optionally consider only the PS ones.

unsigned int Utility::countLayers(const Settings* settings, const vector<const Stub*>& vstubs, bool disableReducedLayerID, bool onlyPS) {

  //=== Unpack configuration parameters

  // Note if using reduced layer ID, so tracker layer can be encoded in 3 bits.
  static bool  reduceLayerID           = settings->reduceLayerID();
  // Define layers using layer ID (true) or by bins in radius of 5 cm width (false).
  static bool  useLayerID              = settings->useLayerID();
  // When counting stubs in layers, actually histogram stubs in distance from beam-line with this bin size.
  static float layerIDfromRadiusBin    = settings->layerIDfromRadiusBin();
  // Inner radius of tracker.
  static float trackerInnerRadius      = settings->trackerInnerRadius();

  // Disable use of reduced layer ID if requested, otherwise take from cfg.
  bool reduce  =  (disableReducedLayerID)  ?  false  :  reduceLayerID;

  const int maxLayerID(30);
  vector<bool> foundLayers(maxLayerID, false);

  if (useLayerID) {
    // Count layers using CMSSW layer ID.
    for (const Stub* stub: vstubs) {
      if ( (! onlyPS) || stub->psModule()) { // Consider only stubs in PS modules if that option specified.
	// Use either normal or reduced layer ID depending on request.
	int layerID = reduce  ?  stub->layerIdReduced()  :  stub->layerId();
	if (layerID >= 0 && layerID < maxLayerID) {
	  foundLayers[layerID] = true;
	} else {
	  throw cms::Exception("Utility::invalid layer ID");
	} 
      }
    }

  } else {
    // Count layers by binning stub distance from beam line.
    for (const Stub* stub: vstubs) {
      if ( (! onlyPS) || stub->psModule()) { // Consider only stubs in PS modules if that option specified.
	// N.B. In this case, no concept of "reduced" layer ID has been defined yet, so don't depend on "reduce";
	int layerID = (int) ( (stub->r() - trackerInnerRadius) / layerIDfromRadiusBin );
	if (layerID >= 0 && layerID < maxLayerID) {
	  foundLayers[layerID] = true;
	} else {
	  throw cms::Exception("Utility::invalid layer ID");
	} 
      }
    }
  }

  unsigned int ncount = 0;
  for (const bool& found: foundLayers) {
    if (found) ncount++;
  }
  
  return ncount;
}

//=== Given a set of stubs (presumably on a reconstructed track candidate)
//=== return the best matching Tracking Particle (if any),
//=== the number of tracker layers in which one of the stubs matched one from this tracking particle,
//=== and the list of the subset of the stubs which match those on the tracking particle.

const TP* Utility::matchingTP(const Settings* settings, const vector<const Stub*>& vstubs,
	                      unsigned int& nMatchedLayersBest, vector<const Stub*>& matchedStubsBest)
{
  // Get matching criteria
  const double        minFracMatchStubsOnReco = settings->minFracMatchStubsOnReco();
  const double        minFracMatchStubsOnTP   = settings->minFracMatchStubsOnTP();
  const unsigned int  minNumMatchLayers       = settings->minNumMatchLayers();
  const unsigned int  minNumMatchPSLayers     = settings->minNumMatchPSLayers();

  // Loop over the given stubs, looking at the TP that produced each one.

  map<const TP*, vector<const Stub*> > tpsToStubs;
  map<const TP*, vector<const Stub*> > tpsToStubsStrict;

  for (const Stub* s : vstubs) {
    // If this stub was produced by one or more TPs, store a link from the TPs to the stub.
    // (The assocated TPs here are influenced by config param "StubMatchStrict"). 
    for (const TP* tp_i : s->assocTPs()) {
      tpsToStubs[ tp_i ].push_back( s );
    }
    // To resolve tie-break situations, do the same, but now only considering strictly associated TP, where the TP contributed
    // to both clusters making up stub.
    if (s->assocTP() != nullptr) {
      tpsToStubsStrict[ s->assocTP() ].push_back( s );
    }
  }

  // Loop over all the TP that matched the given stubs, looking for the best matching TP.

  nMatchedLayersBest = 0;        // initialize
  unsigned int nMatchedLayersStrictBest = 0; // initialize
  matchedStubsBest.clear();     // initialize
  const TP* tpBest = nullptr;   // initialize

  for (const auto& iter: tpsToStubs) {
    const TP*                 tp                 = iter.first;
    const vector<const Stub*> matchedStubsFromTP = iter.second;

    const vector<const Stub*> matchedStubsStrictFromTP = tpsToStubsStrict[tp]; // Empty vector, if this TP didnt produce both clusters in stub.

    // Count number of the given stubs that came from this TP.
    unsigned int nMatchedStubs  = matchedStubsFromTP.size();
    // Count number of tracker layers in which the given stubs came from this TP.
    unsigned int nMatchedLayers = Utility::countLayers( settings, matchedStubsFromTP, true );
    unsigned int nMatchedPSLayers = Utility::countLayers( settings, matchedStubsFromTP, true, true );

    // For tie-breaks, count number of tracker layers in which both clusters of the given stubs came from this TP.
    unsigned int nMatchedLayersStrict = Utility::countLayers( settings, matchedStubsStrictFromTP, true );

    // If enough layers matched, then accept this tracking particle.
    // Of the three criteria used here, usually only one is used, with the cuts on the other two set ultra loose.
        
    if (nMatchedStubs >= minFracMatchStubsOnReco * vstubs.size() && // Fraction of matched stubs relative to number of given stubs  
	nMatchedStubs >= minFracMatchStubsOnTP   * tp->numAssocStubs() && // Fraction of matched stubs relative to number of stubs on TP.
	nMatchedLayers >= minNumMatchLayers && nMatchedPSLayers >= minNumMatchPSLayers) { // Number of matched layers
      // In case more than one matching TP found in this cell, note best match based on number of matching layers.
      // In tie-break situation, count layers in which both clusters in stub came from same TP.
      if (nMatchedLayersBest < nMatchedLayers || (nMatchedLayersBest == nMatchedLayers && nMatchedLayersStrictBest < nMatchedLayersStrict)) {
	// Store data for this TP match.
	nMatchedLayersBest = nMatchedLayers;
	matchedStubsBest   = matchedStubsFromTP;
	tpBest             = tp;
      }
    }
  }

  return tpBest;
}

//=== Determine the minimum number of layers a track candidate must have stubs in to be defined as a track.
//=== The first argument indicates from what type of algorithm this function is called: "HT", "SEED", "DUP" or "FIT".

unsigned int Utility::numLayerCut(string algo, const Settings* settings, unsigned int iPhiSec, unsigned int iEtaReg, float invPt, float eta) {
  if (algo == "HT" || algo == "SEED" || algo == "DUP" || algo == "FIT") {

    unsigned int nLayCut = settings->minStubLayers();

    //--- Check if should reduce cut on number of layers by 1 for any reason.

    bool reduce = false;

    // e.g. To increase efficiency for high Pt tracks.
    bool applyMinPt = (settings->minPtToReduceLayers() < 10000.);
    if (applyMinPt && fabs(invPt) < 1/settings->minPtToReduceLayers()) reduce = true;

    // e.g. Or to increase efficiency in the barrel-endcap transition or very forward regions.
    const vector<unsigned int> etaSecsRed = settings->etaSecsReduceLayers();
    if (std::count(etaSecsRed.begin(), etaSecsRed.end(), iEtaReg) != 0) reduce = true;

    // e.g. Or to increase efficiency in sectors containing dead modules.
    if (settings->deadReduceLayers()) {
      const DeadModuleDB dead;
      if (dead.reduceLayerCut(iPhiSec, iEtaReg)) reduce = true; 
    }
     
    if (reduce) nLayCut--;

    // Avoid minimum number of layers going below 4.
    if (nLayCut < 4) nLayCut = 4;

    // Seed Filter & Track Fitters require only 4 layers.
    if (algo == "SEED" || algo == "FIT") nLayCut = 4;

    return nLayCut;
  } else {
    throw cms::Exception("Utility::numLayerCut() called with invalid algo argument!")<<algo<<endl;
  }
}

}
