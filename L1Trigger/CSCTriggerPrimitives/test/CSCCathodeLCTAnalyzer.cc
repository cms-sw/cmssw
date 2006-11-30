/**
 * Monte Carlo studies for cathode LCTs.
 *
 * Slava Valuev  May 26, 2004
 * Porting from ORCA by S. Valuev in September 2006.
 *
 * $Date: 2006/11/10 15:53:28 $
 * $Revision: 1.3 $
 *
 */

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/src/RadialStripTopology.h>

#include <DataFormats/L1CSCTrackFinder/interface/CSCConstants.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>

using namespace std;

//-----------------
// Static variables
//-----------------

bool CSCCathodeLCTAnalyzer::debug = true;

vector<CSCCathodeLayerInfo> CSCCathodeLCTAnalyzer::getSimInfo(
      const CSCCLCTDigi& clct, const CSCDetId& clctId,
      const CSCComparatorDigiCollection* compdc,
      const edm::PSimHitContainer* allSimHits) {
  // Fills vector of CSCCathodeLayerInfo objects.  There can be up to 6 such
  // objects (one per layer); they contain the list of comparator digis used to
  // build a given CLCT, and the list of associated (closest) SimHits.
  // Filling is done in two steps: first, we construct the list of comparator
  // digis; next, find associated SimHits.
  vector<CSCCathodeLayerInfo> clctInfo = lctDigis(clct, clctId, compdc);

  // Sanity checks.
  if (clctInfo.size() > CSCConstants::NUM_LAYERS) {
    throw cms::Exception("CSCCathodeLCTAnalyzer")
      << "+++ Number of CSCCathodeLayerInfo objects, " << clctInfo.size()
      << ", exceeds max expected, " << CSCConstants::NUM_LAYERS << " +++\n";
  }
  if (clctInfo.size() != (unsigned)clct.getQuality()) {
    edm::LogWarning("CSCCathodeLCTAnalyzer")
      << "+++ Warning: mismatch between CLCT quality, " << clct.getQuality()
      << ", and the number of layers with digis, " << clctInfo.size()
      << ", in clctInfo! +++\n";
  }

  // Find the closest SimHit to each Digi.
  vector<CSCCathodeLayerInfo>::iterator pcli;
  for (pcli = clctInfo.begin(); pcli != clctInfo.end(); pcli++) {
    digiSimHitAssociator(*pcli, allSimHits);
  }

  return clctInfo;
}

vector<CSCCathodeLayerInfo> CSCCathodeLCTAnalyzer::lctDigis(
      const CSCCLCTDigi& clct, const CSCDetId& clctId,
      const CSCComparatorDigiCollection* compdc) {
  // Function to find a list of ComparatorDigis used to create an LCT.
  // The list of ComparatorDigis is stored in a class called CSCLayerInfo which
  // contains the layerId's of the stored ComparatorDigis as well as the actual
  // digis themselves.
  int hfstripDigis[CSCConstants::NUM_HALF_STRIPS];
  int distripDigis[CSCConstants::NUM_HALF_STRIPS];
  int time[CSCConstants::MAX_NUM_STRIPS], triad[CSCConstants::MAX_NUM_STRIPS];
  int digiNum[CSCConstants::MAX_NUM_STRIPS];
  int digiId = -999;
  CSCCathodeLayerInfo tempInfo;
  vector<CSCCathodeLayerInfo> vectInfo;

  // Parameters defining time window for accepting hits; should come from
  // configuration file eventually.
  const int bx_width    = 6;
  const int drift_delay = 2;

  // Inquire the clct for its key half-strip, strip type and pattern number.
  int clct_keystrip  = clct.getKeyStrip();
  int clct_stripType = clct.getStripType();
  int clct_pattern   = clct.getPattern();
  int clct_bx        = clct.getBX();

  // Re-scale the key di-strip number to the count used in the search.
  if (clct_stripType == 0) clct_keystrip /= 4;

  if (debug) {
    char stype = (clct_stripType == 0) ? 'D' : 'H';
    LogDebug("lctDigis")
      << "\nlctDigis: CLCT keystrip = " << clct_keystrip
      << " (" << stype << ")" << " pattern = " << clct_pattern << "; Id:"
      << " endcap " << clctId.endcap() << ", station " << clctId.station()
      << ", ring " << clctId.ring() << ", chamber " << clctId.chamber();
  }

  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    // Clear tempInfo values every iteration before using.
    tempInfo.clear();

    // @ Switch to maps eventually
    vector<CSCComparatorDigi> digiMap;
    int digi_num = 0;
    for (int i_hstrip = 0; i_hstrip < CSCConstants::NUM_HALF_STRIPS;
	 i_hstrip++) {
      hfstripDigis[i_hstrip] = -999;
      distripDigis[i_hstrip] = -999;
    }
    for (int i_strip = 0; i_strip < CSCConstants::MAX_NUM_STRIPS; i_strip++) {
      time[i_strip]    = -999;
      triad[i_strip]   =    0;
      digiNum[i_strip] = -999;
    }

    // CLCTs belong to a chamber, so their layerId is 0.  Comparator digis
    // belong to layers, so in order to access them we need to add layer
    // number to CLCT's endcap, chamber, etc.
    CSCDetId layerId(clctId.endcap(), clctId.station(), clctId.ring(),
		     clctId.chamber(), i_layer+1);

    // 'Staggering' for this layer.
    const CSCLayer* csclayer = geom_->layer(layerId);
    int stagger = (csclayer->geometry()->stagger()+1)/2;

    // Preselection of Digis: right layer and bx.
    const CSCComparatorDigiCollection::Range rcompd = compdc->get(layerId);
    for (CSCComparatorDigiCollection::const_iterator digiIt = rcompd.first;
         digiIt != rcompd.second; ++digiIt) {
      if (debug) LogDebug("lctDigis")
	<< "Comparator digi: layer " << i_layer
	<< " strip/comparator/time =" << (*digiIt);

      if ((*digiIt).getComparator() == 0 || (*digiIt).getComparator() == 1) {
	int bx_time = (*digiIt).getTimeBin();
	bx_time -= 9; // temp hack
	if (bx_time >= CSCConstants::MIN_BUNCH &&
	    bx_time <= CSCConstants::MAX_BUNCH) {

	  // Shift all times of interest by TIME_OFFSET, so that they will
	  // be non-negative.
	  bx_time += CSCConstants::TIME_OFFSET;

	  // Do not use digis which could not have contributed to a given CLCT.
	  int latch_bx = clct_bx + drift_delay;
	  if (bx_time <= latch_bx-bx_width || bx_time > latch_bx) {
	    if (debug) LogDebug("lctDigis")
	      << "Late comparator digi: layer " << i_layer
	      << " strip/comparator/time =" << (*digiIt) << " skipping...";
	    continue;
	  }
	    
	  // If there is more than one digi on the same strip, pick the one
	  // which occurred earlier.
	  int i_strip = (*digiIt).getStrip() - 1; // starting from 0
	  if (time[i_strip] <= 0 || time[i_strip] > bx_time) {
 
	    // @ Switch to maps; check for match in time.
	    int i_hfstrip = 2*i_strip + (*digiIt).getComparator() + stagger;
	    hfstripDigis[i_hfstrip] = digi_num;

	    // Arrays for distrip stagger
	    triad[i_strip]   = (*digiIt).getComparator();
	    time[i_strip]    = bx_time;
	    digiNum[i_strip] = digi_num;

	    if (debug) LogDebug("lctDigis")
	      << "digi_num = " << digi_num << " half-strip = " << i_hfstrip
	      << " strip = " << i_strip;
	  }
	}
      }
      digiMap.push_back(*digiIt);
      digi_num++;
    }

    // Loop for di-strips, including stagger
    for (int i_strip = 0; i_strip < CSCConstants::MAX_NUM_STRIPS; i_strip++) {
      if (time[i_strip] >= 0) {
	int i_distrip = i_strip/2;
	if (i_strip%2 && triad[i_strip] == 1 && stagger == 1)
	  CSCCathodeLCTProcessor::distripStagger(triad, time, digiNum,
						 i_strip);
	distripDigis[i_distrip] = digiNum[i_strip];
      }
    }

    // Loop over all the strips in a pattern.
    for (int i_strip = 0; i_strip < CSCCathodeLCTProcessor::NUM_PATTERN_STRIPS;
	 i_strip++) {
      if (CSCCathodeLCTProcessor::pattern[clct_pattern][i_strip] == i_layer) {
	int strip = clct_keystrip +
	  CSCCathodeLCTProcessor::pre_hit_pattern[1][i_strip];
	if (strip >= 0 && strip < CSCConstants::NUM_HALF_STRIPS) {
	  // stripType=0 corresponds to distrips, stripType=1 for halfstrips.
	  if (clct_stripType == 0)
	    digiId = distripDigis[strip];
	  else if (clct_stripType == 1)
	    digiId = hfstripDigis[strip];
	  // distripDigis/halfstripDigis contains the digi numbers
	  // that were carried through the different transformations
	  // to keystrip. -999 means there was no Digi.
	  if (digiId >= 0) {
	    tempInfo.setId(layerId); // store the layer of this object
	    tempInfo.addComponent(digiMap[digiId]); // and the RecDigi
	    if (debug) LogDebug("lctDigis")
	      << " Digi on CLCT: strip/comp/time " << digiMap[digiId];
	  }
	}
      }
    }

    // Save results for each non-empty layer.
    if (tempInfo.getId().layer() != 0) {
      vectInfo.push_back(tempInfo);
    }
  }
  return vectInfo;
}

void CSCCathodeLCTAnalyzer::digiSimHitAssociator(CSCCathodeLayerInfo& info,
				     const edm::PSimHitContainer* allSimHits) {
  // This routine matches up the closest simHit to every digi on a given layer.
  // Note that it is possible to have up to 2 digis contribute to an LCT
  // on a given layer. In a primitive algorithm used now more than one digi on
  // a layer can be associated with the same simHit.
  vector<PSimHit> simHits;

  vector<CSCComparatorDigi> thisLayerDigis = info.getRecDigis();
  if (!thisLayerDigis.empty()) {
    CSCDetId layerId = info.getId();

    // Get simHits in this layer.
    for (edm::PSimHitContainer::const_iterator simHitIt = allSimHits->begin();
	 simHitIt != allSimHits->end(); simHitIt++) {

      // Find detId where simHit is located.
      CSCDetId hitId = (CSCDetId)(*simHitIt).detUnitId();
      if (hitId == layerId)
	simHits.push_back(*simHitIt);
    }

    if (!simHits.empty()) {
      ostringstream strstrm;
      if (debug) {
	strstrm << "\nLayer " << layerId.layer()
		<< " has " << simHits.size() << " SimHit(s); eta value(s) = ";
      }

      // Get the strip number for every digi and convert to phi.
      for (vector<CSCComparatorDigi>::iterator prd = thisLayerDigis.begin();
	   prd != thisLayerDigis.end(); prd++) {
	double deltaPhiMin = 999.;
	double bestHitPhi  = 999.;
	PSimHit* bestHit   = 0;

	int strip = prd->getStrip();
	double digiPhi = getStripPhi(layerId, strip-0.5);

	const CSCLayer* csclayer = geom_->layer(layerId);
	for (vector <PSimHit>::iterator psh = simHits.begin();
	     psh != simHits.end(); psh++) {
	  // Get the local phi for the simHit.
	  LocalPoint hitLP = psh->localPosition();
	  GlobalPoint hitGP = csclayer->toGlobal(hitLP);
	  double hitPhi = hitGP.phi();
	  if (debug)
	    strstrm << hitPhi << " ";
	  // Find the lowest deltaPhi and store the associated simHit.
	  // Check to see if comparison is on pi border, and if so, make
	  // corrections.
	  double deltaPhi = 999.;
	  if (fabs(hitPhi - digiPhi) < M_PI) {
	    deltaPhi = fabs(hitPhi - digiPhi);
	  }
	  else {
	    if (debug) LogDebug("digiSimHitAssociator")
	      << "SimHit and Digi are close to edge!!!";
	    deltaPhi = fabs(fabs(hitPhi - digiPhi) - 2.*M_PI);
	  }
	  if (deltaPhi < deltaPhiMin) {
	    deltaPhiMin = deltaPhi;
	    bestHit     = &(*psh);
	    bestHitPhi  = hitPhi;
	  }
	}
	if (debug) {
	  strstrm <<"\nDigi Phi: " << digiPhi
		  << ", closest SimHit phi: " << bestHitPhi
		  << ", particle type: " << bestHit->particleType();
	}
	info.addComponent(*bestHit);
      }
      if (debug) {
	LogDebug("digiSimHitAssociator") << strstrm.str();
      }
    }
  }
}

int CSCCathodeLCTAnalyzer::nearestHS(
                           const vector<CSCCathodeLayerInfo>& allLayerInfo,
			   double& closestPhi, double& closestEta) {
  // Function to set the simulated values for comparison to the reconstructed.
  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hit in any one of the layers.  First instance of a hit gives a calculation
  // for phi.
  int nearestHS = -999;
  PSimHit matchedHit;
  bool hit_found = false;
  CSCDetId layerId;

  vector<CSCCathodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) {
    if (pli->getId().layer() == CSCConstants::KEY_LAYER) {
      vector<PSimHit> thisLayerHits = pli->getSimHits();
      if (thisLayerHits.size() > 0) {
	// There can be one RecDigi (and therefore only one SimHit)
	// in a keylayer.
	if (thisLayerHits.size() != 1) {
	  edm::LogWarning("nearestWG")
	    << "+++ Warning: " << thisLayerHits.size()
	    << " SimHits in key layer " << CSCConstants::KEY_LAYER
	    << "! +++ \n";
	  for (unsigned i = 0; i < thisLayerHits.size(); i++) {
	    edm::LogWarning("nearestWG")
	      << " SimHit # " << i << thisLayerHits[i] << "\n";
	  }
	}
	matchedHit = thisLayerHits[0];
	layerId = pli->getId();
	hit_found = true;
	break;
      }
    }
  }

  if (hit_found == false) {
    for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) {
      // if there is any occurrence of simHit size greater that zero, use this.
      if ((pli->getRecDigis()).size() > 0 && (pli->getSimHits()).size() > 0) {
	// Always use the first SimHit for now.
	matchedHit = (pli->getSimHits())[0];
	layerId = pli->getId();
	hit_found = true;
	break;
      }
    }
  }

  // Set phi and eta if there were any hits found.
  if (hit_found) {
    const CSCLayer* csclayer = geom_->layer(layerId);
    const CSCLayerGeometry* layerGeom = csclayer->geometry();
    //nearestStrip = layerGeom->nearestStrip(matchedHit.localPosition());
    const RadialStripTopology* topology =
      (const RadialStripTopology*)layerGeom->topology();
    // Float in units of the strip (angular) width.  From RadialStripTopology
    // comments: "Strip in which a given LocalPoint lies. This is a float
    // which represents the fractional strip position within the detector.
    // Returns zero if the LocalPoint falls at the extreme low edge of the
    // detector or BELOW, and float(nstrips) if it falls at the extreme high
    // edge or ABOVE."
    float strip = topology->strip(matchedHit.localPosition());

    // Should be in the interval [0-MAX_STRIPS).  I see (rarely) cases when
    // strip = nearestStrip = MAX_STRIPS; do not know how to handle them.
    int nearestStrip = static_cast<int>(strip);
    if (nearestStrip < 0 || nearestStrip >= CSCConstants::MAX_NUM_STRIPS) {
      edm::LogWarning("nearestStrip")
	<< "+++ Warning: nearest strip, " << nearestStrip
	<< ", is not in [0-" << CSCConstants::MAX_NUM_STRIPS
	<< ") interval; strip = " << strip << " +++\n";
    }
    // Left/right half of the strip.
    int comp = ((strip - nearestStrip) < 0.5) ? 0 : 1;
    nearestHS = 2*nearestStrip + comp;

    GlobalPoint thisPoint = csclayer->toGlobal(matchedHit.localPosition());
    closestPhi = thisPoint.phi();
    closestEta = thisPoint.eta();
    ostringstream strstrm;
    if (debug)
      strstrm << "Matched cathode phi: " << closestPhi;
    if (closestPhi < 0.) {
      closestPhi += 2.*M_PI;
      if (debug)
	strstrm << "(" << closestPhi << ")";
    }
    if (debug) {
      strstrm << " eta: " << closestEta
	      << " on a layer " << layerId.layer() << " (1-6);"
	      << " nearest strip: " << nearestStrip;
      LogDebug("nearestHS") << strstrm.str();
    }
  }

  return nearestHS;
}

void CSCCathodeLCTAnalyzer::setGeometry(const CSCGeometry* geom) {
  geom_ = geom;
}

double CSCCathodeLCTAnalyzer::getStripPhi(const CSCDetId& layerId,
					  const float strip) {
  // Returns phi position of a given strip.
  if (strip < 0. || strip >= CSCConstants::MAX_NUM_STRIPS) {
    edm::LogWarning("getStripPhi")
      << "+++ Warning: strip, " << strip
      << ", is not in [0-" << CSCConstants::MAX_NUM_STRIPS
      << ") interval +++\n";
  }

  const CSCLayer* csclayer = geom_->layer(layerId);
  const CSCLayerGeometry* layerGeom = csclayer->geometry();
  const RadialStripTopology* topology =
    (const RadialStripTopology*)layerGeom->topology();

  // Position at the center of the strip.
  LocalPoint  digiLP = topology->localPosition(strip);
  // The alternative calculation gives exactly the same answer.
  // double ystrip = 0.0;
  // double xstrip = topology->xOfStrip(strip, ystrip);
  // LocalPoint  digiLP = LocalPoint(xstrip, ystrip);
  GlobalPoint digiGP = csclayer->toGlobal(digiLP);
  double phi = digiGP.phi();

  return phi;
}
