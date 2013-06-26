/**
 * Monte Carlo studies for anode LCTs.
 *
 * Slava Valuev  May 26, 2004.
 * Porting from ORCA by S. Valuev in September 2006.
 *
 * $Id: CSCAnodeLCTAnalyzer.cc,v 1.14 2012/12/05 21:12:53 khotilov Exp $
 *
 */

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCAnodeLCTProcessor.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>

using namespace std;

//-----------------
// Static variables
//-----------------

bool CSCAnodeLCTAnalyzer::debug = true;
bool CSCAnodeLCTAnalyzer::isMTCCMask = true;
bool CSCAnodeLCTAnalyzer::doME1A = true;

vector<CSCAnodeLayerInfo> CSCAnodeLCTAnalyzer::getSimInfo(
      const CSCALCTDigi& alct, const CSCDetId& alctId,
      const CSCWireDigiCollection* wiredc,
      const edm::PSimHitContainer* allSimHits) {
  // Fills vector of CSCAnodeLayerInfo objects.  There can be up to 6 such
  // objects (one per layer); they contain the list of wire digis used to
  // build a given ALCT, and the list of associated (closest) SimHits.
  // Filling is done in two steps: first, we construct the list of wire
  // digis; next, find associated SimHits.
  vector<CSCAnodeLayerInfo> alctInfo = lctDigis(alct, alctId, wiredc);

  // Sanity checks.
  if (alctInfo.size() > CSCConstants::NUM_LAYERS) {
    throw cms::Exception("CSCAnodeLCTAnalyzer")
      << "+++ Number of CSCAnodeLayerInfo objects, " << alctInfo.size()
      << ", exceeds max expected, " << CSCConstants::NUM_LAYERS << " +++\n";
  }
  // not a good check for high PU
  //if (alctInfo.size() != (unsigned)alct.getQuality()+3) {
  //  edm::LogWarning("L1CSCTPEmulatorWrongValues")
  //    << "+++ Warning: mismatch between ALCT quality, " << alct.getQuality()
  //    << ", and the number of layers with digis, " << alctInfo.size()
  //    << ", in alctInfo! +++\n";
  //}

  // Find the closest SimHit to each Digi.
  vector<CSCAnodeLayerInfo>::iterator pali;
  for (pali = alctInfo.begin(); pali != alctInfo.end(); pali++) {
    digiSimHitAssociator(*pali, allSimHits);
  }

  return alctInfo;
}

vector<CSCAnodeLayerInfo> CSCAnodeLCTAnalyzer::lctDigis(
      const CSCALCTDigi& alct, const CSCDetId& alctId,
      const CSCWireDigiCollection* wiredc) {
  // Function to find a list of WireDigis used to create an LCT.
  // The list of WireDigis is stored in a class called CSCLayerInfo which
  // contains the layerId's of the stored WireDigis as well as the actual digis
  // themselves.
  CSCAnodeLayerInfo tempInfo;
  vector<CSCAnodeLayerInfo> vectInfo;

  // Inquire the alct for its pattern and key wiregroup.
  int alct_pattern = 0;
  if (!alct.getAccelerator()) alct_pattern = alct.getCollisionB() + 1;
  int alct_keywire = alct.getKeyWG();
  int alct_bx      = alct.getBX();

  // Choose pattern envelope.
  int MESelection = (alctId.station() < 3) ? 0 : 1;

  if (debug) {
    LogDebug("lctDigis")
      << "\nlctDigis: ALCT keywire = " << alct_keywire << "; alctId:"
      << " endcap " << alctId.endcap() << ", station " << alctId.station()
      << ", ring " << alctId.ring() << ", chamber " << alctId.chamber();
  }

  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    map<int, CSCWireDigi> digiMap;
    // Clear tempInfo values every iteration before using.
    tempInfo.clear();

    // ALCTs belong to a chamber, so their layerId is 0.  Wire digis belong
    // to layers, so in order to access them we need to add layer number to
    // ALCT's endcap, chamber, etc.
    CSCDetId layerId(alctId.endcap(), alctId.station(), alctId.ring(),
		     alctId.chamber(), i_layer+1);
    // Preselection of Digis: right layer and bx.
    preselectDigis(alct_bx, layerId, wiredc, digiMap);

    // In case of ME1/1, one can also look for digis in ME1/A.
    // Keep "on" by defailt since the resolution should not be different
    // from that in ME1/B.
    if (doME1A) {
      if (alctId.station() == 1 && alctId.ring() == 1) {
	CSCDetId layerId_me1a(alctId.endcap(), alctId.station(), 4,
			      alctId.chamber(), i_layer+1);
	preselectDigis(alct_bx, layerId_me1a, wiredc, digiMap);
      }
    }

    // Loop over all the wires in a pattern.
    int mask;
    for (int i_wire = 0; i_wire < CSCAnodeLCTProcessor::NUM_PATTERN_WIRES;
	 i_wire++) {
      if (CSCAnodeLCTProcessor::pattern_envelope[0][i_wire] == i_layer) {
	if (!isMTCCMask) {
	  mask = CSCAnodeLCTProcessor::pattern_mask_slim[alct_pattern][i_wire];
	}
	else {
	  mask = CSCAnodeLCTProcessor::pattern_mask_open[alct_pattern][i_wire];
	}
	if (mask == 1) {
	  int wire = alct_keywire +
	    CSCAnodeLCTProcessor::pattern_envelope[1+MESelection][i_wire];
	  if (wire >= 0 && wire < CSCConstants::MAX_NUM_WIRES) {
	    // Check if there is a "good" Digi on this wire.
	    if (digiMap.count(wire) > 0) {
	      tempInfo.setId(layerId); // store the layer of this object
	      tempInfo.addComponent(digiMap[wire]); // and the RecDigi
	      if (debug) LogDebug("lctDigis")
		<< " Digi on ALCT: wire group " << digiMap[wire].getWireGroup();
	    }
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

void CSCAnodeLCTAnalyzer::preselectDigis(const int alct_bx,
      const CSCDetId& layerId, const CSCWireDigiCollection* wiredc,
      map<int, CSCWireDigi>& digiMap) {
  // Preselection of Digis: right layer and bx.

  // Parameters defining time window for accepting hits; should come from
  // configuration file eventually.
  const int fifo_tbins  = 16;
  const int drift_delay =  2;
  const int hit_persist =  6; // not a config. parameter, just const

  const CSCWireDigiCollection::Range rwired = wiredc->get(layerId);
  for (CSCWireDigiCollection::const_iterator digiIt = rwired.first;
       digiIt != rwired.second; ++digiIt) {
    if (debug) LogDebug("lctDigis")
      << "Wire digi: layer " << layerId.layer()-1 << (*digiIt);
    int bx_time = (*digiIt).getTimeBin();
    if (bx_time >= 0 && bx_time < fifo_tbins) {

      // Do not use digis which could not have contributed to a given ALCT.
      int latch_bx = alct_bx + drift_delay;
      if (bx_time <= latch_bx-hit_persist || bx_time > latch_bx) {
	if (debug) LogDebug("lctDigis")
	  << "Late wire digi: layer " << layerId.layer()-1
	  << " " << (*digiIt) << " skipping...";
	continue;
      }

      int i_wire = (*digiIt).getWireGroup() - 1;

      // If there is more than one digi on the same wire, pick the one
      // which occurred earlier.
      if (digiMap.count(i_wire) > 0) {
	if (digiMap[i_wire].getTimeBin() > bx_time) {
	  if (debug) {
	    LogDebug("lctDigis")
	      << " Replacing good wire digi on wire " << i_wire;
	  }
	  digiMap.erase(i_wire);
	}
      }

      digiMap[i_wire] = *digiIt;
      if (debug) {
	LogDebug("lctDigis") << " Good wire digi: wire group " << i_wire;
      }
    }
  }
}

void CSCAnodeLCTAnalyzer::digiSimHitAssociator(CSCAnodeLayerInfo& info,
				     const edm::PSimHitContainer* allSimHits) {
  // This routine matches up the closest simHit to every digi on a given layer.
  // Iit is possible to have up to 3 digis contribute to an LCT on a given
  // layer.  In a primitive algorithm used now more than one digi on a layer
  // can be associated with the same simHit.
  vector<PSimHit> simHits;

  vector<CSCWireDigi> thisLayerDigis = info.getRecDigis();
  if (!thisLayerDigis.empty()) {
    CSCDetId layerId = info.getId();
    bool me11 = (layerId.station() == 1) && (layerId.ring() == 1);

    // Get simHits in this layer.
    for (edm::PSimHitContainer::const_iterator simHitIt = allSimHits->begin();
	 simHitIt != allSimHits->end(); simHitIt++) {

      // Find detId where simHit is located.
      CSCDetId hitId = (CSCDetId)(*simHitIt).detUnitId();
      if (hitId == layerId)
	simHits.push_back(*simHitIt);
      if (me11) {
	CSCDetId layerId_me1a(layerId.endcap(), layerId.station(), 4,
			      layerId.chamber(), layerId.layer());
	if (hitId == layerId_me1a)
	  simHits.push_back(*simHitIt);
      }
    }

    if (!simHits.empty()) {
      ostringstream strstrm;
      if (debug) {
	strstrm << "\nLayer " << layerId.layer()
		<< " has " << simHits.size() << " SimHit(s); eta value(s) = ";
      }

      // Get the wire number for every digi and convert to eta.
      for (vector<CSCWireDigi>::const_iterator prd = thisLayerDigis.begin();
	   prd != thisLayerDigis.end(); prd++) {
	double deltaEtaMin = 999.;
	double bestHitEta  = 999.;
	PSimHit* bestHit   = 0;

	int wiregroup = prd->getWireGroup(); // counted from 1
	double digiEta = getWGEta(layerId, wiregroup-1);

	const CSCLayer* csclayer = geom_->layer(layerId);
	for (vector <PSimHit>::iterator psh = simHits.begin();
	     psh != simHits.end(); psh++) {
	  // Get the local eta for the simHit.
	  LocalPoint hitLP = psh->localPosition();
	  GlobalPoint hitGP = csclayer->toGlobal(hitLP);
	  double hitEta = hitGP.eta();
	  if (debug)
	    strstrm << hitEta << " ";
	  // Find the lowest deltaEta and store the associated simHit.
	  double deltaEta = fabs(hitEta - digiEta);
	  if (deltaEta < deltaEtaMin) {
	    deltaEtaMin = deltaEta;
	    bestHit     = &(*psh);
	    bestHitEta  = hitEta;
	  }
	}
	if (debug) {
	  strstrm << "\nDigi eta: " << digiEta
		  << ", closest SimHit eta: " << bestHitEta
		  << ", particle type: " << bestHit->particleType();
	  //strstrm << "\nlocal position:" << bestHit->localPosition();
	}
	info.addComponent(*bestHit);
      }
      if (debug) {
	LogDebug("digiSimHitAssociator") << strstrm.str();
      }
    }
  }
}

int CSCAnodeLCTAnalyzer::nearestWG(
                            const vector<CSCAnodeLayerInfo>& allLayerInfo,
			    double& closestPhi, double& closestEta) {
  // Function to set the simulated values for comparison to the reconstructed.
  // It first tries to look for the SimHit in the key layer.  If it is
  // unsuccessful, it loops over all layers and looks for an associated
  // hit in any one of the layers.  First instance of a hit gives a calculation
  // for eta.
  int nearestWG = -999;
  PSimHit matchedHit;
  bool hit_found = false;
  CSCDetId layerId;

  vector<CSCAnodeLayerInfo>::const_iterator pli;
  for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) {
    // For ALCT search, the key layer is the 3rd one, counting from 1.
    if (pli->getId().layer() == CSCConstants::KEY_ALCT_LAYER) {
      vector<PSimHit> thisLayerHits = pli->getSimHits();
      if (thisLayerHits.size() > 0) {
	// There can be only one RecDigi (and therefore only one SimHit)
	// in a key layer.
	if (thisLayerHits.size() != 1) {
	  edm::LogWarning("L1CSCTPEmulatorWrongValues")
	    << "+++ Warning: " << thisLayerHits.size()
	    << " SimHits in key layer " << CSCConstants::KEY_ALCT_LAYER
	    << "! +++ \n";
	  for (unsigned i = 0; i < thisLayerHits.size(); i++) {
	    edm::LogWarning("L1CSCTPEmulatorWrongValues")
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

  if (!hit_found) {
    for (pli = allLayerInfo.begin(); pli != allLayerInfo.end(); pli++) {
      // if there is any occurrence of simHit size greater that zero, use this.
      if ((pli->getRecDigis()).size() > 0 && (pli->getSimHits()).size() > 0) {
	// Always use the first SimHit for now.
	vector<PSimHit> thisLayerHits = pli->getSimHits();
	matchedHit = thisLayerHits[0];
	layerId = pli->getId();
	hit_found = true;
	break;
      }
    }
  }

  // Set the eta if there were any hits found.
  if (hit_found) {
    const CSCLayer* csclayer = geom_->layer(layerId);
    const CSCLayerGeometry* layerGeom = csclayer->geometry();
    int nearestW = layerGeom->nearestWire(matchedHit.localPosition());
    nearestWG = layerGeom->wireGroup(nearestW);
    // Wire groups in ALCTs are counted starting from 0, whereas they
    // are counted from 1 in MC-related info.
    nearestWG -= 1;
    if (nearestWG < 0 || nearestWG >= CSCConstants::MAX_NUM_WIRES) {
      edm::LogWarning("L1CSCTPEmulatorWrongInput")
	<< "+++ Warning: nearest wire group, " << nearestWG
	<< ", is not in [0-" << CSCConstants::MAX_NUM_WIRES
	<< ") interval +++\n";
    }

    GlobalPoint thisPoint = csclayer->toGlobal(matchedHit.localPosition());
    closestPhi = thisPoint.phi();
    closestEta = thisPoint.eta();
    ostringstream strstrm;
    if (debug)
      strstrm << "Matched anode phi: " << closestPhi;
    if (closestPhi < 0.) {
      closestPhi += 2.*M_PI;
      if (debug)
	strstrm << " (" << closestPhi << ")";
    }
    if (debug) {
      strstrm << " eta: " << closestEta
	      << " on a layer " << layerId.layer() << " (1-6);"
	      << " nearest wire group: " << nearestWG;
      LogDebug("nearestWG") << strstrm.str();
    }
  }

  return nearestWG;
}

void CSCAnodeLCTAnalyzer::setGeometry(const CSCGeometry* geom) {
  geom_ = geom;
}

double CSCAnodeLCTAnalyzer::getWGEta(const CSCDetId& layerId,
				     const int wiregroup) {
  // Returns eta position of a given wiregroup.
  if (wiregroup < 0 || wiregroup >= CSCConstants::MAX_NUM_WIRES) {
    edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ Warning: wire group, " << wiregroup
      << ", is not in [0-" << CSCConstants::MAX_NUM_WIRES
      << ") interval +++\n";
  }

  const CSCLayer* csclayer = geom_->layer(layerId);
  const CSCLayerGeometry* layerGeom = csclayer->geometry();
  LocalPoint  digiLP = layerGeom->localCenterOfWireGroup(wiregroup+1);
  //int wirePerWG = layerGeom->numberOfWiresPerGroup(wiregroup+1);
  //float middleW = layerGeom->middleWireOfGroup(wiregroup+1);
  //float ywire = layerGeom->yOfWire(middleW, 0.);
  //digiLP = LocalPoint(0., ywire);
  GlobalPoint digiGP = csclayer->toGlobal(digiLP);
  double eta = digiGP.eta();

  return eta;
}
