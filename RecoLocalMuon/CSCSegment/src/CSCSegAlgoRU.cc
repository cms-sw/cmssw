/**
 * \file CSCSegAlgRU.cc
 *
 * \authors V.Palichik & N.Voytishin
 * \some functions and structure taken from SK algo by M.Sani and SegFit class by T.Cox
 */

#include "CSCSegAlgoRU.h"
#include "CSCSegFit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include <string>

CSCSegAlgoRU::CSCSegAlgoRU(const edm::ParameterSet& ps) : CSCSegmentAlgorithm(ps), myName("CSCSegAlgoRU") {
  doCollisions = ps.getParameter<bool>("doCollisions");
  chi2_str_ = ps.getParameter<double>("chi2_str");
  chi2Norm_2D_ = ps.getParameter<double>("chi2Norm_2D_");
  dRMax = ps.getParameter<double>("dRMax");
  dPhiMax = ps.getParameter<double>("dPhiMax");
  dRIntMax = ps.getParameter<double>("dRIntMax");
  dPhiIntMax = ps.getParameter<double>("dPhiIntMax");
  chi2Max = ps.getParameter<double>("chi2Max");
  wideSeg = ps.getParameter<double>("wideSeg");
  minLayersApart = ps.getParameter<int>("minLayersApart");

  LogDebug("CSC") << myName << " has algorithm cuts set to: \n"
                  << "--------------------------------------------------------------------\n"
                  << "dRMax = " << dRMax << '\n'
                  << "dPhiMax = " << dPhiMax << '\n'
                  << "dRIntMax = " << dRIntMax << '\n'
                  << "dPhiIntMax = " << dPhiIntMax << '\n'
                  << "chi2Max = " << chi2Max << '\n'
                  << "wideSeg = " << wideSeg << '\n'
                  << "minLayersApart = " << minLayersApart << std::endl;

  //reset the thresholds for non-collision data
  if (!doCollisions) {
    dRMax = 2.0;
    dPhiMax = 2 * dPhiMax;
    dRIntMax = 2 * dRIntMax;
    dPhiIntMax = 2 * dPhiIntMax;
    chi2Norm_2D_ = 5 * chi2Norm_2D_;
    chi2_str_ = 100;
    chi2Max = 2 * chi2Max;
  }
}

std::vector<CSCSegment> CSCSegAlgoRU::buildSegments(const CSCChamber* aChamber,
                                                    const ChamberHitContainer& urechits) const {
  ChamberHitContainer rechits = urechits;
  LayerIndex layerIndex(rechits.size());
  int recHits_per_layer[6] = {0, 0, 0, 0, 0, 0};
  //skip events with high multiplicity of hits
  if (rechits.size() > 150) {
    return std::vector<CSCSegment>();
  }
  int iadd = 0;
  for (unsigned int i = 0; i < rechits.size(); i++) {
    recHits_per_layer[rechits[i]->cscDetId().layer() - 1]++;  //count rh per chamber
    layerIndex[i] = rechits[i]->cscDetId().layer();
  }
  double z1 = aChamber->layer(1)->position().z();
  double z6 = aChamber->layer(6)->position().z();
  if (std::abs(z1) > std::abs(z6)) {
    reverse(layerIndex.begin(), layerIndex.end());
    reverse(rechits.begin(), rechits.end());
  }
  if (rechits.size() < 2) {
    return std::vector<CSCSegment>();
  }
  // We have at least 2 hits. We intend to try all possible pairs of hits to start
  // segment building. 'All possible' means each hit lies on different layers in the chamber.
  // after all same size segs are build we get rid of the overcrossed segments using the chi2 criteria
  // the hits from the segs that are left are marked as used and are not added to segs in future iterations
  // the hits from 3p segs are marked as used separately in order to try to assamble them in longer segments
  // in case there is a second pass
  // Choose first hit (as close to IP as possible) h1 and second hit
  // (as far from IP as possible) h2 To do this we iterate over hits
  // in the chamber by layer - pick two layers. Then we
  // iterate over hits within each of these layers and pick h1 and h2
  // these. If they are 'close enough' we build an empty
  // segment. Then try adding hits to this segment.
  // Initialize flags that a given hit has been allocated to a segment
  BoolContainer used(rechits.size(), false);
  BoolContainer used3p(rechits.size(), false);
  // This is going to point to fits to hits, and its content will be used to create a CSCSegment
  AlgoState aState;
  aState.aChamber = aChamber;
  aState.doCollisions = doCollisions;
  aState.dRMax = dRMax;
  aState.dPhiMax = dPhiMax;
  aState.dRIntMax = dRIntMax;
  aState.dPhiIntMax = dPhiIntMax;
  aState.chi2Norm_2D_ = chi2Norm_2D_;
  aState.chi2_str_ = chi2_str_;
  aState.chi2Max = chi2Max;

  // Define buffer for segments we build
  std::vector<CSCSegment> segments;
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  // Possibly allow 3 passes, second widening scale factor for cuts, third for segments from displaced vertices
  aState.windowScale = 1.;  // scale factor for cuts
  bool search_disp = false;
  aState.strip_iadd = 1;
  aState.chi2D_iadd = 1;
  int npass = (wideSeg > 1.) ? 3 : 2;
  for (int ipass = 0; ipass < npass; ++ipass) {
    if (aState.windowScale > 1.) {
      iadd = 1;
      aState.strip_iadd = 4;
      aState.chi2D_iadd = 4;
      aState.chi2Max = 2 * chi2Max;
      if (rechits.size() <= 12)
        iadd = 0;  //allow 3 hit segments for low hit multiplicity chambers
    }

    int used_rh = 0;
    for (ChamberHitContainerCIt i1 = ib; i1 != ie; ++i1) {
      if (used[i1 - ib])
        used_rh++;
    }

    //change the tresholds if it's time to look for displaced mu segments
    if (aState.doCollisions && search_disp &&
        int(rechits.size() - used_rh) >
            2) {  //check if there are enough recHits left to build a segment from displaced vertices
      aState.doCollisions = false;
      aState.windowScale = 1.;  // scale factor for cuts
      aState.dRMax = 4.0;
      aState.dPhiMax = 4 * aState.dPhiMax;
      aState.dRIntMax = 4 * aState.dRIntMax;
      aState.dPhiIntMax = 4 * aState.dPhiIntMax;
      aState.chi2Norm_2D_ = 10 * aState.chi2Norm_2D_;
      aState.chi2_str_ = 200;
      aState.chi2Max = 4 * aState.chi2Max;
    } else {
      search_disp = false;  //make sure the flag is off
    }

    for (unsigned int n_seg_min = 6u; n_seg_min > 2u + iadd; --n_seg_min) {
      BoolContainer common_used(rechits.size(), false);
      std::array<BoolContainer, 120> common_used_it = {};
      for (unsigned int i = 0; i < common_used_it.size(); i++) {
        common_used_it[i] = common_used;
      }
      ChamberHitContainer best_proto_segment[120];
      float min_chi[120] = {9999};
      int common_it = 0;
      bool first_proto_segment = true;
      for (ChamberHitContainerCIt i1 = ib; i1 != ie; ++i1) {
        bool segok = false;
        //skip if rh is used and the layer tat has big rh multiplicity(>25RHs)
        if (used[i1 - ib] || recHits_per_layer[int(layerIndex[i1 - ib]) - 1] > 25 ||
            (n_seg_min == 3 && used3p[i1 - ib]))
          continue;
        int layer1 = layerIndex[i1 - ib];
        const CSCRecHit2D* h1 = *i1;
        for (ChamberHitContainerCIt i2 = ie - 1; i2 != i1; --i2) {
          if (used[i2 - ib] || recHits_per_layer[int(layerIndex[i2 - ib]) - 1] > 25 ||
              (n_seg_min == 3 && used3p[i2 - ib]))
            continue;
          int layer2 = layerIndex[i2 - ib];
          if ((abs(layer2 - layer1) + 1) < int(n_seg_min))
            break;  //decrease n_seg_min
          const CSCRecHit2D* h2 = *i2;
          if (areHitsCloseInR(aState, h1, h2) && areHitsCloseInGlobalPhi(aState, h1, h2)) {
            aState.proto_segment.clear();
            if (!addHit(aState, h1, layer1))
              continue;
            if (!addHit(aState, h2, layer2))
              continue;
            // Can only add hits if already have a segment
            if (aState.sfit)
              tryAddingHitsToSegment(aState, rechits, used, layerIndex, i1, i2);
            segok = isSegmentGood(aState, rechits);
            if (segok) {
              if (aState.proto_segment.size() > n_seg_min) {
                baseline(aState, n_seg_min);
                updateParameters(aState);
              }
              if (aState.sfit->chi2() > aState.chi2Norm_2D_ * aState.chi2D_iadd ||
                  aState.proto_segment.size() < n_seg_min)
                aState.proto_segment.clear();
              if (!aState.proto_segment.empty()) {
                updateParameters(aState);
                //add same-size overcrossed protosegments to the collection
                if (first_proto_segment) {
                  flagHitsAsUsed(aState, rechits, common_used_it[0]);
                  min_chi[0] = aState.sfit->chi2();
                  best_proto_segment[0] = aState.proto_segment;
                  first_proto_segment = false;
                } else {  //for the rest of found proto_segments
                  common_it++;
                  flagHitsAsUsed(aState, rechits, common_used_it[common_it]);
                  min_chi[common_it] = aState.sfit->chi2();
                  best_proto_segment[common_it] = aState.proto_segment;
                  ChamberHitContainerCIt hi, iu, ik;
                  int iter = common_it;
                  for (iu = ib; iu != ie; ++iu) {
                    for (hi = aState.proto_segment.begin(); hi != aState.proto_segment.end(); ++hi) {
                      if (*hi == *iu) {
                        int merge_nr = -1;
                        for (int k = 0; k < iter + 1; k++) {
                          if (common_used_it[k][iu - ib] == true) {
                            if (merge_nr != -1) {
                              //merge the k and merge_nr collections of flaged hits into the merge_nr collection and unmark the k collection hits
                              for (ik = ib; ik != ie; ++ik) {
                                if (common_used_it[k][ik - ib] == true) {
                                  common_used_it[merge_nr][ik - ib] = true;
                                  common_used_it[k][ik - ib] = false;
                                }
                              }
                              //change best_protoseg if min_chi_2 is smaller
                              if (min_chi[k] < min_chi[merge_nr]) {
                                min_chi[merge_nr] = min_chi[k];
                                best_proto_segment[merge_nr] = best_proto_segment[k];
                                best_proto_segment[k].clear();
                                min_chi[k] = 9999;
                              }
                              common_it--;
                            } else {
                              merge_nr = k;
                            }
                          }  //if(common_used[k][iu-ib] == true)
                        }    //for k
                      }      //if
                    }        //for proto_seg
                  }          //for rec_hits
                }            //else
              }              //proto seg not empty
            }
          }  // h1 & h2 close
          if (segok)
            break;
        }  // i2
      }    // i1

      //add the reconstructed segments
      for (int j = 0; j < common_it + 1; j++) {
        aState.proto_segment = best_proto_segment[j];
        best_proto_segment[j].clear();
        //SKIP empty proto-segments
        if (aState.proto_segment.empty())
          continue;
        updateParameters(aState);
        // Create an actual CSCSegment - retrieve all info from the fit
        CSCSegment temp(aState.sfit->hits(),
                        aState.sfit->intercept(),
                        aState.sfit->localdir(),
                        aState.sfit->covarianceMatrix(),
                        aState.sfit->chi2());
        aState.sfit = nullptr;
        segments.push_back(temp);
        //if the segment has 3 hits flag them as used in a particular way
        if (aState.proto_segment.size() == 3) {
          flagHitsAsUsed(aState, rechits, used3p);
        } else {
          flagHitsAsUsed(aState, rechits, used);
        }
        aState.proto_segment.clear();
      }
    }  //for n_seg_min

    if (search_disp) {
      //reset params and flags for the next chamber
      search_disp = false;
      aState.doCollisions = true;
      aState.dRMax = 2.0;
      aState.chi2_str_ = 100;
      aState.dPhiMax = 0.25 * aState.dPhiMax;
      aState.dRIntMax = 0.25 * aState.dRIntMax;
      aState.dPhiIntMax = 0.25 * aState.dPhiIntMax;
      aState.chi2Norm_2D_ = 0.1 * aState.chi2Norm_2D_;
      aState.chi2Max = 0.25 * aState.chi2Max;
    }

    std::vector<CSCSegment>::iterator it = segments.begin();
    bool good_segs = false;
    while (it != segments.end()) {
      if ((*it).nRecHits() > 3) {
        good_segs = true;
        break;
      }
      ++it;
    }
    if (good_segs &&
        aState.doCollisions) {  // only change window if not enough good segments were found (bool can be changed to int if a >0 number of good segs is required)
      search_disp = true;
      continue;  //proceed to search the segs from displaced vertices
    }

    // Increase cut windows by factor of wideSeg only for collisions
    if (!aState.doCollisions && !search_disp)
      break;
    aState.windowScale = wideSeg;
  }  // ipass

  //get rid of enchansed 3p segments
  std::vector<CSCSegment>::iterator it = segments.begin();
  while (it != segments.end()) {
    if ((*it).nRecHits() == 3) {
      bool found_common = false;
      const std::vector<CSCRecHit2D>& theseRH = (*it).specificRecHits();
      for (ChamberHitContainerCIt i1 = ib; i1 != ie; ++i1) {
        if (used[i1 - ib] && used3p[i1 - ib]) {
          const CSCRecHit2D* sh1 = *i1;
          CSCDetId id = sh1->cscDetId();
          int sh1layer = id.layer();
          int RH_centerid = sh1->nStrips() / 2;
          int RH_centerStrip = sh1->channels(RH_centerid);
          int RH_wg = sh1->hitWire();
          std::vector<CSCRecHit2D>::const_iterator sh;
          for (sh = theseRH.begin(); sh != theseRH.end(); ++sh) {
            CSCDetId idRH = sh->cscDetId();
            //find segment hit coord
            int shlayer = idRH.layer();
            int SegRH_centerid = sh->nStrips() / 2;
            int SegRH_centerStrip = sh->channels(SegRH_centerid);
            int SegRH_wg = sh->hitWire();
            if (sh1layer == shlayer && SegRH_centerStrip == RH_centerStrip && SegRH_wg == RH_wg) {
              //remove the enchansed 3p segment
              segments.erase(it, (it + 1));
              found_common = true;
              break;
            }
          }  //theserh
        }
        if (found_common)
          break;  //current seg has already been erased
      }           //camber hits
      if (!found_common)
        ++it;
    }  //its a 3p seg
    else {
      ++it;  //go to the next seg
    }
  }  //while
  // Give the segments to the CSCChamber
  return segments;
}  //build segments

void CSCSegAlgoRU::tryAddingHitsToSegment(AlgoState& aState,
                                          const ChamberHitContainer& rechits,
                                          const BoolContainer& used,
                                          const LayerIndex& layerIndex,
                                          const ChamberHitContainerCIt i1,
                                          const ChamberHitContainerCIt i2) const {
  // Iterate over the layers with hits in the chamber
  // Skip the layers containing the segment endpoints
  // Test each hit on the other layers to see if it is near the segment
  // If it is, see whether there is already a hit on the segment from the same layer
  // - if so, and there are more than 2 hits on the segment, copy the segment,
  // replace the old hit with the new hit. If the new segment chi2 is better
  // then replace the original segment with the new one (by swap)
  // - if not, copy the segment, add the hit. If the new chi2/dof is still satisfactory
  // then replace the original segment with the new one (by swap)
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  for (ChamberHitContainerCIt i = ib; i != ie; ++i) {
    int layer = layerIndex[i - ib];
    if (hasHitOnLayer(aState, layer) && aState.proto_segment.size() <= 2)
      continue;
    if (layerIndex[i - ib] == layerIndex[i1 - ib] || layerIndex[i - ib] == layerIndex[i2 - ib] || used[i - ib])
      continue;

    const CSCRecHit2D* h = *i;
    if (isHitNearSegment(aState, h)) {
      // Don't consider alternate hits on layers holding the two starting points
      if (hasHitOnLayer(aState, layer)) {
        if (aState.proto_segment.size() <= 2)
          continue;
        compareProtoSegment(aState, h, layer);
      } else {
        increaseProtoSegment(aState, h, layer, aState.chi2D_iadd);
      }
    }  // h & seg close
  }    // i
}

bool CSCSegAlgoRU::areHitsCloseInR(const AlgoState& aState, const CSCRecHit2D* h1, const CSCRecHit2D* h2) const {
  float maxWG_width[10] = {0, 0, 4.1, 5.69, 2.49, 5.06, 2.49, 5.06, 1.87, 5.06};
  CSCDetId id = h1->cscDetId();
  int iStn = id.iChamberType() - 1;
  //find maxWG_width for ME11 (tilt = 29deg)
  int wg_num = h2->hitWire();
  if (iStn == 0 || iStn == 1) {
    if (wg_num == 1) {
      maxWG_width[0] = 9.25;
      maxWG_width[1] = 9.25;
    }
    if (wg_num > 1 && wg_num < 48) {
      maxWG_width[0] = 3.14;
      maxWG_width[1] = 3.14;
    }
    if (wg_num == 48) {
      maxWG_width[0] = 10.75;
      maxWG_width[1] = 10.75;
    }
  }
  const CSCLayer* l1 = aState.aChamber->layer(h1->cscDetId().layer());
  GlobalPoint gp1 = l1->toGlobal(h1->localPosition());
  const CSCLayer* l2 = aState.aChamber->layer(h2->cscDetId().layer());
  GlobalPoint gp2 = l2->toGlobal(h2->localPosition());
  //find z to understand the direction
  float h1z = gp1.z();
  float h2z = gp2.z();
  //switch off the IP check for non collisions case
  if (!aState.doCollisions) {
    h1z = 1;
    h2z = 1;
  }
  return (gp2.perp() > ((gp1.perp() - aState.dRMax * aState.strip_iadd * maxWG_width[iStn]) * h2z) / h1z &&
          gp2.perp() < ((gp1.perp() + aState.dRMax * aState.strip_iadd * maxWG_width[iStn]) * h2z) / h1z)
             ? true
             : false;
}

bool CSCSegAlgoRU::areHitsCloseInGlobalPhi(const AlgoState& aState,
                                           const CSCRecHit2D* h1,
                                           const CSCRecHit2D* h2) const {
  float strip_width[10] = {0.003878509,
                           0.002958185,
                           0.002327105,
                           0.00152552,
                           0.00465421,
                           0.002327105,
                           0.00465421,
                           0.002327105,
                           0.00465421,
                           0.002327105};  //in rad
  const CSCLayer* l1 = aState.aChamber->layer(h1->cscDetId().layer());
  GlobalPoint gp1 = l1->toGlobal(h1->localPosition());
  const CSCLayer* l2 = aState.aChamber->layer(h2->cscDetId().layer());
  GlobalPoint gp2 = l2->toGlobal(h2->localPosition());
  float err_stpos_h1 = h1->errorWithinStrip();
  float err_stpos_h2 = h2->errorWithinStrip();
  CSCDetId id = h1->cscDetId();
  int iStn = id.iChamberType() - 1;
  float dphi_incr = 0;
  if (err_stpos_h1 > 0.25 * strip_width[iStn] || err_stpos_h2 > 0.25 * strip_width[iStn])
    dphi_incr = 0.5 * strip_width[iStn];
  float dphi12 = deltaPhi(gp1.barePhi(), gp2.barePhi());
  return (fabs(dphi12) < (aState.dPhiMax * aState.strip_iadd + dphi_incr)) ? true : false;  // +v
}

bool CSCSegAlgoRU::isHitNearSegment(const AlgoState& aState, const CSCRecHit2D* h) const {
  // Is hit near segment?
  // Requires deltaphi and deltaR within ranges specified in parameter set.
  // Note that to make intuitive cuts on delta(phi) one must work in
  // phi range (-pi, +pi] not [0, 2pi)
  float strip_width[10] = {0.003878509,
                           0.002958185,
                           0.002327105,
                           0.00152552,
                           0.00465421,
                           0.002327105,
                           0.00465421,
                           0.002327105,
                           0.00465421,
                           0.002327105};  //in rad
  const CSCLayer* l1 = aState.aChamber->layer((*(aState.proto_segment.begin()))->cscDetId().layer());
  GlobalPoint gp1 = l1->toGlobal((*(aState.proto_segment.begin()))->localPosition());
  const CSCLayer* l2 = aState.aChamber->layer((*(aState.proto_segment.begin() + 1))->cscDetId().layer());
  GlobalPoint gp2 = l2->toGlobal((*(aState.proto_segment.begin() + 1))->localPosition());
  float err_stpos_h1 = (*(aState.proto_segment.begin()))->errorWithinStrip();
  float err_stpos_h2 = (*(aState.proto_segment.begin() + 1))->errorWithinStrip();
  const CSCLayer* l = aState.aChamber->layer(h->cscDetId().layer());
  GlobalPoint hp = l->toGlobal(h->localPosition());
  float err_stpos_h = h->errorWithinStrip();
  float hphi = hp.phi();  // in (-pi, +pi]
  if (hphi < 0.)
    hphi += 2. * M_PI;                  // into range [0, 2pi)
  float sphi = phiAtZ(aState, hp.z());  // in [0, 2*pi)
  float phidif = sphi - hphi;
  if (phidif < 0.)
    phidif += 2. * M_PI;  // into range [0, 2pi)
  if (phidif > M_PI)
    phidif -= 2. * M_PI;  // into range (-pi, pi]
  SVector6 r_glob;
  CSCDetId id = h->cscDetId();
  int iStn = id.iChamberType() - 1;
  float dphi_incr = 0;
  float pos_str = 1;
  //increase dPhi cut if the hit is on the edge of the strip
  float stpos = (*h).positionWithinStrip();
  bool centr_str = false;
  if (iStn != 0 && iStn != 1) {
    if (stpos > -0.25 && stpos < 0.25)
      centr_str = true;
  }
  if (err_stpos_h1 < 0.25 * strip_width[iStn] || err_stpos_h2 < 0.25 * strip_width[iStn] ||
      err_stpos_h < 0.25 * strip_width[iStn]) {
    dphi_incr = 0.5 * strip_width[iStn];
  } else {
    if (centr_str)
      pos_str = 1.3;
  }
  r_glob((*(aState.proto_segment.begin()))->cscDetId().layer() - 1) = gp1.perp();
  r_glob((*(aState.proto_segment.begin() + 1))->cscDetId().layer() - 1) = gp2.perp();
  float R = hp.perp();
  int layer = h->cscDetId().layer();
  float r_interpolated = fit_r_phi(aState, r_glob, layer);
  float dr = fabs(r_interpolated - R);
  float maxWG_width[10] = {0, 0, 4.1, 5.69, 2.49, 5.06, 2.49, 5.06, 1.87, 5.06};
  //find maxWG_width for ME11 (tilt = 29deg)
  int wg_num = h->hitWire();
  if (iStn == 0 || iStn == 1) {
    if (wg_num == 1) {
      maxWG_width[0] = 9.25;
      maxWG_width[1] = 9.25;
    }
    if (wg_num > 1 && wg_num < 48) {
      maxWG_width[0] = 3.14;
      maxWG_width[1] = 3.14;
    }
    if (wg_num == 48) {
      maxWG_width[0] = 10.75;
      maxWG_width[1] = 10.75;
    }
  }
  return (fabs(phidif) < aState.dPhiIntMax * aState.strip_iadd * pos_str + dphi_incr &&
          fabs(dr) < aState.dRIntMax * aState.strip_iadd * maxWG_width[iStn])
             ? true
             : false;
}

float CSCSegAlgoRU::phiAtZ(const AlgoState& aState, float z) const {
  if (!aState.sfit)
    return 0.;
  // Returns a phi in [ 0, 2*pi )
  const CSCLayer* l1 = aState.aChamber->layer((*(aState.proto_segment.begin()))->cscDetId().layer());
  GlobalPoint gp = l1->toGlobal(aState.sfit->intercept());
  GlobalVector gv = l1->toGlobal(aState.sfit->localdir());
  float x = gp.x() + (gv.x() / gv.z()) * (z - gp.z());
  float y = gp.y() + (gv.y() / gv.z()) * (z - gp.z());
  float phi = atan2(y, x);
  if (phi < 0.f)
    phi += 2. * M_PI;
  return phi;
}

bool CSCSegAlgoRU::isSegmentGood(const AlgoState& aState, const ChamberHitContainer& rechitsInChamber) const {
  // If the chamber has 20 hits or fewer, require at least 3 hits on segment
  // If the chamber has >20 hits require at least 4 hits
  // If it's the second cycle of the builder and there are <= 12 hits in chamber, require at least 3 hits on segment
  //@@ THESE VALUES SHOULD BECOME PARAMETERS?
  bool ok = false;
  unsigned int iadd = (rechitsInChamber.size() > 20) ? 1 : 0;
  if (aState.windowScale > 1.) {
    iadd = 1;
    if (rechitsInChamber.size() <= 12)
      iadd = 0;
  }
  if (aState.proto_segment.size() >= 3 + iadd)
    ok = true;
  return ok;
}

void CSCSegAlgoRU::flagHitsAsUsed(const AlgoState& aState,
                                  const ChamberHitContainer& rechitsInChamber,
                                  BoolContainer& used) const {
  // Flag hits on segment as used
  ChamberHitContainerCIt ib = rechitsInChamber.begin();
  ChamberHitContainerCIt hi, iu;
  for (hi = aState.proto_segment.begin(); hi != aState.proto_segment.end(); ++hi) {
    for (iu = ib; iu != rechitsInChamber.end(); ++iu) {
      if (*hi == *iu)
        used[iu - ib] = true;
    }
  }
}

bool CSCSegAlgoRU::addHit(AlgoState& aState, const CSCRecHit2D* aHit, int layer) const {
  // Return true if hit was added successfully
  // (and then parameters are updated).
  // Return false if there is already a hit on the same layer, or insert failed.
  ChamberHitContainer::const_iterator it;
  for (it = aState.proto_segment.begin(); it != aState.proto_segment.end(); it++)
    if (((*it)->cscDetId().layer() == layer) && (aHit != (*it)))
      return false;
  aState.proto_segment.push_back(aHit);
  // make a fit
  updateParameters(aState);
  return true;
}

void CSCSegAlgoRU::updateParameters(AlgoState& aState) const {
  // Delete input CSCSegFit, create a new one and make the fit
  // delete sfit;
  aState.sfit = std::make_unique<CSCSegFit>(aState.aChamber, aState.proto_segment);
  aState.sfit->fit();
}

float CSCSegAlgoRU::fit_r_phi(const AlgoState& aState, const SVector6& points, int layer) const {
  //find R or Phi on the given layer using the given points for the interpolation
  float Sx = 0;
  float Sy = 0;
  float Sxx = 0;
  float Sxy = 0;
  for (int i = 1; i < 7; i++) {
    if (points(i - 1) == 0.)
      continue;
    Sy = Sy + (points(i - 1));
    Sx = Sx + i;
    Sxx = Sxx + (i * i);
    Sxy = Sxy + ((i)*points(i - 1));
  }
  float delta = 2 * Sxx - Sx * Sx;
  float intercept = (Sxx * Sy - Sx * Sxy) / delta;
  float slope = (2 * Sxy - Sx * Sy) / delta;
  return (intercept + slope * layer);
}

void CSCSegAlgoRU::baseline(AlgoState& aState, int n_seg_min) const {
  int nhits = aState.proto_segment.size();
  //initialise vectors for strip position and error within strip
  SVector6 sp;
  SVector6 se;
  unsigned int init_size = aState.proto_segment.size();
  ChamberHitContainer buffer;
  buffer.clear();
  buffer.reserve(init_size);
  while (buffer.size() < init_size) {
    ChamberHitContainer::iterator min;
    int min_layer = 10;
    for (ChamberHitContainer::iterator k = aState.proto_segment.begin(); k != aState.proto_segment.end(); k++) {
      const CSCRecHit2D* iRHk = *k;
      CSCDetId idRHk = iRHk->cscDetId();
      int kLayer = idRHk.layer();
      if (kLayer < min_layer) {
        min_layer = kLayer;
        min = k;
      }
    }
    buffer.push_back(*min);
    aState.proto_segment.erase(min);
  }  //while

  aState.proto_segment.clear();
  for (ChamberHitContainer::const_iterator cand = buffer.begin(); cand != buffer.end(); cand++) {
    aState.proto_segment.push_back(*cand);
  }

  for (ChamberHitContainer::const_iterator iRH = aState.proto_segment.begin(); iRH != aState.proto_segment.end();
       iRH++) {
    const CSCRecHit2D* iRHp = *iRH;
    CSCDetId idRH = iRHp->cscDetId();
    int kRing = idRH.ring();
    int kStation = idRH.station();
    int kLayer = idRH.layer();
    // Find the strip containing this hit
    int centerid = iRHp->nStrips() / 2;
    int centerStrip = iRHp->channels(centerid);
    float stpos = (*iRHp).positionWithinStrip();
    se(kLayer - 1) = (*iRHp).errorWithinStrip();
    // Take into account half-strip staggering of layers (ME1/1 has no staggering)
    if (kStation == 1 && (kRing == 1 || kRing == 4))
      sp(kLayer - 1) = stpos + centerStrip;
    else {
      if (kLayer == 1 || kLayer == 3 || kLayer == 5)
        sp(kLayer - 1) = stpos + centerStrip;
      if (kLayer == 2 || kLayer == 4 || kLayer == 6)
        sp(kLayer - 1) = stpos - 0.5 + centerStrip;
    }
  }
  float chi2_str;
  fitX(aState, sp, se, -1, -1, chi2_str);

  //-----------------------------------------------------
  // Optimal point rejection method
  //-----------------------------------------------------
  float minSum = 1000;
  int iworst = -1;
  int bad_layer = -1;
  ChamberHitContainer::const_iterator rh_to_be_deleted_1;
  ChamberHitContainer::const_iterator rh_to_be_deleted_2;
  if ((chi2_str) > aState.chi2_str_ * aState.chi2D_iadd) {  ///(nhits-2)
    for (ChamberHitContainer::const_iterator i1 = aState.proto_segment.begin(); i1 != aState.proto_segment.end();
         ++i1) {
      const CSCRecHit2D* i1_1 = *i1;
      CSCDetId idRH1 = i1_1->cscDetId();
      int z1 = idRH1.layer();
      for (ChamberHitContainer::const_iterator i2 = i1 + 1; i2 != aState.proto_segment.end(); ++i2) {
        const CSCRecHit2D* i2_1 = *i2;
        CSCDetId idRH2 = i2_1->cscDetId();
        int z2 = idRH2.layer();
        int irej = 0;
        for (ChamberHitContainer::const_iterator ir = aState.proto_segment.begin(); ir != aState.proto_segment.end();
             ++ir) {
          ++irej;
          if (ir == i1 || ir == i2)
            continue;
          float dsum = 0;
          const CSCRecHit2D* ir_1 = *ir;
          CSCDetId idRH = ir_1->cscDetId();
          int worst_layer = idRH.layer();
          for (ChamberHitContainer::const_iterator i = aState.proto_segment.begin(); i != aState.proto_segment.end();
               ++i) {
            const CSCRecHit2D* i_1 = *i;
            if (i == i1 || i == i2 || i == ir)
              continue;
            float slope = (sp(z2 - 1) - sp(z1 - 1)) / (z2 - z1);
            float intersept = sp(z1 - 1) - slope * z1;
            CSCDetId idRH = i_1->cscDetId();
            int z = idRH.layer();
            float di = fabs(sp(z - 1) - intersept - slope * z);
            dsum = dsum + di;
          }  //i
          if (dsum < minSum) {
            minSum = dsum;
            bad_layer = worst_layer;
            iworst = irej;
            rh_to_be_deleted_1 = ir;
          }
        }  //ir
      }    //i2
    }      //i1
    fitX(aState, sp, se, bad_layer, -1, chi2_str);
  }  //if chi2prob<1.0e-4

  //find worst from n-1 hits
  int iworst2 = -1;
  int bad_layer2 = -1;
  if (iworst > -1 && (nhits - 1) > n_seg_min && (chi2_str) > aState.chi2_str_ * aState.chi2D_iadd) {  ///(nhits-3)
    iworst = -1;
    float minSum = 1000;
    for (ChamberHitContainer::const_iterator i1 = aState.proto_segment.begin(); i1 != aState.proto_segment.end();
         ++i1) {
      const CSCRecHit2D* i1_1 = *i1;
      CSCDetId idRH1 = i1_1->cscDetId();
      int z1 = idRH1.layer();
      for (ChamberHitContainer::const_iterator i2 = i1 + 1; i2 != aState.proto_segment.end(); ++i2) {
        const CSCRecHit2D* i2_1 = *i2;
        CSCDetId idRH2 = i2_1->cscDetId();
        int z2 = idRH2.layer();
        int irej = 0;
        for (ChamberHitContainer::const_iterator ir = aState.proto_segment.begin(); ir != aState.proto_segment.end();
             ++ir) {
          ++irej;
          int irej2 = 0;
          if (ir == i1 || ir == i2)
            continue;
          const CSCRecHit2D* ir_1 = *ir;
          CSCDetId idRH = ir_1->cscDetId();
          int worst_layer = idRH.layer();
          for (ChamberHitContainer::const_iterator ir2 = aState.proto_segment.begin();
               ir2 != aState.proto_segment.end();
               ++ir2) {
            ++irej2;
            if (ir2 == i1 || ir2 == i2 || ir2 == ir)
              continue;
            float dsum = 0;
            const CSCRecHit2D* ir2_1 = *ir2;
            CSCDetId idRH = ir2_1->cscDetId();
            int worst_layer2 = idRH.layer();
            for (ChamberHitContainer::const_iterator i = aState.proto_segment.begin(); i != aState.proto_segment.end();
                 ++i) {
              const CSCRecHit2D* i_1 = *i;
              if (i == i1 || i == i2 || i == ir || i == ir2)
                continue;
              float slope = (sp(z2 - 1) - sp(z1 - 1)) / (z2 - z1);
              float intersept = sp(z1 - 1) - slope * z1;
              CSCDetId idRH = i_1->cscDetId();
              int z = idRH.layer();
              float di = fabs(sp(z - 1) - intersept - slope * z);
              dsum = dsum + di;
            }  //i
            if (dsum < minSum) {
              minSum = dsum;
              iworst2 = irej2;
              iworst = irej;
              bad_layer = worst_layer;
              bad_layer2 = worst_layer2;
              rh_to_be_deleted_1 = ir;
              rh_to_be_deleted_2 = ir2;
            }
          }  //ir2
        }    //ir
      }      //i2
    }        //i1
    fitX(aState, sp, se, bad_layer, bad_layer2, chi2_str);
  }  //if prob(n-1)<e-4

  //----------------------------------
  //erase bad_hits
  //----------------------------------
  if (iworst2 - 1 >= 0 && iworst2 <= int(aState.proto_segment.size())) {
    aState.proto_segment.erase(rh_to_be_deleted_2);
  }
  if (iworst - 1 >= 0 && iworst <= int(aState.proto_segment.size())) {
    aState.proto_segment.erase(rh_to_be_deleted_1);
  }
}

float CSCSegAlgoRU::fitX(
    const AlgoState& aState, SVector6 points, SVector6 errors, int ir, int ir2, float& chi2_str) const {
  float S = 0;
  float Sx = 0;
  float Sy = 0;
  float Sxx = 0;
  float Sxy = 0;
  float sigma2 = 0;
  for (int i = 1; i < 7; i++) {
    if (i == ir || i == ir2 || points(i - 1) == 0.)
      continue;
    sigma2 = errors(i - 1) * errors(i - 1);
    float i1 = i - 3.5;
    S = S + (1 / sigma2);
    Sy = Sy + (points(i - 1) / sigma2);
    Sx = Sx + ((i1) / sigma2);
    Sxx = Sxx + (i1 * i1) / sigma2;
    Sxy = Sxy + (((i1)*points(i - 1)) / sigma2);
  }
  float delta = S * Sxx - Sx * Sx;
  float intercept = (Sxx * Sy - Sx * Sxy) / delta;
  float slope = (S * Sxy - Sx * Sy) / delta;
  float chi_str = 0;
  chi2_str = 0;
  // calculate chi2_str
  for (int i = 1; i < 7; i++) {
    if (i == ir || i == ir2 || points(i - 1) == 0.)
      continue;
    chi_str = (points(i - 1) - intercept - slope * (i - 3.5)) / (errors(i - 1));
    chi2_str = chi2_str + chi_str * chi_str;
  }
  return (intercept + slope * 0);
}

bool CSCSegAlgoRU::hasHitOnLayer(const AlgoState& aState, int layer) const {
  // Is there is already a hit on this layer?
  ChamberHitContainerCIt it;
  for (it = aState.proto_segment.begin(); it != aState.proto_segment.end(); it++)
    if ((*it)->cscDetId().layer() == layer)
      return true;
  return false;
}

bool CSCSegAlgoRU::replaceHit(AlgoState& aState, const CSCRecHit2D* h, int layer) const {
  // replace a hit from a layer
  ChamberHitContainer::const_iterator it;
  for (it = aState.proto_segment.begin(); it != aState.proto_segment.end();) {
    if ((*it)->cscDetId().layer() == layer)
      it = aState.proto_segment.erase(it);
    else
      ++it;
  }
  return addHit(aState, h, layer);
}

void CSCSegAlgoRU::compareProtoSegment(AlgoState& aState, const CSCRecHit2D* h, int layer) const {
  // Copy the input CSCSegFit
  std::unique_ptr<CSCSegFit> oldfit;
  oldfit = std::make_unique<CSCSegFit>(aState.aChamber, aState.proto_segment);
  oldfit->fit();
  ChamberHitContainer oldproto = aState.proto_segment;

  // May create a new fit
  bool ok = replaceHit(aState, h, layer);
  if ((aState.sfit->chi2() >= oldfit->chi2()) || !ok) {
    // keep original fit
    aState.proto_segment = oldproto;
    aState.sfit = std::move(oldfit);  // reset to the original input fit
  }
}

void CSCSegAlgoRU::increaseProtoSegment(AlgoState& aState, const CSCRecHit2D* h, int layer, int chi2_factor) const {
  // Creates a new fit
  std::unique_ptr<CSCSegFit> oldfit;
  ChamberHitContainer oldproto = aState.proto_segment;
  oldfit = std::make_unique<CSCSegFit>(aState.aChamber, aState.proto_segment);
  oldfit->fit();

  bool ok = addHit(aState, h, layer);
  //@@ TEST ON ndof<=0 IS JUST TO ACCEPT nhits=2 CASE??
  if (!ok || ((aState.sfit->ndof() > 0) && (aState.sfit->chi2() / aState.sfit->ndof() >= aState.chi2Max))) {
    // reset to original fit
    aState.proto_segment = oldproto;
    aState.sfit = std::move(oldfit);
  }
}
