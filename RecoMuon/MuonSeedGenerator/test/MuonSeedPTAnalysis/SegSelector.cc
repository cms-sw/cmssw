// Class Header
#include "SegSelector.h"

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TVector3.h"

#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <string>
#include <stdio.h>
#include <algorithm>

//DEFINE_FWK_MODULE(SegSelector);
using namespace std;
using namespace edm;

// constructors
SegSelector::SegSelector(const ParameterSet& pset) {
  //SegSelector::SegSelector(){

  debug = pset.getUntrackedParameter<bool>("debug");
  recHitLabel = pset.getUntrackedParameter<string>("recHitLabel");
  cscSegmentLabel = pset.getUntrackedParameter<string>("cscSegmentLabel");
  dtrecHitLabel = pset.getUntrackedParameter<string>("dtrecHitLabel");
  dtSegmentLabel = pset.getUntrackedParameter<string>("dtSegmentLabel");
  simHitLabel = pset.getUntrackedParameter<string>("simHitLabel");
  simTrackLabel = pset.getUntrackedParameter<string>("simTrackLabel");
}

// destructor
SegSelector::~SegSelector() {
  if (debug)
    cout << "[SeedQualityAnalysis] Destructor called" << endl;
}

// ********************************************
// ***********  Utility functions  ************
// ********************************************
std::vector<SimSegment> SegSelector::Sim_DTSegments(int trkId,
                                                    Handle<edm::PSimHitContainer> dsimHits,
                                                    ESHandle<DTGeometry> dtGeom) {
  SimSegment sim_dseg;

  hit_V1.clear();
  sDT_v.clear();
  int d1 = 0;
  for (PSimHitContainer::const_iterator sh_i = dsimHits->begin(); sh_i != dsimHits->end(); ++sh_i) {
    if (static_cast<int>((*sh_i).trackId()) != trkId)
      continue;
    if (abs((*sh_i).particleType()) != 13)
      continue;

    DTLayerId detId = DTLayerId((*sh_i).detUnitId());

    int d2 = (100000 * detId.station()) + (1000 * detId.sector()) + (100 + detId.wheel());

    if (d1 == 0) {
      d1 = d2;
      hit_V1.clear();
      hit_V1.push_back(*sh_i);
    } else if (d1 == d2) {
      DTLayerId detId1 = DTLayerId(hit_V1[hit_V1.size() - 1].detUnitId());
      if (detId1.layer() != detId.layer()) {
        hit_V1.push_back(*sh_i);
      }
      if ((hit_V1.size() == 12) || ((hit_V1.size() == 8) && (d2 / 100000 == 4))) {
        DTSimHitFit(dtGeom);
        DTChamberId det_seg(detId.wheel(), detId.station(), detId.sector());
        sim_dseg.chamber_type = 2;
        sim_dseg.dt_DetId = det_seg;
        sim_dseg.sLocalOrg = LSimOrg1;
        sim_dseg.sGlobalVec = GSimVec1;
        sim_dseg.sGlobalOrg = GSimOrg1;
        sim_dseg.simhit_v = hit_V1;
        sDT_v.push_back(sim_dseg);
      }
    } else {
      if ((hit_V1.size() < 12) && (hit_V1.size() > 5)) {
        DTSimHitFit(dtGeom);
        int old_st = d1 / 100000;
        int old_se = (d1 - (old_st * 100000)) / 1000;
        int old_wl = (d1 - (old_st * 100000) - (old_se * 1000)) - 100;
        DTChamberId det_seg(old_wl, old_st, old_se);
        sim_dseg.chamber_type = 2;
        sim_dseg.dt_DetId = det_seg;
        sim_dseg.sLocalOrg = LSimOrg1;
        sim_dseg.sGlobalVec = GSimVec1;
        sim_dseg.sGlobalOrg = GSimOrg1;
        sim_dseg.simhit_v = hit_V1;
        sDT_v.push_back(sim_dseg);
      }
      d1 = d2;
      hit_V1.clear();
      hit_V1.push_back(*sh_i);
    }
  }
  return sDT_v;
}

// build a sim-segment sets
std::vector<SimSegment> SegSelector::Sim_CSCSegments(int trkId,
                                                     Handle<edm::PSimHitContainer> csimHits,
                                                     ESHandle<CSCGeometry> cscGeom) {
  SimSegment sim_cseg;

  // collect the simhits in the same chamber and then build a sim segment
  hit_V.clear();
  sCSC_v.clear();
  int d1 = 0;
  for (PSimHitContainer::const_iterator sh_i = csimHits->begin(); sh_i != csimHits->end(); ++sh_i) {
    if (static_cast<int>((*sh_i).trackId()) != trkId)
      continue;
    if (abs((*sh_i).particleType()) != 13)
      continue;
    //if ( (*sh_i).particleType()!= -13 ) continue;
    CSCDetId detId = CSCDetId((*sh_i).detUnitId());

    int d2 = (1000 * detId.station()) + (100 * detId.ring()) + detId.chamber();
    if (d1 == 0) {
      d1 = d2;
      hit_V.clear();
      hit_V.push_back(*sh_i);
    } else if (d1 == d2) {
      hit_V.push_back(*sh_i);
      if (hit_V.size() == 6) {
        CSCSimHitFit(cscGeom);
        int old_st = d1 / 1000;
        int old_rg = (d1 - (old_st * 1000)) / 100;
        int old_cb = d1 - (old_st * 1000) - (old_rg * 100);
        CSCDetId det_seg(detId.endcap(), old_st, old_rg, old_cb, 0);
        sim_cseg.chamber_type = 1;
        sim_cseg.csc_DetId = det_seg;
        sim_cseg.sLocalOrg = LSimOrg;
        sim_cseg.sGlobalVec = GSimVec;
        sim_cseg.sGlobalOrg = GSimOrg;
        sim_cseg.simhit_v = hit_V;
        sCSC_v.push_back(sim_cseg);
      }
    } else {
      if ((hit_V.size() < 6) && (hit_V.size() > 2)) {
        CSCSimHitFit(cscGeom);
        CSCDetId det_seg(detId.endcap(), detId.station(), detId.ring(), detId.chamber(), 0);
        sim_cseg.chamber_type = 1;
        sim_cseg.csc_DetId = det_seg;
        sim_cseg.sLocalOrg = LSimOrg;
        sim_cseg.sGlobalVec = GSimVec;
        sim_cseg.sGlobalOrg = GSimOrg;
        sim_cseg.simhit_v = hit_V;
        sCSC_v.push_back(sim_cseg);
      }
      d1 = d2;
      hit_V.clear();
      hit_V.push_back(*sh_i);
    }
  }
  return sCSC_v;
}

// pick up the DT Segments which are studied
std::vector<DTRecSegment4D> SegSelector::Select_DTSeg(Handle<DTRecSegment4DCollection> dtSeg,
                                                      ESHandle<DTGeometry> dtGeom,
                                                      std::vector<SimSegment> sDT_v1) {
  dtseg_V.clear();
  for (std::vector<SimSegment>::const_iterator it1 = sDT_v1.begin(); it1 != sDT_v1.end(); it1++) {
    // pick up the segment which are alone in chamber
    std::vector<DTRecSegment4D> segtemp;
    segtemp.clear();
    for (DTRecSegment4DCollection::const_iterator it2 = dtSeg->begin(); it2 != dtSeg->end(); it2++) {
      //if ( (*it2).dimension() != 4 ) continue;
      if (!(*it2).hasPhi())
        continue;
      if ((*it2).chamberId().station() < 4 && !(*it2).hasZed())
        continue;

      if ((*it1).dt_DetId != (*it2).chamberId())
        continue;

      segtemp.push_back(*it2);
    }
    if (segtemp.size() == 1) {
      dtseg_V.push_back(segtemp[0]);
    }
    if (segtemp.size() > 1) {
      // pick-up the relative long segments
      LongDTSegment(segtemp);
      if (longsegV1.size() == 1) {
        dtseg_V.push_back(longsegV1[0]);
      }
      if (longsegV1.size() > 1) {
        // picking up the rec-segment which is the closest to sim-segment
        std::vector<double> dgv1(7, 999.);
        double k = 0.;
        DTRecSegment4D closestseg;
        for (std::vector<DTRecSegment4D>::const_iterator it3 = longsegV1.begin(); it3 != longsegV1.end(); it3++) {
          k += 1.0;
          DTChamberId rDetId = (*it3).chamberId();
          LocalVector rec_v = (*it3).localDirection();
          LocalPoint rec_o = (*it3).localPosition();
          const DTChamber* dtchamber = dtGeom->chamber(rDetId);
          GlobalVector g_rec_v = dtchamber->toGlobal(rec_v);
          GlobalPoint g_rec_o = dtchamber->toGlobal(rec_o);

          std::vector<double> dgv2(7, 999.);
          dgv2[0] = ((*it1).sGlobalVec).x() - g_rec_v.x();
          dgv2[1] = ((*it1).sGlobalVec).y() - g_rec_v.y();
          dgv2[2] = ((*it1).sGlobalVec).z() - g_rec_v.z();
          dgv2[3] = fabs(((*it1).sGlobalOrg).x() - g_rec_o.x());
          dgv2[4] = fabs(((*it1).sGlobalOrg).y() - g_rec_o.y());
          dgv2[5] = sqrt((dgv2[0] * dgv2[0]) + (dgv2[1] * dgv2[1]) + (dgv2[2] * dgv2[2]));
          dgv2[6] = k;

          if (k == 1.0) {
            dgv1 = dgv2;
            closestseg = *it3;
          } else {
            closestseg = (dgv2[3] < dgv1[3]) ? (*it3) : closestseg;
            dgv1 = (dgv2[3] < dgv1[3]) ? dgv2 : dgv1;
          }
        }
        dtseg_V.push_back(closestseg);
      }
    }
  }
  return dtseg_V;
}

// pick up the CSC segments which are studied
std::vector<CSCSegment> SegSelector::Select_CSCSeg(Handle<CSCSegmentCollection> cscSeg,
                                                   ESHandle<CSCGeometry> cscGeom,
                                                   std::vector<SimSegment> sCSC_v1) {
  cscseg_V.clear();
  for (std::vector<SimSegment>::const_iterator it1 = sCSC_v1.begin(); it1 != sCSC_v1.end(); it1++) {
    // pick up the segment which are alone in chamber
    std::vector<CSCSegment> segtemp;
    segtemp.clear();
    for (CSCSegmentCollection::const_iterator seg_i = cscSeg->begin(); seg_i != cscSeg->end(); seg_i++) {
      CSCDetId rDetId = (CSCDetId)(*seg_i).cscDetId();
      if ((*it1).csc_DetId == rDetId) {
        segtemp.push_back(*seg_i);
      }
    }
    if (segtemp.size() == 1) {
      cscseg_V.push_back(segtemp[0]);
    }
    if (segtemp.size() > 1) {
      // pick-up the relative long segments
      LongCSCSegment(segtemp);
      if (longsegV.size() == 1) {
        cscseg_V.push_back(longsegV[0]);
      }
      if (longsegV.size() > 1) {
        // picking up the rec-segment which is the closest to sim-segment
        std::vector<double> dgv1(7, 999.);
        double k = 0.;
        CSCSegment closestseg;
        for (std::vector<CSCSegment>::const_iterator i1 = longsegV.begin(); i1 != longsegV.end(); i1++) {
          k += 1.0;
          CSCDetId rDetId = (CSCDetId)(*i1).cscDetId();
          LocalVector rec_v = (*i1).localDirection();
          LocalPoint rec_o = (*i1).localPosition();
          const CSCChamber* cscchamber = cscGeom->chamber(rDetId);
          GlobalVector g_rec_v = cscchamber->toGlobal(rec_v);
          GlobalPoint g_rec_o = cscchamber->toGlobal(rec_o);

          std::vector<double> dgv2(7, 999.);
          dgv2[0] = ((*it1).sGlobalVec).x() - g_rec_v.x();
          dgv2[1] = ((*it1).sGlobalVec).y() - g_rec_v.y();
          dgv2[2] = ((*it1).sGlobalVec).z() - g_rec_v.z();
          dgv2[3] = fabs(((*it1).sGlobalOrg).x() - g_rec_o.x());
          dgv2[4] = fabs(((*it1).sGlobalOrg).y() - g_rec_o.y());
          dgv2[5] = sqrt((dgv2[0] * dgv2[0]) + (dgv2[1] * dgv2[1]) + (dgv2[2] * dgv2[2]));
          dgv2[6] = k;

          if (k == 1.0) {
            dgv1 = dgv2;
            closestseg = *i1;
          } else {
            closestseg = (dgv2[3] < dgv1[3]) ? (*i1) : closestseg;
            dgv1 = (dgv2[3] < dgv1[3]) ? dgv2 : dgv1;
          }
        }
        cscseg_V.push_back(closestseg);
      }
    }
  }
  return cscseg_V;
}

// collect the long csc segments
void SegSelector::LongCSCSegment(std::vector<CSCSegment> cscsegs) {
  int n = 0;
  CSCSegment longseg;
  longsegV.clear();
  for (std::vector<CSCSegment>::const_iterator i1 = cscsegs.begin(); i1 != cscsegs.end(); i1++) {
    n++;
    if (n == 1) {
      longseg = *i1;
      longsegV.push_back(*i1);
    } else if ((*i1).nRecHits() == longseg.nRecHits()) {
      longsegV.push_back(*i1);
    } else {
      longseg = ((*i1).nRecHits() > longseg.nRecHits()) ? (*i1) : longseg;
      longsegV.clear();
      longsegV.push_back(longseg);
    }
  }
}

// collect the long dt segments
void SegSelector::LongDTSegment(std::vector<DTRecSegment4D> dtsegs) {
  int n = 0;
  int nh = 0;
  DTRecSegment4D longseg;
  longsegV1.clear();
  for (std::vector<DTRecSegment4D>::const_iterator i1 = dtsegs.begin(); i1 != dtsegs.end(); i1++) {
    n++;
    int n_phi = 0;
    int n_zed = 0;
    if ((*i1).hasPhi())
      n_phi = ((*i1).phiSegment())->specificRecHits().size();
    if ((*i1).hasZed())
      n_zed = ((*i1).zSegment())->specificRecHits().size();
    int n_hits = n_phi + n_zed;

    if (n == 1) {
      longseg = *i1;
      longsegV1.push_back(*i1);
      nh = n_hits;
    } else if (n_hits == nh) {
      longsegV1.push_back(*i1);
    } else {
      longseg = (n_hits > nh) ? (*i1) : longseg;
      longsegV1.clear();
      longsegV1.push_back(longseg);
    }
  }
}

// DT Sim Segment fitting
void SegSelector::DTSimHitFit(ESHandle<DTGeometry> dtGeom) {
  std::vector<PSimHit> sp;
  std::vector<PSimHit> sp1;
  std::vector<PSimHit> sp2;
  sp1.clear();
  sp2.clear();
  DTLayerId DT_Id;
  for (std::vector<PSimHit>::const_iterator sh_i = hit_V1.begin(); sh_i != hit_V1.end(); ++sh_i) {
    DT_Id = DTLayerId((*sh_i).detUnitId());
    if ((DT_Id.superLayer() == 1) || (DT_Id.superLayer() == 3)) {
      sp1.push_back(*sh_i);
    }
    if (DT_Id.superLayer() == 2) {
      sp2.push_back(*sh_i);
    }
  }
  DTLayerId DT_Id0(DT_Id.wheel(), DT_Id.station(), DT_Id.sector(), 0, 0);

  // Simple Least Square Fit without error weighting
  // sum[0]= sum_x ,  sum[1]=sum_z , sum[2]=sum_xz , sum[3]=sum_z^2
  // par1[0]= orig_x , par1[1]=slop_xz ;  x = par1[1]*z + par1[0]
  // par2[0]= orig_y , par2[1]=slop_yz ;  y = par2[1]*z + par2[0]
  double par1[2] = {0.0};
  double par2[2] = {0.0};
  for (int i = 1; i < 4; i++) {
    double sum[4] = {0.0};
    double par[2] = {0.0};
    double N = 0.0;
    if (i == 1) {
      sp = sp1;
    }
    if (i == 2) {
      sp = sp2;
    }
    if ((i == 3) && (sp2.size() > 0))
      continue;
    if (i == 3) {
      sp = sp1;
    }
    for (std::vector<PSimHit>::const_iterator sh_i = sp.begin(); sh_i != sp.end(); ++sh_i) {
      DTWireId DT_Id = DTWireId((*sh_i).detUnitId());
      LocalPoint dt_xyz = (*sh_i).localPosition();

      const DTLayer* dtlayer = dtGeom->layer(DT_Id);
      const DTChamber* dtchamber = dtGeom->chamber(DT_Id);
      GlobalPoint gdt_xyz = dtlayer->toGlobal(dt_xyz);
      LocalPoint ldt_xyz = dtchamber->toLocal(gdt_xyz);

      N += 1.0;
      if (i == 1) {
        sum[0] += ldt_xyz.x();
        sum[1] += ldt_xyz.z();
        sum[2] += ldt_xyz.x() * ldt_xyz.z();
        sum[3] += ldt_xyz.z() * ldt_xyz.z();
      }
      if ((i == 2) || (i == 3)) {
        sum[0] += ldt_xyz.y();
        sum[1] += ldt_xyz.z();
        sum[2] += ldt_xyz.y() * ldt_xyz.z();
        sum[3] += ldt_xyz.z() * ldt_xyz.z();
      }
    }
    par[0] = ((sum[0] * sum[3]) - (sum[2] * sum[1])) / ((N * sum[3]) - (sum[1] * sum[1]));
    par[1] = ((N * sum[2]) - (sum[0] * sum[1])) / ((N * sum[3]) - (sum[1] * sum[1]));
    if (i == 1) {
      par1[0] = par[0];
      par1[1] = par[1];
    }
    if ((i == 2) || (i == 3)) {
      par2[0] = par[0];
      par2[1] = par[1];
    }
  }

  double v_zz = -1. / sqrt((par1[1] * par1[1]) + (par2[1] * par2[1]) + 1.0);
  double v_xx = v_zz * par1[1];
  double v_yy = v_zz * par2[1];
  LSimVec1 = LocalVector(v_xx, v_yy, v_zz);
  LSimOrg1 = LocalPoint(par1[0], par2[0], 0.);

  const DTLayer* DT_layer = dtGeom->layer(DT_Id0);
  GSimOrg1 = DT_layer->toGlobal(LSimOrg1);
  GSimVec1 = DT_layer->toGlobal(LSimVec1);
}

// CSC Sim Segment fitting
void SegSelector::CSCSimHitFit(ESHandle<CSCGeometry> cscGeom) {
  bool rv_flag = false;
  for (std::vector<PSimHit>::const_iterator sh_i = hit_V.begin(); sh_i != hit_V.end(); ++sh_i) {
    CSCDetId Sim_Id1 = (CSCDetId)(*sh_i).detUnitId();
    const CSCChamber* cscchamber = cscGeom->chamber(Sim_Id1);

    double z1 = (cscchamber->layer(1))->position().z();
    double z6 = (cscchamber->layer(6))->position().z();

    if (((z1 > z6) && (z1 > 0.)) || ((z1 < z6) && (z1 < 0.))) {
      rv_flag = true;
    }
  }

  // Simple Least Square Fit without error weighting
  // sum1[0]=sum_x , sum1[1]=sum_z , sum1[2]=sum_xz , sum1[3]=sum_z^2
  // sum2[0]=sum_y , sum2[1]=sum_z , sum2[2]=sum_yz , sum2[3]=sum_z^2
  // par1[0]= orig_x , par1[1]=slop_xz ;  x = par1[1]*z + par1[0]
  // par2[0]= orig_y , par2[1]=slop_yz ;  y = par2[1]*z + par2[0]
  double sum1[4] = {0.0};
  double sum2[4] = {0.0};
  par1[0] = 0.0;
  par1[1] = 0.0;
  par2[0] = 0.0;
  par2[1] = 0.0;
  double N = 0.0;
  CSCDetId Sim_Id;
  for (std::vector<PSimHit>::const_iterator sh_i = hit_V.begin(); sh_i != hit_V.end(); ++sh_i) {
    Sim_Id = (CSCDetId)(*sh_i).detUnitId();
    LocalPoint Sim_xyz = (*sh_i).localPosition();

    const CSCLayer* csclayer = cscGeom->layer(Sim_Id);
    const CSCChamber* cscchamber = cscGeom->chamber(Sim_Id);
    GlobalPoint gsh_xyz = csclayer->toGlobal(Sim_xyz);
    LocalPoint lsh_xyz = cscchamber->toLocal(gsh_xyz);
    double zz = lsh_xyz.z();

    if (rv_flag) {
      zz = (-1.0) * lsh_xyz.z();
    }

    N += 1.0;
    sum1[0] += lsh_xyz.x();
    sum1[1] += zz;
    sum1[2] += lsh_xyz.x() * zz;
    sum1[3] += zz * zz;

    sum2[0] += lsh_xyz.y();
    sum2[1] += zz;
    sum2[2] += lsh_xyz.y() * zz;
    sum2[3] += zz * zz;
  }
  par1[0] = ((sum1[0] * sum1[3]) - (sum1[2] * sum1[1])) / ((N * sum1[3]) - (sum1[1] * sum1[1]));
  par1[1] = ((N * sum1[2]) - (sum1[0] * sum1[1])) / ((N * sum1[3]) - (sum1[1] * sum1[1]));

  par2[0] = ((sum2[0] * sum2[3]) - (sum2[2] * sum2[1])) / ((N * sum2[3]) - (sum2[1] * sum2[1]));
  par2[1] = ((N * sum2[2]) - (sum2[0] * sum2[1])) / ((N * sum2[3]) - (sum2[1] * sum2[1]));

  double dzz = 1. / sqrt((par1[1] * par1[1]) + (par2[1] * par2[1]) + 1.0);
  double dxx = dzz * par1[1];
  double dyy = dzz * par2[1];
  LocalVector LV = LocalVector(dxx, dyy, dzz);
  LSimOrg = LocalPoint(par1[0], par2[0], 0.);

  const CSCChamber* cscchamber = cscGeom->chamber(Sim_Id);
  GlobalVector GV = cscchamber->toGlobal(LV);
  GlobalPoint GP = cscchamber->toGlobal(LSimOrg);
  LSimVec = LV;
  GSimOrg = GP;

  double directionSign = GP.z() * GV.z();
  LV = (directionSign * LV).unit();
  GV = cscchamber->toGlobal(LV);
  GSimVec = GV;

  if ((Sim_Id.station() == 3) || (Sim_Id.station() == 4)) {
    GSimVec = GlobalVector(-1. * GV.x(), -1. * GV.y(), GV.z());
  } else {
    GSimVec = GV;
  }

  /*
     if ( (Sim_Id.station()==1)||(Sim_Id.station()==2) ) {
        GSimVec = GlobalVector( GV.x(), GV.y(), (-1.*GV.z()) );
     }else {
        GSimVec = GV;
     }
     */

  // flip the wrong global vector for sim-segment
  /*
     if ( (( GV.z() >= 0.0) && (Sim_Id.endcap()==2)) ||
          (( GV.z() < 0.0) && (Sim_Id.endcap()==1)) ){
        LSimVec = LocalVector((-1.*dxx), (-1.*dyy), (-1.*dzz));         
     }
     else {
          LSimVec = LocalVector(dxx, dyy, dzz);
     }
     */
  //GSimVec = cscchamber->toGlobal(LSimVec);
}
