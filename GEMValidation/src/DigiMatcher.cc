#include "DigiMatcher.h"
#include "SimHitMatcher.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <cmath>

using namespace std;

namespace {

// translate half-strip number [1..nstrip*2] into fractional strip number [0..nstrip)
float halfstripToStrip(int hs)
{
  //return (hs + 1)/2;
  return 0.5 * hs - 0.25;
}


bool is_gem(unsigned int detid)
{
  DetId id;
  if (id.subdetId() == MuonSubdetId::GEM) return true;
  return false;
}

bool is_csc(unsigned int detid)
{
  DetId id;
  if (id.subdetId() == MuonSubdetId::CSC) return true;
  return false;
}

}


DigiMatcher::DigiMatcher(SimHitMatcher& sh)
: BaseMatcher(sh.trk(), sh.vtx(), sh.conf(), sh.event(), sh.eventSetup())
, simhit_matcher_(&sh)
{
  edm::ESHandle<CSCGeometry> csc_g;
  eventSetup()->get<MuonGeometryRecord>().get(csc_g);
  csc_geo_ = &*csc_g;

  edm::ESHandle<GEMGeometry> gem_g;
  eventSetup()->get<MuonGeometryRecord>().get(gem_g);
  gem_geo_ = &*gem_g;
}


DigiMatcher::~DigiMatcher() {}


GlobalPoint
DigiMatcher::digiPosition(const Digi& digi)
{
  unsigned int id;
  int strip;
  DigiType t;
  std::tie(id, strip, std::ignore, t) = digi;

  GlobalPoint gp;
  if ( t == GEM_STRIP )
  {
    GEMDetId idd(id);
    LocalPoint lp = gem_geo_->etaPartition(idd)->centreOfStrip(strip);
    gp = gem_geo_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_PAD )
  {
    GEMDetId idd(id);
    LocalPoint lp = gem_geo_->etaPartition(idd)->centreOfPad(strip);
    gp = gem_geo_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_COPAD)
  {
    GEMDetId id1(id);
    LocalPoint lp1 = gem_geo_->etaPartition(id1)->centreOfPad(strip);
    GlobalPoint gp1 = gem_geo_->idToDet(id)->surface().toGlobal(lp1);

    GEMDetId id2(id1.region(), id1.ring(), id1.station(), 2, id1.chamber(), id1.roll());
    LocalPoint lp2 = gem_geo_->etaPartition(id2)->centreOfPad(strip);
    GlobalPoint gp2 = gem_geo_->idToDet(id2())->surface().toGlobal(lp2);

    gp = GlobalPoint( (gp1.x()+gp2.x())/2., (gp1.y()+gp2.y())/2., (gp1.z()+gp2.z())/2.);
  }
  else if ( t == CSC_STRIP )
  {
    CSCDetId idd(id);
    // "strip" here is actually a half-strip in geometry's terms
    int fractional_strip = halfstripToStrip(strip);
    auto strip_topo = csc_geo_->layer(id)->geometry()->topology();
    LocalPoint lp = strip_topo->localPosition(fractional_strip);
    gp = csc_geo_->idToDet(id)->surface().toGlobal(lp);
  }
  return gp;
}


GlobalPoint
DigiMatcher::digisMeanPosition(const DigiMatcher::DigiContainer& digis)
{
  GlobalPoint point_zero;
  if (digis.empty()) return point_zero; // point "zero"

  float sumx, sumy, sumz;
  sumx = sumy = sumz = 0.f;
  size_t n = 0;
  for (auto& d: digis)
  {
    GlobalPoint gp = digiPosition(d);
    if (gp == point_zero) continue;

    sumx += gp.x();
    sumy += gp.y();
    sumz += gp.z();
    ++n;
  }
  if (n == 0) return GlobalPoint();
  return GlobalPoint(sumx/n, sumy/n, sumz/n);
}


int DigiMatcher::median(const DigiContainer& digis)
{
  size_t sz = digis.size();
  vector<int> strips(sz);
  std::transform(digis.begin(), digis.end(), strips.begin(), [](const Digi& d) {return std::get<1>(d);} );
  std::sort(strips.begin(), strips.end());
  if ( sz % 2 == 0 ) // even
  {
    return (strips[sz/2 - 1] + strips[sz/2])/2;
  }
  else
  {
    return strips[sz/2];
  }
}


GlobalPoint
DigiMatcher::digisCSCMedianPosition(const DigiMatcher::DigiContainer& strip_digis, const DigiMatcher::DigiContainer& wire_digis)
{
  if (strip_digis.empty() || wire_digis.empty())
  {
    if (strip_digis.empty()) cout<<"digisCSCMedianPosition strip_digis.empty"<<endl;
    if (wire_digis.empty()) cout<<"digisCSCMedianPosition wire_digis.empty"<<endl;
    return GlobalPoint();
  }

  // assume all strip and wire digis were from the same chamber
  CSCDetId id(std::get<0>(strip_digis[0]));
  auto layer_geo = csc_geo_->chamber(id)->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();

  int median_hs = median(strip_digis);
  int median_wg = median(wire_digis);

  float strip = halfstripToStrip(median_hs);
  float wire = layer_geo->middleWireOfGroup(median_wg);
  LocalPoint intersect = layer_geo->intersectionOfStripAndWire(strip, wire);
  if (! layer_geo->inside(intersect)) {cout<<"digisCSCMedianPosition: intersect not inside!"<<endl; return GlobalPoint();}

  // return global point on the KEY_CLCT_LAYER layer
  CSCDetId key_id(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  return csc_geo_->idToDet(key_id)->surface().toGlobal(intersect);
}

std::pair<DigiMatcher::Digi, GlobalPoint>
DigiMatcher::digiInGEMClosestToCSC(const DigiContainer& gem_digis, const GlobalPoint& csc_gp)
{
  GlobalPoint gp;
  Digi best_digi;

  if (gem_digis.empty() || std::abs(csc_gp.z()) < 0.001 ) // no digis or bad CSC input
  {
    if (gem_digis.empty()) cout<<"digiInGEMClosestToCSC gem_digis.empty"<<endl;
    if (std::abs(csc_gp.z()) < 0.001 ) cout<<"digiInGEMClosestToCSC wire_digis.empty"<<endl;
    return make_pair(best_digi, gp);
  }

  float prev_dr2 = 99999.;
  for (auto& d: gem_digis)
  {
    DigiType t = std::get<3>(d);
    if ( !(t == GEM_STRIP || t == GEM_PAD || t == GEM_COPAD) ) continue;

    GlobalPoint curr_gp = digiPosition(d);
    if (std::abs(curr_gp.z()) < 0.001) continue; // invalid position

    // in deltaR calculation, give x20 larger weight to deltaPhi to make them comparable
    // but with slight bias towards dphi:
    float dphi = 20.*deltaPhi(csc_gp.phi(), curr_gp.phi());
    float deta = csc_gp.eta() - curr_gp.eta();
    float curr_dr2 = dphi*dphi + deta*deta;
    if (std::abs(gp.z()) < 000.1 || // gp was not assigned yet
        curr_dr2 < prev_dr2 ) // current gp is closer in phi then the previous
    {
      gp = curr_gp;
      best_digi = d;
      prev_dr2 = curr_dr2;
    }
  }
  return make_pair(best_digi, gp);
}
