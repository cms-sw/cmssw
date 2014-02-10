#include "GEMCode/GEMValidation/src/DigiMatcher.h"
#include "GEMCode/GEMValidation/src/SimHitMatcher.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <cmath>

using namespace std;
using namespace matching;


namespace {

// translate half-strip number [1..nstrip*2] into fractional strip number [0..nstrip)
float halfstripToStrip(int hs)
{
  //return (hs + 1)/2;
  return 0.5 * hs - 0.25;
}

}


DigiMatcher::DigiMatcher(SimHitMatcher& sh)
: BaseMatcher(sh.trk(), sh.vtx(), sh.conf(), sh.event(), sh.eventSetup())
, simhit_matcher_(&sh)
{
}


DigiMatcher::~DigiMatcher() {}


GlobalPoint
DigiMatcher::digiPosition(const Digi& digi) const
{
  unsigned int id = digi_id(digi);
  int strip = digi_channel(digi);
  DigiType t = digi_type(digi);

  GlobalPoint gp;
  if ( t == GEM_STRIP )
  {
    GEMDetId idd(id);
    LocalPoint lp = gemGeometry_->etaPartition(idd)->centreOfStrip(strip);
    gp = gemGeometry_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_PAD )
  {
    GEMDetId idd(id);
    LocalPoint lp = gemGeometry_->etaPartition(idd)->centreOfPad(strip);
    gp = gemGeometry_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == GEM_COPAD)
  {
    GEMDetId id1(id);
    LocalPoint lp1 = gemGeometry_->etaPartition(id1)->centreOfPad(strip);
    GlobalPoint gp1 = gemGeometry_->idToDet(id)->surface().toGlobal(lp1);

    GEMDetId id2(id1.region(), id1.ring(), id1.station(), 2, id1.chamber(), id1.roll());
    LocalPoint lp2 = gemGeometry_->etaPartition(id2)->centreOfPad(strip);
    GlobalPoint gp2 = gemGeometry_->idToDet(id2())->surface().toGlobal(lp2);

    gp = GlobalPoint( (gp1.x()+gp2.x())/2., (gp1.y()+gp2.y())/2., (gp1.z()+gp2.z())/2.);
  }
  else if ( t == CSC_STRIP )
  {
    CSCDetId idd(id);
    // "strip" here is actually a half-strip in geometry's terms
    int fractional_strip = halfstripToStrip(strip);
    auto strip_topo = cscGeometry_->layer(id)->geometry()->topology();
    LocalPoint lp = strip_topo->localPosition(fractional_strip);
    gp = cscGeometry_->idToDet(id)->surface().toGlobal(lp);
  }
  else if ( t == CSC_CLCT )
  {
    CSCDetId idd(id);
    // "strip" here is actually a half-strip in geometry's terms
    int fractional_strip = halfstripToStrip(strip);
    auto strip_topo = cscGeometry_->chamber(id)->layer(CSCConstants::KEY_CLCT_LAYER)->geometry()->topology();
    LocalPoint lp = strip_topo->localPosition(fractional_strip);

    // return global point on the KEY_CLCT_LAYER layer
    CSCDetId key_id(idd.endcap(), idd.station(), idd.ring(), idd.chamber(), CSCConstants::KEY_CLCT_LAYER);
    gp = cscGeometry_->idToDet(key_id)->surface().toGlobal(lp);
  }
  else if ( t == CSC_LCT )
  {
    CSCDetId idd(id);
    auto layer_geo = cscGeometry_->chamber(idd)->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();

    // "strip" here is actually a half-strip in geometry's terms
    float fractional_strip = halfstripToStrip(strip);
    int wg = digi_wg(digi);
    float wire = layer_geo->middleWireOfGroup(wg);
    LocalPoint intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);

    // return global point on the KEY_CLCT_LAYER layer
    CSCDetId key_id(idd.endcap(), idd.station(), idd.ring(), idd.chamber(), CSCConstants::KEY_CLCT_LAYER);
    gp = cscGeometry_->idToDet(key_id)->surface().toGlobal(intersect);

    if (! layer_geo->inside(intersect))
    {
      cout<<"digiPosition LCT: intersect not inside! hs"<<strip<<" wg"<<wg<<" "<<gp<<endl;
    }
  }
  return gp;
}


GlobalPoint
DigiMatcher::digisMeanPosition(const DigiMatcher::DigiContainer& digis) const
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


int DigiMatcher::median(const DigiContainer& digis) const
{
  size_t sz = digis.size();
  vector<int> strips(sz);
  std::transform(digis.begin(), digis.end(), strips.begin(), [](const Digi& d) {return digi_channel(d);} );
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
DigiMatcher::digisCSCMedianPosition(const DigiMatcher::DigiContainer& strip_digis, const DigiMatcher::DigiContainer& wire_digis) const
{
  if (strip_digis.empty() || wire_digis.empty())
  {
    if (strip_digis.empty()) cout<<"digisCSCMedianPosition strip_digis.empty"<<endl;
    if (wire_digis.empty()) cout<<"digisCSCMedianPosition wire_digis.empty"<<endl;
    return GlobalPoint();
  }

  // assume all strip and wire digis were from the same chamber
  CSCDetId id(digi_id(strip_digis[0]));
  auto layer_geo = cscGeometry_->chamber(id)->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();

  int median_hs = median(strip_digis);
  int median_wg = median(wire_digis);

  float strip = halfstripToStrip(median_hs);
  float wire = layer_geo->middleWireOfGroup(median_wg);
  LocalPoint intersect = layer_geo->intersectionOfStripAndWire(strip, wire);
  if (! layer_geo->inside(intersect))
  {
    cout<<"digisCSCMedianPosition: intersect not inside! hs"<<median_hs<<" wg"<<median_wg<<" "<<intersect<<endl;
  }

  // return global point on the KEY_CLCT_LAYER layer
  CSCDetId key_id(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  return cscGeometry_->idToDet(key_id)->surface().toGlobal(intersect);
}


std::pair<matching::Digi, GlobalPoint>
DigiMatcher::digiInGEMClosestToCSC(const DigiContainer& gem_digis, const GlobalPoint& csc_gp) const
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
    DigiType t = digi_type(d);
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
