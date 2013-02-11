#include "CSCDigiMatcher.h"
#include "SimHitMatcher.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

using namespace std;


CSCDigiMatcher::CSCDigiMatcher(SimHitMatcher& sh)
: DigiMatcher(sh)
{
  cscComparatorDigiInput_ = conf()->getUntrackedParameter<edm::InputTag>("cscComparatorDigiInput",
      edm::InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi"));
  cscWireDigiInput_ = conf()->getUntrackedParameter<edm::InputTag>("cscWireDigiInput",
      edm::InputTag("simMuonCSCDigis", "MuonCSCWireDigi"));

  minBXCSCComp_ = conf()->getUntrackedParameter<int>("minBXCSCComp", 3);
  maxBXCSCComp_ = conf()->getUntrackedParameter<int>("maxBXCSCComp", 9);
  minBXCSCWire_ = conf()->getUntrackedParameter<int>("minBXCSCWire", 3);
  maxBXCSCWire_ = conf()->getUntrackedParameter<int>("maxBXCSCWire", 8);

  matchDeltaStrip_ = conf()->getUntrackedParameter<int>("matchDeltaStripCSC", 1);

  setVerbose(conf()->getUntrackedParameter<int>("verboseCSCDigi", 0));

  init();
}


CSCDigiMatcher::~CSCDigiMatcher() {}


void CSCDigiMatcher::init()
{
  edm::Handle<CSCComparatorDigiCollection> comp_digis;
  event()->getByLabel(cscComparatorDigiInput_, comp_digis);

  edm::Handle<CSCWireDigiCollection> wire_digis;
  event()->getByLabel(cscWireDigiInput_, wire_digis);

  matchTriggerDigisToSimTrack(*comp_digis.product(), *wire_digis.product());
}


void
CSCDigiMatcher::matchTriggerDigisToSimTrack(const CSCComparatorDigiCollection& comparators, const CSCWireDigiCollection& wires)
{
  auto det_ids = simhit_matcher_->detIdsCSC();
  for (auto id: det_ids)
  {
    CSCDetId layer_id(id);
    //auto layer_geo = csc_geo_->layer(CSCDetId())->geometry();
    int max_nstrips = csc_geo_->layer(id)->geometry()->topology()->nstrips();

    auto comps_in_det = comparators.get(layer_id);

    auto hit_strips = simhit_matcher_->hitStripsInDetId(id);
    set<int> hit_strips_fat;
    for (auto s: hit_strips)
    {
      int smin = s - matchDeltaStrip_;
      smin = (smin > 0) ? smin : 1;
      int smax = s + matchDeltaStrip_;
      smax = (smax <= max_nstrips) ? smax : max_nstrips;
      for (int ss = smin; ss <= smax; ++ss) hit_strips_fat.insert(ss);
    }

    if (verbose())
    {
      cout<<"sit_strips "<<layer_id<<" ";
      copy(hit_strips.begin(), hit_strips.end(), ostream_iterator<int>(cout, " "));
      cout<<endl;
      cout<<"sit_strips_fat ";
      copy(hit_strips_fat.begin(), hit_strips_fat.end(), ostream_iterator<int>(cout, " "));
      cout<<endl;
    }

    for (auto c = comps_in_det.first; c != comps_in_det.second; ++c)
    {
      if (verbose()) cout<<"sdigi "<<layer_id<<" "<<*c<<endl;

      // check that the first BX for this digi wasn't too early or too late
      if (c->getTimeBin() < minBXCSCComp_ || c->getTimeBin() > maxBXCSCComp_) continue;

      int strip = c->getStrip(); // strips are counted from 1
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_strips_fat.find(strip) == hit_strips_fat.end()) continue;
      if (verbose()) cout<<"oki"<<endl;

      // get half-strip, counting from 1
      int half_strip = 2*strip - 1 + c->getComparator();

      auto mydigi = std::make_tuple(id, half_strip, c->getTimeBin(), CSC_STRIP);
      detid_to_halfstrips_[id].push_back(mydigi);
      chamber_to_halfstrips_[ layer_id.chamberId().rawId() ].push_back(mydigi);
    }

    auto hit_wires = simhit_matcher_->hitWiregroupsInDetId(id);
    auto wires_in_det = wires.get(layer_id);

    for (auto w = wires_in_det.first; w != wires_in_det.second; ++w)
    {
      // check that the first BX for this digi wasn't too early or too late
      if (w->getTimeBin() < minBXCSCWire_ || w->getTimeBin() > maxBXCSCWire_) continue;

      int wg = w->getWireGroup(); // wiregroups are counted from 1
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_wires.find(wg) == hit_wires.end()) continue;

      auto mydigi = std::make_tuple(id, wg, w->getTimeBin(), CSC_WIRE);
      detid_to_wires_[id].push_back(mydigi);
      chamber_to_wires_[ layer_id.chamberId().rawId() ].push_back(mydigi);
    }
  }
}


std::set<unsigned int>
CSCDigiMatcher::detIdsStrip()
{
  std::set<unsigned int> result;
  for (auto& p: detid_to_halfstrips_) result.insert(p.first);
  return result;
}

std::set<unsigned int>
CSCDigiMatcher::detIdsWire()
{
  std::set<unsigned int> result;
  for (auto& p: detid_to_wires_) result.insert(p.first);
  return result;
}


std::set<unsigned int>
CSCDigiMatcher::chamberIdsStrip()
{
  std::set<unsigned int> result;
  for (auto& p: chamber_to_halfstrips_) result.insert(p.first);
  return result;
}

std::set<unsigned int>
CSCDigiMatcher::chamberIdsWire()
{
  std::set<unsigned int> result;
  for (auto& p: chamber_to_wires_) result.insert(p.first);
  return result;
}


CSCDigiMatcher::DigiContainer
CSCDigiMatcher::stripDigisInDetId(unsigned int detid)
{
  if (detid_to_halfstrips_.find(detid) == detid_to_halfstrips_.end()) return DigiContainer();
  return detid_to_halfstrips_[detid];
}

CSCDigiMatcher::DigiContainer
CSCDigiMatcher::stripDigisInChamber(unsigned int detid)
{
  if (chamber_to_halfstrips_.find(detid) == chamber_to_halfstrips_.end()) return DigiContainer();
  return chamber_to_halfstrips_[detid];
}

CSCDigiMatcher::DigiContainer
CSCDigiMatcher::wireDigisInDetId(unsigned int detid)
{
  if (detid_to_wires_.find(detid) == detid_to_wires_.end()) return DigiContainer();
  return detid_to_wires_[detid];
}

CSCDigiMatcher::DigiContainer
CSCDigiMatcher::wireDigisInChamber(unsigned int detid)
{
  if (chamber_to_wires_.find(detid) == chamber_to_wires_.end()) return DigiContainer();
  return chamber_to_wires_[detid];
}


int
CSCDigiMatcher::nLayersWithStripInChamber(unsigned int detid)
{
  set<int> layers_with_hits;
  auto digis = stripDigisInChamber(detid);
  for (auto& d: digis)
  {
    CSCDetId idd(std::get<0>(d));
    layers_with_hits.insert(idd.layer());
  }
  return layers_with_hits.size();
}

int
CSCDigiMatcher::nLayersWithWireInChamber(unsigned int detid)
{
  set<int> layers_with_hits;
  auto digis = wireDigisInChamber(detid);
  for (auto& d: digis)
  {
    CSCDetId idd(std::get<0>(d));
    layers_with_hits.insert(idd.layer());
  }
  return layers_with_hits.size();
}


int
CSCDigiMatcher::nCoincidenceStripChambers(int min_n_layers)
{
  int result = 0;
  auto chamber_ids = chamberIdsStrip();
  for (auto id: chamber_ids)
  {
    if (nLayersWithStripInChamber(id) >= min_n_layers) result += 1;
  }
  return result;
}

int
CSCDigiMatcher::nCoincidenceWireChambers(int min_n_layers)
{
  int result = 0;
  auto chamber_ids = chamberIdsWire();
  for (auto id: chamber_ids)
  {
    if (nLayersWithWireInChamber(id) >= min_n_layers) result += 1;
  }
  return result;
}


std::set<int>
CSCDigiMatcher::stripsInDetId(unsigned int detid)
{
  set<int> result;
  DigiContainer digis;
  digis = stripDigisInDetId(detid);
  for (auto& d: digis)
  {
    result.insert( std::get<1>(d) );
  }
  return result;
}

std::set<int>
CSCDigiMatcher::wiregroupsInDetId(unsigned int detid)
{
  set<int> result;
  DigiContainer digis;
  digis = wireDigisInDetId(detid);
  for (auto& d: digis)
  {
    result.insert( std::get<1>(d) );
  }
  return result;
}
