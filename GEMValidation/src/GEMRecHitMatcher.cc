#include "GEMRecHitMatcher.h"
#include "SimHitMatcher.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

using namespace std;

GEMRecHitMatcher::GEMRecHitMatcher(SimHitMatcher& sh)
  : BaseMatcher(sh.trk(), sh.vtx(), sh.conf(), sh.event(), sh.eventSetup())
  , simhit_matcher_(&sh)

{
  gemRecHitInput_ = conf().getUntrackedParameter<edm::InputTag>("gemRecHitInput",
      edm::InputTag("gemRecHits"));

  minBXGEM_ = conf().getUntrackedParameter<int>("minBXGEM", -1);
  maxBXGEM_ = conf().getUntrackedParameter<int>("maxBXGEM", 1);

  matchDeltaStrip_ = conf().getUntrackedParameter<int>("matchDeltaStripGEM", 1);

  setVerbose(conf().getUntrackedParameter<int>("verboseGEMRecHit", 0));

  if (!(gemRecHitInput_.label().empty()))
  {
    init();
  }
}

GEMRecHitMatcher::~GEMRecHitMatcher() {}


void
GEMRecHitMatcher::init()
{
  edm::Handle<GEMRecHitCollection> gem_rechits;
  event().getByLabel(gemRecHitInput_, gem_rechits);
  matchRecHitsToSimTrack(*gem_rechits.product());

  edm::ESHandle<GEMGeometry> gem_g;
  eventSetup().get<MuonGeometryRecord>().get(gem_g);
  gem_geo_ = &*gem_g;
}


void
GEMRecHitMatcher::matchRecHitsToSimTrack(const GEMRecHitCollection& rechits)
{
  /*
  auto det_ids = simhit_matcher_->detIdsGEM();
  for (auto id: det_ids)
  {
    GEMDetId p_id(id);
    GEMDetId superch_id(p_id.region(), p_id.ring(), p_id.station(), 1, p_id.chamber(), 0);

    auto hit_strips = simhit_matcher_->hitStripsInDetId(id, matchDeltaStrip_);
    if (verbose())
    {
      cout<<"hit_strips_fat ";
      copy(hit_strips.begin(), hit_strips.end(), ostream_iterator<int>(cout, " "));
      cout<<endl;
    }

    auto digis_in_det = digis.get(GEMDetId(id));

    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d)
    {
      if (verbose()) cout<<"gdigi "<<p_id<<" "<<*d<<endl;
      // check that the digi is within BX range
      if (d->bx() < minBXGEM_ || d->bx() > maxBXGEM_) continue;
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_strips.find(d->strip()) == hit_strips.end()) continue;
      if (verbose()) cout<<"oki"<<endl;

      auto mydigi = make_digi(id, d->strip(), d->bx(), GEM_STRIP);
      detid_to_digis_[id].push_back(mydigi);
      chamber_to_digis_[ p_id.chamberId().rawId() ].push_back(mydigi);
      superchamber_to_digis_[ superch_id() ].push_back(mydigi);

      //int pad_num = 1 + static_cast<int>( roll->padOfStrip(d->strip()) ); // d->strip() is int
      //digi_map[ make_pair(pad_num, d->bx()) ].push_back( d->strip() );
    }
  }
  */
}


std::set<unsigned int>
GEMRecHitMatcher::detIds() const
{
  std::set<unsigned int> result;
  for (auto& p: detid_to_recHits_) result.insert(p.first);
  return result;
}


std::set<unsigned int>
GEMRecHitMatcher::chamberIds() const
{
  std::set<unsigned int> result;
  for (auto& p: chamber_to_recHits_) result.insert(p.first);
  return result;
}

std::set<unsigned int>
GEMRecHitMatcher::superChamberIds() const
{
  std::set<unsigned int> result;
  for (auto& p: superchamber_to_recHits_) result.insert(p.first);
  return result;
}


const GEMRecHitCollection&
GEMRecHitMatcher::recHitsInDetId(unsigned int detid) const
{
  if (detid_to_recHits_.find(detid) == detid_to_recHits_.end()) return no_recHits_;
  return detid_to_recHits_.at(detid);
}

const GEMRecHitCollection&
GEMRecHitMatcher::recHitsInChamber(unsigned int detid) const
{
  if (chamber_to_recHits_.find(detid) == chamber_to_recHits_.end()) return no_recHits_;
  return chamber_to_recHits_.at(detid);
}

const GEMRecHitCollection&
GEMRecHitMatcher::recHitsInSuperChamber(unsigned int detid) const
{
  if (superchamber_to_recHits_.find(detid) == superchamber_to_recHits_.end()) return no_recHits_;
  return superchamber_to_recHits_.at(detid);
}

int
GEMRecHitMatcher::nLayersWithRecHitsInSuperChamber(unsigned int detid) const
{
  set<int> layers;
  /*
  auto recHits = recHitsInSuperChamber(detid);
  for (auto& d: recHits)
  {
    GEMDetId idd(digi_id(d));
    layers.insert(idd.layer());
  }
  */
  return layers.size();
}


std::set<int>
GEMRecHitMatcher::stripNumbersInDetId(unsigned int detid) const
{
  set<int> result;
  /*
  auto recHits = recHitsInDetId(detid);
  for (auto& d: recHits)
  {
    result.insert( digi_channel(d) );
  }
  */
  return result;
}

std::set<int>
GEMRecHitMatcher::partitionNumbers() const
{
  std::set<int> result;

  auto detids = detIds();
  for (auto id: detids)
  {
    GEMDetId idd(id);
    result.insert( idd.roll() );
  }
  return result;
}

