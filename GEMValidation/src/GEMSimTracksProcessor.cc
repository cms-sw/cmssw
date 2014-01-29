/*
 * GEMSimTracksProcessor.cc
 *
 * Original Author:  "Vadim Khotilovich"
 */

#include "GEMCode/GEMValidation/interface/GEMSimTracksProcessor.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "iostream"

using namespace std;

GEMSimTracksProcessor::GEMSimTracksProcessor(const edm::ParameterSet& iConfig)
{
  minTrackPt_ = iConfig.getParameter<double>("minTrackPt");
}

void GEMSimTracksProcessor::init(const edm::EventSetup& iSetup)
{
  //Get the Magnetic field from the setup
  iSetup.get< IdealMagneticFieldRecord >().get(magfield_);

  // Get the propagators
  //iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyRK", propagator_);
  iSetup.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorAlong", propagator_);
  iSetup.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorOpposite", propagatorOpposite_);
}


void GEMSimTracksProcessor::fillTracks(const edm::SimTrackContainer &trks, const edm::SimVertexContainer &vtxs)
{
  trackids_.clear();
  trackid_to_trackextra_.clear();
  gemid_to_simhits_.clear();
  cscid_to_simhits_.clear();

  for (const auto & trk: trks)
  {
    if (trk.noVertex()) continue;
    addSimTrack(&trk, &((vtxs)[trk.vertIndex()]));
  }
}


void GEMSimTracksProcessor::addSimTrack(const SimTrack * trk, const SimVertex * vtx)
{
  if (std::abs(trk->type()) != 13) return; // only interested in direct muon simtracks
  if (trk->momentum().pt() < minTrackPt_) return; // TODO: make minpt configurable
  float eta = std::abs(trk->momentum().eta());
  if (eta > 2.2 || eta < 1.5) return; // no GEMs could be in such eta

  trackids_.push_back(trk->trackId());
  trackid_to_trackextra_[trk->trackId()] = SimTrackExtra(trk, vtx);
}


void GEMSimTracksProcessor::addSimHit(const PSimHit &hit, GlobalPoint &hit_gp)
{
  // first check if we know the SimTrack that this hit belongs to
  auto trk_itr = trackid_to_trackextra_.find(hit.trackId());
  if (trk_itr == trackid_to_trackextra_.end()) return;

  SimTrackExtra & trkx = trk_itr->second;

  DetId id(hit.detUnitId());
  if (id.subdetId() == MuonSubdetId::CSC)
  {
    CSCDetId cid(id);
    if (cid.iChamberType() != 2) return; // we only want ME1/b CSC simhits

    cscid_to_simhits_[hit.detUnitId()].push_back(hit_gp);
    trkx.csc_ids_.insert(hit.detUnitId());
  }
  else if (id.subdetId() == MuonSubdetId::GEM)
  {
    GEMDetId gid(id);
    if (!(gid.station() == 1 && gid.ring() == 1)) return; // only want GE1/1

    gemid_to_simhits_[hit.detUnitId()].push_back(hit_gp);
    trkx.gem_ids_.insert(hit.detUnitId());
  }
  // don't want RPC simhits
}


const SimTrack * GEMSimTracksProcessor::track(size_t itrk)
{
  if (itrk >= size()) return nullptr;
  
  unsigned int trkid = trackids_[itrk];
  SimTrackExtra & trkx = trackid_to_trackextra_[trkid];
  return trkx.trk_;
}


set< uint32_t > GEMSimTracksProcessor::getDetIdsGEM(size_t itrk, ChamberType odd_even)
{
  set< uint32_t > result;
  if (odd_even == NONE) return result; // protection from nonsense input

  unsigned int trkid = trackids_[itrk];
  SimTrackExtra & trkx = trackid_to_trackextra_[trkid];
  if (odd_even == BOTH) return trkx.gem_ids_;

  for (auto id : trkx.gem_ids_)
  {
    GEMDetId gid(id);
    if (gid.chamber() % 2 == 1 && odd_even == EVEN) continue;
    if (gid.chamber() % 2 == 0 && odd_even == ODD) continue;
    result.insert(id);
  }
  return result;
}


set< uint32_t > GEMSimTracksProcessor::getDetIdsCSC(size_t itrk, ChamberType odd_even)
{
  set< uint32_t > result;
  if (odd_even == NONE) return result; // protection from nonsense input

  unsigned int trkid = trackids_[itrk];
  SimTrackExtra & trkx = trackid_to_trackextra_[trkid];
  if (odd_even == BOTH) return trkx.csc_ids_;

  for (auto id : trkx.csc_ids_)
  {
    CSCDetId cid(id);
    if (cid.chamber() % 2 == 1 && odd_even == EVEN) continue;
    if (cid.chamber() % 2 == 0 && odd_even == ODD) continue;
    result.insert(id);
  }
  return result;
}


GEMSimTracksProcessor::ChamberType GEMSimTracksProcessor::chamberTypesHitGEM(size_t itrk, int layer)
{
  auto idset = getDetIdsGEM(itrk, BOTH);
  if (idset.empty()) return NONE;

  bool has_odd = false;
  bool has_even = false;
  for (auto id : idset)
  {
    GEMDetId gid(id);
    if (gid.layer() != layer) continue;
    if (gid.chamber() % 2)
      has_odd = true;
    else
      has_even = true;
  }
  if (has_even && has_odd) return BOTH;
  if (has_even) return EVEN;
  if (has_odd) return ODD;
  return NONE;
}


GEMSimTracksProcessor::ChamberType GEMSimTracksProcessor::chamberTypesHitCSC(size_t itrk)
{
  auto idset = getDetIdsCSC(itrk, BOTH);
  if (idset.empty()) return NONE;

  bool has_odd = false;
  bool has_even = false;
  for (auto id : idset)
  {
    CSCDetId cid(id);
    if (cid.chamber() % 2)
      has_odd = true;
    else
      has_even = true;
  }
  if (has_even && has_odd) return BOTH;
  if (has_even) return EVEN;
  if (has_odd) return ODD;
  return NONE;
}


GlobalPoint GEMSimTracksProcessor::meanSimHitsPositionGEM(size_t itrk, int layer, GEMSimTracksProcessor::ChamberType odd_even)
{
  auto idset = getDetIdsGEM(itrk, odd_even);
  if (idset.empty()) return GlobalPoint(); // point "zero"

  float sumx, sumy, sumz;
  sumx = sumy = sumz = 0.f;
  size_t n = 0;
  for (auto id : idset)
  {
    GEMDetId gid(id);
    if (gid.layer() != layer) continue;

    for (auto & gp : gemid_to_simhits_[id])
    {
      sumx += gp.x();
      sumy += gp.y();
      sumz += gp.z();
      ++n;
    }
  }
  if (n == 0) return GlobalPoint();
  float nn = static_cast< float >(n);
  return GlobalPoint(sumx/nn, sumy/nn, sumz/nn);
}


GlobalPoint GEMSimTracksProcessor::meanSimHitsPositionCSC(size_t itrk, GEMSimTracksProcessor::ChamberType odd_even)
{
  auto idset = getDetIdsCSC(itrk, odd_even);
  if (idset.empty()) return GlobalPoint(); // point "zero"

  float sumx, sumy, sumz;
  sumx = sumy = sumz = 0.f;
  size_t n = 0;
  for (auto id : idset)
  {
    for (auto & gp : cscid_to_simhits_[id])
    {
      sumx += gp.x();
      sumy += gp.y();
      sumz += gp.z();
      ++n;
    }
  }
  if (n == 0) return GlobalPoint();
  float nn = static_cast< float >(n);
  return GlobalPoint(sumx/nn, sumy/nn, sumz/nn);
}


GlobalPoint GEMSimTracksProcessor::propagatedPositionGEM(size_t itrk, int layer, ChamberType odd_even)
{
  GlobalPoint hits_gp = meanSimHitsPositionGEM(itrk, layer, odd_even);
  return propagateToZ(itrk, hits_gp.z());
}


GlobalPoint GEMSimTracksProcessor::propagatedPositionCSC(size_t itrk, ChamberType odd_even)
{
  GlobalPoint hits_gp = meanSimHitsPositionCSC(itrk, odd_even);
  return propagateToZ(itrk, hits_gp.z());
}


GlobalPoint GEMSimTracksProcessor::propagateToZ(size_t itrk, float z)
{
  unsigned int trkid = trackids_[itrk];
  SimTrackExtra & trkx = trackid_to_trackextra_[trkid];
  const SimTrack *trk = trkx.trk_;
  const SimVertex *vtx = trkx.vtx_;

  Plane::PositionType pos(0.f, 0.f, z);
  Plane::RotationType rot;
  Plane::PlanePointer my_plane = Plane::build(pos, rot);

  GlobalPoint inner_point(vtx->position().x(), vtx->position().y(), vtx->position().z());
  GlobalVector inner_vec (trk->momentum().x(), trk->momentum().y(), trk->momentum().z());

  FreeTrajectoryState state_start(inner_point, inner_vec, trk->charge(), &*magfield_);

  TrajectoryStateOnSurface tsos = propagator_->propagate(state_start, *my_plane);
  if (!tsos.isValid()) tsos = propagatorOpposite_->propagate(state_start, *my_plane);

  if (tsos.isValid()) return tsos.globalPosition();
  return GlobalPoint();
}
