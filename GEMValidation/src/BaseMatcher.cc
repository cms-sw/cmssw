#include "GEMCode/GEMValidation/src/BaseMatcher.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"


BaseMatcher::BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es)
: trk_(t), vtx_(v), conf_(ps), ev_(ev), es_(es), verbose_(0)
{
  // list of CSC chamber type numbers to use
  std::vector<int> csc_types = conf().getUntrackedParameter<std::vector<int> >("useCSCChamberTypes", std::vector<int>() );
  for (int i=0; i <= CSC_ME42; ++i) useCSCChamberTypes_[i] = false;
  for (auto t: csc_types)
  {
    if (t >= 0 && t <= CSC_ME42) useCSCChamberTypes_[t] = 1;
  }
  // empty list means use all the chamber types
  if (csc_types.empty()) useCSCChamberTypes_[CSC_ALL] = 1;

  // Get the magnetic field
  es.get< IdealMagneticFieldRecord >().get(magfield_);

  // Get the propagators                                                                                  
  es.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorAlong", propagator_);
  es.get< TrackingComponentsRecord >().get("SteppingHelixPropagatorOpposite", propagatorOpposite_);

  /// get the geometry
  hasGEMGeometry_ = false;
  hasRPCGeometry_ = false;
  hasCSCGeometry_ = false;
  hasME0Geometry_ = false;

  try {
    es.get<MuonGeometryRecord>().get(gem_geom);
    gemGeometry_ = &*gem_geom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    hasGEMGeometry_ = false;
    LogDebug("MuonSimHitAnalyzer") << "+++ Info: GEM geometry is unavailable. +++\n";
  }

  try {
    es.get<MuonGeometryRecord>().get(me0_geom);
    me0Geometry_ = &*me0_geom;
  } catch (edm::eventsetup::NoProxyException<ME0Geometry>& e) {
    hasME0Geometry_ = false;
    LogDebug("MuonSimHitAnalyzer") << "+++ Info: ME0 geometry is unavailable. +++\n";
  }

  try {
    es.get<MuonGeometryRecord>().get(csc_geom);
    cscGeometry_ = &*csc_geom;
  } catch (edm::eventsetup::NoProxyException<CSCGeometry>& e) {
    hasCSCGeometry_ = false;
    LogDebug("MuonSimHitAnalyzer") << "+++ Info: CSC geometry is unavailable. +++\n";
  }

  try {
    es.get<MuonGeometryRecord>().get(rpc_geom);
    rpcGeometry_ = &*rpc_geom;
  } catch (edm::eventsetup::NoProxyException<RPCGeometry>& e) {
    hasRPCGeometry_ = false;
    LogDebug("MuonSimHitAnalyzer") << "+++ Info: RPC geometry is unavailable. +++\n";
  }
}


BaseMatcher::~BaseMatcher()
{
}


bool BaseMatcher::useCSCChamberType(int csc_type)
{
  if (csc_type < 0 || csc_type > CSC_ME42) return false;
  return useCSCChamberTypes_[csc_type];
}


GlobalPoint
BaseMatcher::propagateToZ(GlobalPoint &inner_point, GlobalVector &inner_vec, float z) const
{
  Plane::PositionType pos(0.f, 0.f, z);
  Plane::RotationType rot;
  Plane::PlanePointer my_plane(Plane::build(pos, rot));

  FreeTrajectoryState state_start(inner_point, inner_vec, trk_.charge(), &*magfield_);

  TrajectoryStateOnSurface tsos(propagator_->propagate(state_start, *my_plane));
  if (!tsos.isValid()) tsos = propagatorOpposite_->propagate(state_start, *my_plane);

  if (tsos.isValid()) return tsos.globalPosition();
  return GlobalPoint();
}


GlobalPoint
BaseMatcher::propagateToZ(float z) const
{
  GlobalPoint inner_point(vtx_.position().x(), vtx_.position().y(), vtx_.position().z());
  GlobalVector inner_vec (trk_.momentum().x(), trk_.momentum().y(), trk_.momentum().z());
  return propagateToZ(inner_point, inner_vec, z);
}

GlobalPoint
BaseMatcher::propagatedPositionGEM() const
{
  const double eta(trk().momentum().eta());
  const int endcap( (eta > 0.) ? 1 : -1);
  return propagateToZ(endcap*AVERAGE_GEM_Z);
}
