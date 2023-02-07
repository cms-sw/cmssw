/* for High Granularity Calorimeter TestBeam
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include <cmath>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::Tr3D Tr3D;
typedef std::vector<float> ParmVec;

//#define EDM_ML_DEBUG

HGCalTBGeometry::HGCalTBGeometry(const HGCalTBTopology& topology_)
    : m_topology(topology_),
      m_validGeomIds(topology_.totalGeomModules()),
      m_det(topology_.detector()),
      m_subdet(topology_.subDetector()),
      twoBysqrt3_(2.0 / std::sqrt(3.0)) {
  m_cellVec = CellVec(topology_.totalGeomModules());
  m_validIds.reserve(m_topology.totalModules());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Expected total # of Geometry Modules " << m_topology.totalGeomModules();
#endif
}

HGCalTBGeometry::~HGCalTBGeometry() {}

void HGCalTBGeometry::fillNamedParams(DDFilteredView fv) {}

void HGCalTBGeometry::initializeParms() {}

void HGCalTBGeometry::localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int i, Pt3D& ref) {
  FlatHexagon::localCorners(lc, pv, ref);
}

void HGCalTBGeometry::newCell(
    const GlobalPoint& f1, const GlobalPoint& f2, const GlobalPoint& f3, const CCGFloat* parm, const DetId& detId) {
  DetId geomId = getGeometryDetId(detId);
  int cells(0);
  HGCalTBTopology::DecodedDetId id = m_topology.decode(detId);
  cells = m_topology.dddConstants().numberCellsHexagon(id.iSec1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "NewCell " << HGCalDetId(detId) << " GEOM " << HGCalDetId(geomId);
#endif
  const uint32_t cellIndex(m_topology.detId2denseGeomId(geomId));

  m_cellVec.at(cellIndex) = FlatHexagon(cornersMgr(), f1, f2, f3, parm);
  m_validGeomIds.at(cellIndex) = geomId;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Store for DetId " << std::hex << detId.rawId() << " GeomId " << geomId.rawId()
                                << std::dec << " Index " << cellIndex << " cells " << cells;
  unsigned int nOld = m_validIds.size();
#endif
  for (int cell = 0; cell < cells; ++cell) {
    id.iCell1 = cell;
    DetId idc = m_topology.encode(id);
    if (m_topology.valid(idc)) {
      m_validIds.emplace_back(idc);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Valid Id [" << cell << "] " << HGCalDetId(idc);
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeometry::newCell-> [" << cellIndex << "]"
                                << " front:" << f1.x() << '/' << f1.y() << '/' << f1.z() << " back:" << f2.x() << '/'
                                << f2.y() << '/' << f2.z() << " eta|phi " << m_cellVec[cellIndex].etaPos() << ":"
                                << m_cellVec[cellIndex].phiPos();
  unsigned int nNew = m_validIds.size();
  edm::LogVerbatim("HGCalGeom") << "ID: " << HGCalDetId(detId) << " with valid DetId from " << nOld << " to " << nNew;
#endif
}

std::shared_ptr<const CaloCellGeometry> HGCalTBGeometry::getGeometry(const DetId& detId) const {
  if (detId == DetId())
    return nullptr;  // nothing to get
  DetId geomId = getGeometryDetId(detId);
  const uint32_t cellIndex(m_topology.detId2denseGeomId(geomId));
  const GlobalPoint pos = (detId != geomId) ? getPosition(detId, false) : GlobalPoint();
  return cellGeomPtr(cellIndex, pos);
}

bool HGCalTBGeometry::present(const DetId& detId) const {
  if (detId == DetId())
    return false;
  DetId geomId = getGeometryDetId(detId);
  const uint32_t index(m_topology.detId2denseGeomId(geomId));
  return (nullptr != getGeometryRawPtr(index));
}

GlobalPoint HGCalTBGeometry::getPosition(const DetId& detid, bool debug) const {
  unsigned int cellIndex = indexFor(detid);
  GlobalPoint glob;
  unsigned int maxSize = m_cellVec.size();
  if (cellIndex < maxSize) {
    HGCalTBTopology::DecodedDetId id = m_topology.decode(detid);
    std::pair<float, float> xy;
    xy = m_topology.dddConstants().locateCellHex(id.iCell1, id.iSec1, true);
    const HepGeom::Point3D<float> lcoord(xy.first, xy.second, 0);
    glob = m_cellVec[cellIndex].getPosition(lcoord);
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "getPosition:: index " << cellIndex << " Local " << lcoord.x() << ":"
                                    << lcoord.y() << " ID " << id.iCell1 << ":" << id.iSec1 << " Global " << glob;
  }
  return glob;
}

GlobalPoint HGCalTBGeometry::getWaferPosition(const DetId& detid) const {
  unsigned int cellIndex = indexFor(detid);
  GlobalPoint glob;
  unsigned int maxSize = m_cellVec.size();
  if (cellIndex < maxSize) {
    const HepGeom::Point3D<float> lcoord(0, 0, 0);
    glob = m_cellVec[cellIndex].getPosition(lcoord);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "getWaferPosition:: ID " << std::hex << detid.rawId() << std::dec << " index "
                                  << cellIndex << " Global " << glob;
#endif
  }
  return glob;
}

double HGCalTBGeometry::getArea(const DetId& detid) const {
  HGCalTBGeometry::CornersVec corners = getNewCorners(detid);
  double area(0);
  if (corners.size() > 1) {
    int n = corners.size() - 1;
    int j = n - 1;
    for (int i = 0; i < n; ++i) {
      area += ((corners[j].x() + corners[i].x()) * (corners[i].y() - corners[j].y()));
      j = i;
    }
  }
  return std::abs(0.5 * area);
}

HGCalTBGeometry::CornersVec HGCalTBGeometry::getCorners(const DetId& detid) const {
  unsigned int ncorner = ((m_det == DetId::HGCalHSc) ? FlatTrd::ncorner_ : FlatHexagon::ncorner_);
  HGCalTBGeometry::CornersVec co(ncorner, GlobalPoint(0, 0, 0));
  unsigned int cellIndex = indexFor(detid);
  HGCalTBTopology::DecodedDetId id = m_topology.decode(detid);
  if (cellIndex < m_cellVec.size()) {
    std::pair<float, float> xy;
    xy = m_topology.dddConstants().locateCellHex(id.iCell1, id.iSec1, true);
    float dx = m_cellVec[cellIndex].param()[FlatHexagon::k_r];
    float dy = k_half * m_cellVec[cellIndex].param()[FlatHexagon::k_R];
    float dz = m_cellVec[cellIndex].param()[FlatHexagon::k_dZ];
    static const int signx[] = {0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1};
    static const int signy[] = {-2, -1, 1, 2, 1, -1, -2, -1, 1, 2, 1, -1};
    static const int signz[] = {-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1};
    for (unsigned int i = 0; i < ncorner; ++i) {
      const HepGeom::Point3D<float> lcoord(xy.first + signx[i] * dx, xy.second + signy[i] * dy, signz[i] * dz);
      co[i] = m_cellVec[cellIndex].getPosition(lcoord);
    }
  }
  return co;
}

HGCalTBGeometry::CornersVec HGCalTBGeometry::get8Corners(const DetId& detid) const {
  unsigned int ncorner = FlatTrd::ncorner_;
  HGCalTBGeometry::CornersVec co(ncorner, GlobalPoint(0, 0, 0));
  unsigned int cellIndex = indexFor(detid);
  HGCalTBTopology::DecodedDetId id = m_topology.decode(detid);
  if (cellIndex < m_cellVec.size()) {
    std::pair<float, float> xy;
    float dx(0);
    static const int signx[] = {-1, -1, 1, 1, -1, -1, 1, 1};
    static const int signy[] = {-1, 1, 1, -1, -1, 1, 1, -1};
    static const int signz[] = {-1, -1, -1, -1, 1, 1, 1, 1};
    xy = m_topology.dddConstants().locateCellHex(id.iCell1, id.iSec1, true);
    dx = m_cellVec[cellIndex].param()[FlatHexagon::k_r];
    float dz = m_cellVec[cellIndex].param()[FlatHexagon::k_dZ];
    for (unsigned int i = 0; i < ncorner; ++i) {
      const HepGeom::Point3D<float> lcoord(xy.first + signx[i] * dx, xy.second + signy[i] * dx, signz[i] * dz);
      co[i] = m_cellVec[cellIndex].getPosition(lcoord);
    }
  }
  return co;
}

HGCalTBGeometry::CornersVec HGCalTBGeometry::getNewCorners(const DetId& detid, bool debug) const {
  unsigned int ncorner = (m_det == DetId::HGCalHSc) ? 5 : 7;
  HGCalTBGeometry::CornersVec co(ncorner, GlobalPoint(0, 0, 0));
  unsigned int cellIndex = indexFor(detid);
  HGCalTBTopology::DecodedDetId id = m_topology.decode(detid);
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "NewCorners for Layer " << id.iLay << " Wafer " << id.iSec1 << ":" << id.iSec2
                                  << " Cell " << id.iCell1 << ":" << id.iCell2;
  if (cellIndex < m_cellVec.size()) {
    std::pair<float, float> xy;
    float dx = k_fac2 * m_cellVec[cellIndex].param()[FlatHexagon::k_r];
    float dy = k_fac1 * m_cellVec[cellIndex].param()[FlatHexagon::k_R];
    float dz = -id.zSide * m_cellVec[cellIndex].param()[FlatHexagon::k_dZ];
    static const int signx[] = {1, -1, -2, -1, 1, 2};
    static const int signy[] = {1, 1, 0, -1, -1, 0};
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "kfac " << k_fac1 << ":" << k_fac2 << " dx:dy:dz " << dx << ":" << dy << ":"
                                    << dz;
#endif
    xy = m_topology.dddConstants().locateCellHex(id.iCell1, id.iSec1, true);
    for (unsigned int i = 0; i < ncorner - 1; ++i) {
      const HepGeom::Point3D<float> lcoord(xy.first + signx[i] * dx, xy.second + signy[i] * dy, dz);
      co[i] = m_cellVec[cellIndex].getPosition(lcoord);
    }
    // Used to pass downstream the thickness of this cell
    co[ncorner - 1] = GlobalPoint(0, 0, -2 * dz);
  }
  return co;
}

DetId HGCalTBGeometry::neighborZ(const DetId& idin, const GlobalVector& momentum) const {
  DetId idnew;
  HGCalTBTopology::DecodedDetId id = m_topology.decode(idin);
  int lay = ((momentum.z() * id.zSide > 0) ? (id.iLay + 1) : (id.iLay - 1));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "neighborz1:: ID " << id.iLay << ":" << id.iSec1 << ":" << id.iSec2 << ":"
                                << id.iCell1 << ":" << id.iCell2 << " New Layer " << lay << " Range "
                                << m_topology.dddConstants().firstLayer() << ":"
                                << m_topology.dddConstants().lastLayer(true) << " pz " << momentum.z();
#endif
  if ((lay >= m_topology.dddConstants().firstLayer()) && (lay <= m_topology.dddConstants().lastLayer(true)) &&
      (momentum.z() != 0.0)) {
    GlobalPoint v = getPosition(idin, false);
    double z = id.zSide * m_topology.dddConstants().waferZ(lay, true);
    double grad = (z - v.z()) / momentum.z();
    GlobalPoint p(v.x() + grad * momentum.x(), v.y() + grad * momentum.y(), z);
    double r = p.perp();
    auto rlimit = topology().dddConstants().rangeR(z, true);
    if (r >= rlimit.first && r <= rlimit.second)
      idnew = getClosestCell(p);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "neighborz1:: Position " << v << " New Z " << z << ":" << grad << " new position "
                                  << p << " r-limit " << rlimit.first << ":" << rlimit.second;
#endif
  }
  return idnew;
}

DetId HGCalTBGeometry::neighborZ(const DetId& idin,
                                 const MagneticField* bField,
                                 int charge,
                                 const GlobalVector& momentum) const {
  DetId idnew;
  HGCalTBTopology::DecodedDetId id = m_topology.decode(idin);
  int lay = ((momentum.z() * id.zSide > 0) ? (id.iLay + 1) : (id.iLay - 1));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "neighborz2:: ID " << id.iLay << ":" << id.iSec1 << ":" << id.iSec2 << ":"
                                << id.iCell1 << ":" << id.iCell2 << " New Layer " << lay << " Range "
                                << m_topology.dddConstants().firstLayer() << ":"
                                << m_topology.dddConstants().lastLayer(true) << " pz " << momentum.z();
#endif
  if ((lay >= m_topology.dddConstants().firstLayer()) && (lay <= m_topology.dddConstants().lastLayer(true)) &&
      (momentum.z() != 0.0)) {
    GlobalPoint v = getPosition(idin, false);
    double z = id.zSide * m_topology.dddConstants().waferZ(lay, true);
    FreeTrajectoryState fts(v, momentum, charge, bField);
    Plane::PlanePointer nPlane = Plane::build(Plane::PositionType(0, 0, z), Plane::RotationType());
    AnalyticalPropagator myAP(bField, alongMomentum, 2 * M_PI);
    TrajectoryStateOnSurface tsos = myAP.propagate(fts, *nPlane);
    GlobalPoint p;
    auto rlimit = topology().dddConstants().rangeR(z, true);
    if (tsos.isValid()) {
      p = tsos.globalPosition();
      double r = p.perp();
      if (r >= rlimit.first && r <= rlimit.second)
        idnew = getClosestCell(p);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "neighborz2:: Position " << v << " New Z " << z << ":" << charge << ":"
                                  << tsos.isValid() << " new position " << p << " r limits " << rlimit.first << ":"
                                  << rlimit.second;
#endif
  }
  return idnew;
}

DetId HGCalTBGeometry::getClosestCell(const GlobalPoint& r) const {
  unsigned int cellIndex = getClosestCellIndex(r);
  if (cellIndex < m_cellVec.size()) {
    HGCalTBTopology::DecodedDetId id = m_topology.decode(m_validGeomIds[cellIndex]);
    if (id.det == 0)
      id.det = static_cast<int>(m_topology.detector());
    HepGeom::Point3D<float> local;
    if (r.z() > 0) {
      local = HepGeom::Point3D<float>(r.x(), r.y(), 0);
      id.zSide = 1;
    } else {
      local = HepGeom::Point3D<float>(-r.x(), r.y(), 0);
      id.zSide = -1;
    }
    const auto& kxy = m_topology.dddConstants().assignCell(local.x(), local.y(), id.iLay, id.iType, true);
    id.iCell1 = kxy.second;
    id.iSec1 = kxy.first;
    id.iType = m_topology.dddConstants().waferTypeT(kxy.first);
    if (id.iType != 1)
      id.iType = -1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "getClosestCell: local " << local << " Id " << id.det << ":" << id.zSide << ":"
                                  << id.iLay << ":" << id.iSec1 << ":" << id.iSec2 << ":" << id.iType << ":"
                                  << id.iCell1 << ":" << id.iCell2;
#endif
    //check if returned cell is valid
    if (id.iCell1 >= 0)
      return m_topology.encode(id);
  }

  //if not valid or out of bounds return a null DetId
  return DetId();
}

HGCalTBGeometry::DetIdSet HGCalTBGeometry::getCells(const GlobalPoint& r, double dR) const {
  HGCalTBGeometry::DetIdSet dss;
  return dss;
}

std::string HGCalTBGeometry::cellElement() const {
  if (m_subdet == HGCEE)
    return "HGCalEE";
  else if (m_subdet == HGCHEF)
    return "HGCalHEFront";
  else
    return "Unknown";
}

unsigned int HGCalTBGeometry::indexFor(const DetId& detId) const {
  unsigned int cellIndex = m_cellVec.size();
  if (detId != DetId()) {
    DetId geomId = getGeometryDetId(detId);
    cellIndex = m_topology.detId2denseGeomId(geomId);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "indexFor " << std::hex << detId.rawId() << ":" << geomId.rawId() << std::dec
                                  << " index " << cellIndex;
#endif
  }
  return cellIndex;
}

unsigned int HGCalTBGeometry::sizeForDenseIndex() const { return m_topology.totalGeomModules(); }

const CaloCellGeometry* HGCalTBGeometry::getGeometryRawPtr(uint32_t index) const {
  // Modify the RawPtr class
  if (m_cellVec.size() < index)
    return nullptr;
  const CaloCellGeometry* cell(&m_cellVec[index]);
  return (nullptr == cell->param() ? nullptr : cell);
}

std::shared_ptr<const CaloCellGeometry> HGCalTBGeometry::cellGeomPtr(uint32_t index) const {
  if (index >= m_cellVec.size())
    return nullptr;
  static const auto do_not_delete = [](const void*) {};
  auto cell = std::shared_ptr<const CaloCellGeometry>(&m_cellVec[index], do_not_delete);
  if (nullptr == cell->param())
    return nullptr;
  return cell;
}

std::shared_ptr<const CaloCellGeometry> HGCalTBGeometry::cellGeomPtr(uint32_t index, const GlobalPoint& pos) const {
  if ((index >= m_cellVec.size()) || (m_validGeomIds[index].rawId() == 0))
    return nullptr;
  if (pos == GlobalPoint())
    return cellGeomPtr(index);
  auto cell = std::make_shared<FlatHexagon>(m_cellVec[index]);
  cell->setPosition(pos);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "cellGeomPtr " << index << ":" << cell;
#endif
  if (nullptr == cell->param())
    return nullptr;
  return cell;
}

void HGCalTBGeometry::addValidID(const DetId& id) {
  edm::LogError("HGCalGeom") << "HGCalTBGeometry::addValidID is not implemented";
}

unsigned int HGCalTBGeometry::getClosestCellIndex(const GlobalPoint& r) const {
  return (getClosestCellIndex(r, m_cellVec));
}

template <class T>
unsigned int HGCalTBGeometry::getClosestCellIndex(const GlobalPoint& r, const std::vector<T>& vec) const {
  float phip = r.phi();
  float zp = r.z();
  float dzmin(9999), dphimin(9999), dphi10(0.175);
  unsigned int cellIndex = vec.size();
  for (unsigned int k = 0; k < vec.size(); ++k) {
    float dphi = phip - vec[k].phiPos();
    while (dphi > M_PI)
      dphi -= 2 * M_PI;
    while (dphi <= -M_PI)
      dphi += 2 * M_PI;
    if (std::abs(dphi) < dphi10) {
      float dz = std::abs(zp - vec[k].getPosition().z());
      if (dz < (dzmin + 0.001)) {
        dzmin = dz;
        if (std::abs(dphi) < (dphimin + 0.01)) {
          cellIndex = k;
          dphimin = std::abs(dphi);
        } else {
          if (cellIndex >= vec.size())
            cellIndex = k;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "getClosestCellIndex::Input " << zp << ":" << phip << " Index " << cellIndex;
  if (cellIndex < vec.size())
    edm::LogVerbatim("HGCalGeom") << " Cell z " << vec[cellIndex].getPosition().z() << ":" << dzmin << " phi "
                                  << vec[cellIndex].phiPos() << ":" << dphimin;
#endif
  return cellIndex;
}

// FIXME: Change sorting algorithm if needed
namespace {
  struct rawIdSort {
    bool operator()(const DetId& a, const DetId& b) { return (a.rawId() < b.rawId()); }
  };
}  // namespace

void HGCalTBGeometry::sortDetIds(void) {
  m_validIds.shrink_to_fit();
  std::sort(m_validIds.begin(), m_validIds.end(), rawIdSort());
}

void HGCalTBGeometry::getSummary(CaloSubdetectorGeometry::TrVec& trVector,
                                 CaloSubdetectorGeometry::IVec& iVector,
                                 CaloSubdetectorGeometry::DimVec& dimVector,
                                 CaloSubdetectorGeometry::IVec& dinsVector) const {
  unsigned int numberOfCells = m_topology.totalGeomModules();  // total Geom Modules both sides
  unsigned int numberOfShapes = k_NumberOfShapes;
  unsigned int numberOfParametersPerShape = ((m_det == DetId::HGCalHSc) ? (unsigned int)(k_NumberOfParametersPerTrd)
                                                                        : (unsigned int)(k_NumberOfParametersPerHex));

  trVector.reserve(numberOfCells * numberOfTransformParms());
  iVector.reserve(numberOfCells);
  dimVector.reserve(numberOfShapes * numberOfParametersPerShape);
  dinsVector.reserve(numberOfCells);

  for (unsigned itr = 0; itr < m_topology.dddConstants().getTrFormN(); ++itr) {
    HGCalTBParameters::hgtrform mytr = m_topology.dddConstants().getTrForm(itr);
    int layer = mytr.lay;

    for (int wafer = 0; wafer < m_topology.dddConstants().sectors(); ++wafer) {
      if (m_topology.dddConstants().waferInLayer(wafer, layer, true)) {
        HGCalTBParameters::hgtrap vol = m_topology.dddConstants().getModule(wafer, true, true);
        ParmVec params(numberOfParametersPerShape, 0);
        params[FlatHexagon::k_dZ] = vol.dz;
        params[FlatHexagon::k_r] = vol.cellSize;
        params[FlatHexagon::k_R] = twoBysqrt3_ * params[FlatHexagon::k_r];
        dimVector.insert(dimVector.end(), params.begin(), params.end());
      }
    }
  }

  for (unsigned int i(0); i < numberOfCells; ++i) {
    DetId detId = m_validGeomIds[i];
    int layer = HGCalDetId(detId).layer();
    dinsVector.emplace_back(m_topology.detId2denseGeomId(detId));
    iVector.emplace_back(layer);

    Tr3D tr;
    auto ptr = cellGeomPtr(i);
    if (nullptr != ptr) {
      ptr->getTransform(tr, (Pt3DVec*)nullptr);

      if (Tr3D() == tr) {  // there is no rotation
        const GlobalPoint& gp(ptr->getPosition());
        tr = HepGeom::Translate3D(gp.x(), gp.y(), gp.z());
      }

      const CLHEP::Hep3Vector tt(tr.getTranslation());
      trVector.emplace_back(tt.x());
      trVector.emplace_back(tt.y());
      trVector.emplace_back(tt.z());
      if (6 == numberOfTransformParms()) {
        const CLHEP::HepRotation rr(tr.getRotation());
        const ROOT::Math::Transform3D rtr(
            rr.xx(), rr.xy(), rr.xz(), tt.x(), rr.yx(), rr.yy(), rr.yz(), tt.y(), rr.zx(), rr.zy(), rr.zz(), tt.z());
        ROOT::Math::EulerAngles ea;
        rtr.GetRotation(ea);
        trVector.emplace_back(ea.Phi());
        trVector.emplace_back(ea.Theta());
        trVector.emplace_back(ea.Psi());
      }
    }
  }
}

DetId HGCalTBGeometry::getGeometryDetId(DetId detId) const {
  return static_cast<DetId>(HGCalDetId(detId).geometryCell());
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTBGeometry);
