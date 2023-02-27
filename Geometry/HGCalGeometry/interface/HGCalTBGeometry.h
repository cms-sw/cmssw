#ifndef GeometryHGCalGeometryHGCalTBGeometry_h
#define GeometryHGCalGeometryHGCalTBGeometry_h

/*
 * Geometry for High Granularity Calorimeter TestBeam
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */

#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatHexagon.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HGCalTBTopology.h"
#include "Geometry/Records/interface/HGCalGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <vector>

class HGCalTBGeometry final : public CaloSubdetectorGeometry {
public:
  typedef std::vector<FlatHexagon> CellVec;
  typedef std::vector<FlatTrd> CellVec2;
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;

  typedef std::set<DetId> DetIdSet;
  typedef std::vector<GlobalPoint> CornersVec;

  typedef HGCalGeometryRecord AlignedRecord;  // NOTE: not aligned yet
  typedef PHGCalRcd PGeometryRecord;

  static constexpr unsigned int k_NumberOfParametersPerTrd = 12;   // FlatTrd
  static constexpr unsigned int k_NumberOfParametersPerHex = 3;    // FlatHexagon
  static constexpr unsigned int k_NumberOfParametersPerShape = 3;  // FlatHexagon
  static constexpr unsigned int k_NumberOfShapes = 100;
  static constexpr unsigned int k_NumberOfShapesTrd = 1000;

  static std::string dbString() { return "PHGCalTBRcd"; }

  HGCalTBGeometry(const HGCalTBTopology& topology);

  ~HGCalTBGeometry() override;

  void localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int i, Pt3D& ref);

  void newCell(const GlobalPoint& f1,
               const GlobalPoint& f2,
               const GlobalPoint& f3,
               const CCGFloat* parm,
               const DetId& detId) override;

  /// Get the cell geometry of a given detector id.  Should return false if not found.
  std::shared_ptr<const CaloCellGeometry> getGeometry(const DetId& id) const override;

  bool present(const DetId& id) const override;

  void getSummary(CaloSubdetectorGeometry::TrVec& trVector,
                  CaloSubdetectorGeometry::IVec& iVector,
                  CaloSubdetectorGeometry::DimVec& dimVector,
                  CaloSubdetectorGeometry::IVec& dinsVector) const override;

  GlobalPoint getPosition(const DetId& id, bool debug = false) const;
  GlobalPoint getWaferPosition(const DetId& id) const;

  /// Returns area of a cell
  double getArea(const DetId& detid) const;

  /// Returns the corner points of this cell's volume.
  CornersVec getCorners(const DetId& id) const;
  CornersVec get8Corners(const DetId& id) const;
  CornersVec getNewCorners(const DetId& id, bool debug = false) const;

  // Get neighbor in z along a direction
  DetId neighborZ(const DetId& idin, const GlobalVector& p) const;
  DetId neighborZ(const DetId& idin, const MagneticField* bField, int charge, const GlobalVector& momentum) const;

  // avoid sorting set in base class
  const std::vector<DetId>& getValidDetIds(DetId::Detector det = DetId::Detector(0), int subdet = 0) const override {
    return m_validIds;
  }
  const std::vector<DetId>& getValidGeomDetIds(void) const { return m_validGeomIds; }

  // Get closest cell, etc...
  DetId getClosestCell(const GlobalPoint& r) const override;

  /** \brief Get a list of all cells within a dR of the given cell
      
      The default implementation makes a loop over all cell geometries.
      Cleverer implementations are suggested to use rough conversions between
      eta/phi and ieta/iphi and test on the boundaries.
  */
  DetIdSet getCells(const GlobalPoint& r, double dR) const override;

  virtual void fillNamedParams(DDFilteredView fv);
  void initializeParms() override;

  static std::string producerTag() { return "HGCalTB"; }
  std::string cellElement() const;

  const HGCalTBTopology& topology() const { return m_topology; }
  void sortDetIds();

protected:
  unsigned int indexFor(const DetId& id) const override;
  using CaloSubdetectorGeometry::sizeForDenseIndex;
  unsigned int sizeForDenseIndex() const;

  // Modify the RawPtr class
  const CaloCellGeometry* getGeometryRawPtr(uint32_t index) const override;

  std::shared_ptr<const CaloCellGeometry> cellGeomPtr(uint32_t index) const override;

  void addValidID(const DetId& id);
  unsigned int getClosestCellIndex(const GlobalPoint& r) const;

private:
  template <class T>
  unsigned int getClosestCellIndex(const GlobalPoint& r, const std::vector<T>& vec) const;
  std::shared_ptr<const CaloCellGeometry> cellGeomPtr(uint32_t index, const GlobalPoint& p) const;
  DetId getGeometryDetId(DetId detId) const;

  static constexpr double k_half = 0.5;
  static constexpr double k_fac1 = 0.5;
  static constexpr double k_fac2 = 1.0 / 3.0;

  const HGCalTBTopology& m_topology;
  CellVec m_cellVec;
  std::vector<DetId> m_validGeomIds;
  DetId::Detector m_det;
  ForwardSubdetector m_subdet;
  const double twoBysqrt3_;
};

#endif
