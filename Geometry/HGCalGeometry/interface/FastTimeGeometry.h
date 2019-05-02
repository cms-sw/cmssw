#ifndef GeometryHGCalGeometryFastTimeGeometry_h
#define GeometryHGCalGeometryFastTimeGeometry_h

/*
 * Geometry for Fast Timing Layer
 * This geometry is essentially driven by topology, 
 * which is thus encapsulated in this class. 
 * This makes this geometry not suitable to be loaded
 * by regular CaloGeometryLoader<T>
 */

#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "Geometry/Records/interface/FastTimeGeometryRecord.h"
#include <vector>

class FlatTrd;

class FastTimeGeometry final : public CaloSubdetectorGeometry {
public:
  typedef std::vector<FlatTrd> CellVec;

  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;

  typedef std::set<DetId> DetIdSet;
  typedef std::vector<GlobalPoint> CornersVec;

  typedef FastTimeGeometryRecord AlignedRecord;  // NOTE: not aligned yet
  typedef PFastTimeRcd PGeometryRecord;

  enum { k_NumberOfParametersPerShape = 12 };  // FlatTrd
  enum { k_NumberOfShapes = 50 };

  static std::string dbString() { return "PFastTimeRcd"; }

  FastTimeGeometry(const FastTimeTopology& topology);

  ~FastTimeGeometry() override;

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

  GlobalPoint getPosition(const DetId& id) const;

  /// Returns the corner points of this cell's volume.
  CornersVec getCorners(const DetId& id) const;

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

  static std::string producerTag() { return "FastTime"; }
  std::string cellElement() const;

  const FastTimeTopology& topology() const { return m_topology; }
  void sortDetIds();

protected:
  unsigned int indexFor(const DetId& id) const override;
  using CaloSubdetectorGeometry::sizeForDenseIndex;
  unsigned int sizeForDenseIndex() const;

  // Modify the RawPtr class
  const CaloCellGeometry* getGeometryRawPtr(uint32_t index) const override;
  std::shared_ptr<const CaloCellGeometry> cellGeomPtr(uint32_t index) const override;

  void addValidID(const DetId& id);

private:
  std::shared_ptr<const CaloCellGeometry> cellGeomPtr(uint32_t index, const GlobalPoint& p) const;

  const FastTimeTopology& m_topology;

  CellVec m_cellVec;
  std::vector<DetId> m_validGeomIds;
  int m_Type;
  ForwardSubdetector m_subdet;
};

#endif
