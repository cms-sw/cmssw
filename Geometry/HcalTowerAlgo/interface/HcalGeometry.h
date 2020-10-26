#ifndef HcalGeometry_h
#define HcalGeometry_h

#include "DataFormats/Common/interface/AtomicPtrCache.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"

class HcalFlexiHardcodeGeometryLoader;
class HcalHardcodeGeometryLoader;

class HcalGeometry : public CaloSubdetectorGeometry {
  friend class HcalGeometryPlan1Tester;

public:
  friend class HcalFlexiHardcodeGeometryLoader;
  friend class HcalHardcodeGeometryLoader;

  typedef std::vector<IdealObliquePrism> HBCellVec;
  typedef std::vector<IdealObliquePrism> HECellVec;
  typedef std::vector<IdealObliquePrism> HOCellVec;
  typedef std::vector<IdealZPrism> HFCellVec;

  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;

  typedef HcalAlignmentRcd AlignmentRecord;
  typedef HcalGeometryRecord AlignedRecord;
  typedef PHcalRcd PGeometryRecord;
  typedef HcalDetId DetIdType;

  enum { k_NumberOfParametersPerShape = 5 };

  static std::string dbString() { return "PHcalRcd"; }

  unsigned int numberOfShapes() const override { return m_topology.getNumberOfShapes(); }
  unsigned int numberOfParametersPerShape() const override { return k_NumberOfParametersPerShape; }

  explicit HcalGeometry(const HcalTopology& topology);

  /// The HcalGeometry will delete all its cell geometries at destruction time
  ~HcalGeometry() override;

  const std::vector<DetId>& getValidDetIds(DetId::Detector det = DetId::Detector(0), int subdet = 0) const override;

  std::shared_ptr<const CaloCellGeometry> getGeometry(const DetId& id) const override;

  DetId getClosestCell(const GlobalPoint& r) const override;
  DetId getClosestCell(const GlobalPoint& r, bool ignoreCorrect) const;

  CaloSubdetectorGeometry::DetIdSet getCells(const GlobalPoint& r, double dR) const override;

  GlobalPoint getPosition(const DetId& id) const;
  GlobalPoint getPosition(uint32_t id, bool type) const;
  GlobalPoint getBackPosition(const DetId& id) const;
  GlobalPoint getBackPosition(uint32_t id, bool type) const;
  CaloCellGeometry::CornersVec getCorners(const DetId& id) const;

  static std::string producerTag() { return "HCAL"; }

  static unsigned int numberOfBarrelAlignments() { return 36; }

  static unsigned int numberOfEndcapAlignments() { return 36; }

  static unsigned int numberOfForwardAlignments() { return 36; }

  static unsigned int numberOfOuterAlignments() { return 60; }

  unsigned int getHxSize(const int type) const;

  static unsigned int numberOfAlignments() {
    return (numberOfBarrelAlignments() + numberOfEndcapAlignments() + numberOfOuterAlignments() +
            numberOfForwardAlignments());
  }

  static unsigned int alignmentBarrelIndexLocal(const DetId& id);
  static unsigned int alignmentEndcapIndexLocal(const DetId& id);
  static unsigned int alignmentForwardIndexLocal(const DetId& id);
  static unsigned int alignmentOuterIndexLocal(const DetId& id);
  static unsigned int alignmentTransformIndexLocal(const DetId& id);

  static unsigned int alignmentBarEndForIndexLocal(const DetId& id, unsigned int nD);

  static unsigned int alignmentTransformIndexGlobal(const DetId& id);

  static DetId detIdFromLocalAlignmentIndex(unsigned int i);
  static DetId detIdFromBarrelAlignmentIndex(unsigned int i);
  static DetId detIdFromEndcapAlignmentIndex(unsigned int i);
  static DetId detIdFromForwardAlignmentIndex(unsigned int i);
  static DetId detIdFromOuterAlignmentIndex(unsigned int i);

  void localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int i, Pt3D& ref);

  void newCell(const GlobalPoint& f1,
               const GlobalPoint& f2,
               const GlobalPoint& f3,
               const CCGFloat* parm,
               const DetId& detId) override;

  void getSummary(CaloSubdetectorGeometry::TrVec& trVector,
                  CaloSubdetectorGeometry::IVec& iVector,
                  CaloSubdetectorGeometry::DimVec& dimVector,
                  CaloSubdetectorGeometry::IVec& dinsVector) const override;

  const HcalTopology& topology() const { return m_topology; }

protected:
  unsigned int indexFor(const DetId& id) const override { return m_topology.detId2denseId(id); }
  unsigned int sizeForDenseIndex(const DetId& id) const override { return m_topology.ncells(); }

  // Modify the RawPtr class
  const CaloCellGeometry* getGeometryRawPtr(uint32_t index) const override;

private:
  // Base clas for getting geometry
  std::shared_ptr<const CaloCellGeometry> getGeometryBase(const DetId& id) const {
    return cellGeomPtr(m_topology.detId2denseId(id));
  }

  //returns din
  unsigned int newCellImpl(
      const GlobalPoint& f1, const GlobalPoint& f2, const GlobalPoint& f3, const CCGFloat* parm, const DetId& detId);

  //can only be used by friend classes, to ensure sorting is done at the end
  void newCellFast(
      const GlobalPoint& f1, const GlobalPoint& f2, const GlobalPoint& f3, const CCGFloat* parm, const DetId& detId);

  void increaseReserve(unsigned int extra);
  void sortValidIds();

  void fillDetIds() const;

  void init();

  /// helper methods for getClosestCell
  int etaRing(HcalSubdetector bc, double abseta) const;
  int phiBin(HcalSubdetector bc, int etaring, double phi) const;
  DetId correctId(const DetId& id) const;

  const HcalTopology& m_topology;
  bool m_mergePosition;

  mutable edm::AtomicPtrCache<std::vector<DetId>> m_hbIds;
  mutable edm::AtomicPtrCache<std::vector<DetId>> m_heIds;
  mutable edm::AtomicPtrCache<std::vector<DetId>> m_hoIds;
  mutable edm::AtomicPtrCache<std::vector<DetId>> m_hfIds;
  mutable edm::AtomicPtrCache<std::vector<DetId>> m_emptyIds;
  CaloSubdetectorGeometry::IVec m_dins;

  HBCellVec m_hbCellVec;
  HECellVec m_heCellVec;
  HOCellVec m_hoCellVec;
  HFCellVec m_hfCellVec;
};

#endif
