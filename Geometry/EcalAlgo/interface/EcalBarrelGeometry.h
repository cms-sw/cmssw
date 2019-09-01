#ifndef EcalBarrelGeometry_h
#define EcalBarrelGeometry_h

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "Geometry/Records/interface/PEcalBarrelRcd.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <vector>
#include <atomic>

class EcalBarrelGeometry final : public CaloSubdetectorGeometry {
public:
  typedef std::vector<TruncatedPyramid> CellVec;

  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;

  typedef IdealGeometryRecord IdealRecord;
  typedef EcalBarrelGeometryRecord AlignedRecord;
  typedef EBAlignmentRcd AlignmentRecord;
  typedef PEcalBarrelRcd PGeometryRecord;

  typedef EZArrayFL<EEDetId> OrderedListOfEEDetId;  // like an stl vector: begin(), end(), [i]

  typedef std::vector<OrderedListOfEEDetId*> VecOrdListEEDetIdPtr;

  typedef EcalBarrelNumberingScheme NumberingScheme;

  typedef EBDetId DetIdType;

  enum { k_NumberOfCellsForCorners = EBDetId::kSizeForDenseIndexing };

  enum { k_NumberOfShapes = 17 };

  enum { k_NumberOfParametersPerShape = 11 };

  static std::string dbString() { return "PEcalBarrelRcd"; }

  unsigned int numberOfShapes() const override { return k_NumberOfShapes; }
  unsigned int numberOfParametersPerShape() const override { return k_NumberOfParametersPerShape; }

  EcalBarrelGeometry();

  ~EcalBarrelGeometry() override;

  int getNumXtalsPhiDirection() const { return _nnxtalPhi; }

  int getNumXtalsEtaDirection() const { return _nnxtalEta; }

  const std::vector<int>& getEtaBaskets() const { return _EtaBaskets; }

  int getBasketSizeInPhi() const { return _PhiBaskets; }

  void setNumXtalsPhiDirection(const int& nnxtalPhi) { _nnxtalPhi = nnxtalPhi; }

  void setNumXtalsEtaDirection(const int& nnxtalEta) { _nnxtalEta = nnxtalEta; }

  void setEtaBaskets(const std::vector<int>& EtaBaskets) { _EtaBaskets = EtaBaskets; }

  void setBasketSizeInPhi(const int& PhiBaskets) { _PhiBaskets = PhiBaskets; }

  const OrderedListOfEEDetId* getClosestEndcapCells(EBDetId id) const;

  // Get closest cell, etc...

  DetId getClosestCell(const GlobalPoint& r) const override;

  CaloSubdetectorGeometry::DetIdSet getCells(const GlobalPoint& r, double dR) const override;

  CCGFloat avgRadiusXYFrontFaceCenter() const;

  static std::string hitString() { return "EcalHitsEB"; }

  static std::string producerTag() { return "EcalBarrel"; }

  static unsigned int numberOfAlignments() { return 36; }

  static unsigned int alignmentTransformIndexLocal(const DetId& id);

  static unsigned int alignmentTransformIndexGlobal(const DetId& id);

  static DetId detIdFromLocalAlignmentIndex(unsigned int iLoc);

  static void localCorners(Pt3DVec& lc, const CCGFloat* pv, unsigned int i, Pt3D& ref);

  void newCell(const GlobalPoint& f1,
               const GlobalPoint& f2,
               const GlobalPoint& f3,
               const CCGFloat* parm,
               const DetId& detId) override;

  bool present(const DetId& id) const override;

protected:
  // Modify the RawPtr class
  const CaloCellGeometry* getGeometryRawPtr(uint32_t index) const override;

private:
  /** number of crystals in eta direction */
  int _nnxtalEta;

  /** number of crystals in phi direction */
  int _nnxtalPhi;

  /** size of the baskets in the eta direction. This is needed
	  to find out whether two adjacent crystals lie in the same
	  basked ('module') or not (e.g. this can be used for correcting
	  cluster energies etc.) */
  std::vector<int> _EtaBaskets;

  /** size of one basket in phi */
  int _PhiBaskets;

  mutable std::atomic<EZMgrFL<EEDetId>*> m_borderMgr;

  mutable std::atomic<VecOrdListEEDetIdPtr*> m_borderPtrVec;
  CMS_THREAD_GUARD(m_check) mutable CCGFloat m_radius;
  mutable std::atomic<bool> m_check;

  CellVec m_cellVec;
};

#endif
