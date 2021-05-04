#ifndef GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H 1

#include <vector>
#include <map>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

/** \class HcalTopology

   The HcalTopology class contains a set of constants which represent
   the topology (tower relationship) of the CMS HCAL as built.  These
   constants can be used to determine neighbor relationships and
   existence of cells.

   For use with limited setups (testbeam, cosmic stands, etc), the
   topology can be limited by creating a rejection list -- a list of
   cells which would normally exist in the full CMS HCAL, but are not
   present for the specified topology.
    
*/
class HcalTopology : public CaloSubdetectorTopology {
public:
  HcalTopology(const HcalDDDRecConstants* hcons, const bool mergePosition = false);
  HcalTopology(HcalTopologyMode::Mode mode,
               int maxDepthHB,
               int maxDepthHE,
               HcalTopologyMode::TriggerMode tmode = HcalTopologyMode::TriggerMode_2009);

  HcalTopologyMode::Mode mode() const { return mode_; }
  HcalTopologyMode::TriggerMode triggerMode() const { return triggerMode_; }
  /** Add a cell to exclusion list */
  void exclude(const HcalDetId& id);
  /** Exclude an entire subdetector */
  void excludeSubdetector(HcalSubdetector subdet);
  /** Exclude an eta/phi/depth range for a given subdetector */
  int exclude(HcalSubdetector subdet, int ieta1, int ieta2, int iphi1, int iphi2, int depth1 = 1, int depth2 = 4);

  static std::string producerTag() { return "HCAL"; }

  /// return a linear packed id
  unsigned int detId2denseId(const DetId& id) const override;
  /// return a linear packed id
  DetId denseId2detId(unsigned int /*denseid*/) const override;
  /// return a count of valid cells (for dense indexing use)
  unsigned int ncells() const override;
  /// return a version which identifies the given topology
  int topoVersion() const override;

  /** Is this a valid cell id? */
  bool valid(const DetId& id) const override;
  /** Is this a valid cell id? */
  bool validHcal(const HcalDetId& id) const;
  bool validDetId(HcalSubdetector subdet, int ieta, int iphi, int depth) const;
  bool validHT(const HcalTrigTowerDetId& id) const;
  bool validCalib(const HcalCalibDetId& id) const;
  /** Is this a valid cell in context of Plan1 */
  bool validHcal(const HcalDetId& id, const unsigned int flag) const;
  /** Flag=0 for unmerged Id's; =1 for for merged Id's; =2 for either */

  /** Get the neighbors of the given cell in east direction*/
  std::vector<DetId> east(const DetId& id) const override;
  /** Get the neighbors of the given cell in west direction*/
  std::vector<DetId> west(const DetId& id) const override;
  /** Get the neighbors of the given cell in north direction*/
  std::vector<DetId> north(const DetId& id) const override;
  /** Get the neighbors of the given cell in south direction*/
  std::vector<DetId> south(const DetId& id) const override;
  /** Get the neighbors of the given cell in up direction (outward)*/
  std::vector<DetId> up(const DetId& id) const override;
  /** Get the neighbors of the given cell in down direction (inward)*/
  std::vector<DetId> down(const DetId& id) const override;

  /** Get the neighbors of the given cell with higher (signed) ieta */
  int incIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower (signed) ieta */
  int decIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbor (if present) of the given cell with higher iphi */
  bool incIPhi(const HcalDetId& id, HcalDetId& neighbor) const;
  /** Get the neighbor (if present) of the given cell with lower iphi */
  bool decIPhi(const HcalDetId& id, HcalDetId& neighbor) const;
  /** Get the detector behind this one */
  bool incrementDepth(HcalDetId& id) const;
  /** Get the detector in front of this one */
  bool decrementDepth(HcalDetId& id) const;

  int firstHBRing() const { return firstHBRing_; }
  int lastHBRing() const { return lastHBRing_; }
  int firstHERing() const { return firstHERing_; }
  int lastHERing() const { return lastHERing_; }
  int lastHBHERing() const { return std::max(lastHBRing_, lastHERing_); }
  int firstHFRing() const { return firstHFRing_; }
  int lastHFRing() const { return lastHFRing_; }
  int firstHORing() const { return firstHORing_; }
  int lastHORing() const { return lastHORing_; }

  int firstHEDoublePhiRing() const { return firstHEDoublePhiRing_; }
  int firstHEQuadPhiRing() const { return firstHEQuadPhiRing_; }
  int firstHFQuadPhiRing() const { return firstHFQuadPhiRing_; }
  int firstHETripleDepthRing() const { return firstHETripleDepthRing_; }
  int singlePhiBins() const { return singlePhiBins_; }
  int doublePhiBins() const { return doublePhiBins_; }

  /// finds the number of depth bins and which is the number to start with
  void depthBinInformation(
      HcalSubdetector subdet, int etaRing, int iphi, int zside, int& nDepthBins, int& startingBin) const;
  bool mergedDepth29(HcalDetId id) const { return hcons_->mergedDepthList29(id.ieta(), id.iphi(), id.depth()); }
  std::vector<int> mergedDepthList29(HcalDetId id) const { return hcons_->mergedDepthList29(id.ieta(), id.iphi()); }

  /// how many phi segments in this ring
  int nPhiBins(int etaRing) const;
  int nPhiBins(HcalSubdetector subdet, int etaRing) const;

  /// eta and phi index from eta, phi values
  int etaRing(HcalSubdetector subdet, double eta) const;
  int phiBin(HcalSubdetector subdet, int etaRing, double phi) const;

  /// for each of the ~17 depth segments, specify which readout bin they belong to
  /// if the ring is not found, the first one with a lower ring will be returned.
  void getDepthSegmentation(const unsigned ring, std::vector<int>& readoutDepths, const bool flag = false) const;
  void setDepthSegmentation(const unsigned ring, const std::vector<int>& readoutDepths, const bool flag);
  /// returns the boundaries of the depth segmentation, so that the first
  /// result is the first segment, and the second result is the first one
  /// of the next segment.  Used for calculating physical bounds.
  std::pair<int, int> segmentBoundaries(const unsigned ring, const unsigned depth, const bool flag = false) const;
  int getPhiZOne(std::vector<std::pair<int, int> >& phiz) const { return hcons_->getPhiZOne(phiz); }

  unsigned int getHBSize() const { return HBSize_; }
  unsigned int getHESize() const { return HESize_; }
  unsigned int getHOSize() const { return HOSize_; }
  unsigned int getHFSize() const { return HFSize_; }
  unsigned int getHTSize() const { return HTSize_; }
  unsigned int getCALIBSize() const { return CALIBSize_; }

  int maxDepthHB() const { return maxDepthHB_; }
  int maxDepthHE() const { return maxDepthHE_; }
  int maxDepth(void) const;
  int maxDepth(HcalSubdetector subdet) const;
  double etaMax(HcalSubdetector subdet) const;
  std::pair<double, double> etaRange(HcalSubdetector subdet, int ieta) const;

  /// return a linear packed id from HB
  unsigned int detId2denseIdHB(const DetId& id) const;
  /// return a linear packed id from HE
  unsigned int detId2denseIdHE(const DetId& id) const;
  /// return a linear packed id from HO
  unsigned int detId2denseIdHO(const DetId& id) const;
  /// return a linear packed id from HF
  unsigned int detId2denseIdHF(const DetId& id) const;
  /// return a linear packed id from HT
  unsigned int detId2denseIdHT(const DetId& id) const;
  /// return a linear packed id from CALIB
  unsigned int detId2denseIdCALIB(const DetId& id) const;
  /// return a Calib DetId from linear packed id
  HcalCalibDetId denseId2detIdCALIB(const unsigned int& id) const;

  unsigned int getNumberOfShapes() const { return numberOfShapes_; }
  bool isBH() const { return ((hcons_ == nullptr) ? false : hcons_->isBH()); }

  const HcalDDDRecConstants* dddConstants() const { return hcons_; }
  bool withSpecialRBXHBHE() const { return hcons_->withSpecialRBXHBHE(); }
  HcalDetId mergedDepthDetId(const HcalDetId& id) const { return hcons_->mergedDepthDetId(id); }
  bool getMergePositionFlag() const { return mergePosition_; }
  void unmergeDepthDetId(const HcalDetId& id, std::vector<HcalDetId>& ids) const { hcons_->unmergeDepthDetId(id, ids); }
  // Returns the DetId of the front Id if it is a merged RecHit in "Plan 1"
  HcalDetId idFront(const HcalDetId& id) const { return hcons_->idFront(id); }
  HcalDetId idBack(const HcalDetId& id) const { return hcons_->idBack(id); }

private:
  /** Get the neighbors of the given cell with higher absolute ieta */
  int incAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower absolute ieta */
  int decAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;

  /** Is this a valid cell id, ignoring the exclusion list */
  bool validDetIdPreLS1(const HcalDetId& id) const;
  bool validRaw(const HcalDetId& id) const;
  unsigned int detId2denseIdPreLS1(const DetId& id) const;
  bool isExcluded(const HcalDetId& id) const;

  const HcalDDDRecConstants* hcons_;
  bool mergePosition_;
  std::vector<HcalDetId> exclusionList_;
  bool excludeHB_, excludeHE_, excludeHO_, excludeHF_;

  HcalTopologyMode::Mode mode_;
  HcalTopologyMode::TriggerMode triggerMode_;

  int firstHBRing_, lastHBRing_;
  int firstHERing_, lastHERing_;
  int firstHFRing_, lastHFRing_;
  int firstHORing_, lastHORing_;

  std::vector<HcalDDDRecConstants::HcalEtaBin> etaBinsHB_, etaBinsHE_;
  int nEtaHB_, nEtaHE_;

  int firstHEDoublePhiRing_, firstHEQuadPhiRing_, firstHFQuadPhiRing_;
  int firstHETripleDepthRing_;
  int singlePhiBins_, doublePhiBins_;
  int maxDepthHB_, maxDepthHE_, maxDepthHF_;
  int etaHE2HF_, etaHF2HE_;
  int maxEta_, maxPhiHE_;

  unsigned int HBSize_;
  unsigned int HESize_;
  unsigned int HOSize_;
  unsigned int HFSize_;
  unsigned int HTSize_;
  unsigned int CALIBSize_;
  unsigned int numberOfShapes_;

  std::vector<double> etaTable, etaTableHF, dPhiTable, dPhiTableHF;
  std::vector<double> phioff;
  std::vector<int> unitPhi, unitPhiHF;

  int topoVersion_;

  // index is ring;
  typedef std::map<unsigned, std::vector<int> > SegmentationMap;
  SegmentationMap depthSegmentation_;
  SegmentationMap depthSegmentationOne_;

  enum {
    kHBhalf = 1296,
    kHEhalf = 1296,
    kHOhalf = 1080,
    kHFhalf = 864,
    kHThalf = 2088,
    kZDChalf = 11,
    kCASTORhalf = 224,
    kCALIBhalf = 693,
    kHThalfPhase1 = 2520,
    kHcalhalf = kHBhalf + kHEhalf + kHOhalf + kHFhalf
  };
  enum { kSizeForDenseIndexingPreLS1 = 2 * kHcalhalf };
  enum { kHBSizePreLS1 = 2 * kHBhalf };
  enum { kHESizePreLS1 = 2 * kHEhalf };
  enum { kHOSizePreLS1 = 2 * kHOhalf };
  enum { kHFSizePreLS1 = 2 * kHFhalf };
  enum { kHTSizePreLS1 = 2 * kHThalf };
  enum { kHTSizePhase1 = 2 * kHThalfPhase1 };
  enum { kCALIBSizePreLS1 = 2 * kCALIBhalf };
  static constexpr int minMaxDepth_ = 4;
  static constexpr unsigned int minPhi_ = 1, maxPhi_ = 72;
  static constexpr unsigned int kOffCalibHB_ = 0;
  static constexpr unsigned int nchanCalibHB_ = 3, nEtaCalibHB_ = 2;
  static constexpr unsigned int mPhiCalibHB_ = 4, kPhiCalibHB_ = maxPhi_ / mPhiCalibHB_;
  static constexpr int chanCalibHB_[nchanCalibHB_] = {0, 1, 2};
  static constexpr int etaCalibHB_[nEtaCalibHB_] = {-1, 1};
  static constexpr unsigned int kchanCalibHB_ = nchanCalibHB_ * kPhiCalibHB_;
  static constexpr unsigned int nCalibHB_ = kchanCalibHB_ * nEtaCalibHB_;
  static constexpr unsigned int kOffCalibHE1_ = kOffCalibHB_ + nCalibHB_;
  static constexpr unsigned int nchanCalibHE1_ = 6, nEtaCalibHE_ = 2;
  static constexpr unsigned int mPhiCalibHE_ = 4, kPhiCalibHE_ = maxPhi_ / mPhiCalibHE_;
  static constexpr int chanCalibHE1_[nchanCalibHE1_] = {0, 1, 3, 4, 5, 6};
  static constexpr int etaCalibHE_[nEtaCalibHE_] = {-1, 1};
  static constexpr unsigned int kchanCalibHE1_ = nchanCalibHE1_ * kPhiCalibHE_;
  static constexpr unsigned int nCalibHE1_ = kchanCalibHE1_ * nEtaCalibHE_;
  static constexpr unsigned int kOffCalibHF1_ = kOffCalibHE1_ + nCalibHE1_;
  static constexpr unsigned int nchanCalibHF1_ = 3, nEtaCalibHF_ = 2;
  static constexpr unsigned int mPhiCalibHF1_ = 18, kPhiCalibHF1_ = maxPhi_ / mPhiCalibHF1_;
  static constexpr int chanCalibHF1_[nchanCalibHF1_] = {0, 1, 8};
  static constexpr int etaCalibHF_[nEtaCalibHF_] = {-1, 1};
  static constexpr unsigned int kchanCalibHF1_ = nchanCalibHF1_ * kPhiCalibHF1_;
  static constexpr unsigned int nCalibHF1_ = kchanCalibHF1_ * nEtaCalibHF_;
  static constexpr unsigned int kOffCalibHO1_ = kOffCalibHF1_ + nCalibHF1_;
  static constexpr unsigned int nchanCalibHO_ = 2, nEtaCalibHO_ = 5;
  static constexpr int chanCalibHOs_ = 7;
  static constexpr int chanCalibHO_[nchanCalibHO_] = {0, 1};
  static constexpr int etaCalibHO_[nEtaCalibHO_] = {-2, -1, 0, 1, 2};
  static constexpr int phiCalibHO_[nEtaCalibHO_] = {59, 47, 53, 47, 47};
  static constexpr unsigned int mPhiCalibHO0_ = 6, mPhiCalibHO1_ = 12;
  static constexpr unsigned int kPhiCalibHO0_ = maxPhi_ / mPhiCalibHO0_;
  static constexpr unsigned int kPhiCalibHO1_ = maxPhi_ / mPhiCalibHO1_;
  static constexpr unsigned int kPhiCalibHO2_ = 4 * kPhiCalibHO1_ + kPhiCalibHO0_;
  static constexpr unsigned int nCalibHO1_ = kPhiCalibHO2_ * nchanCalibHO_;
  static constexpr unsigned int kOffCalibHO2_ = kOffCalibHO1_ + nCalibHO1_;
  static constexpr unsigned int nCalibHO2_ = nEtaCalibHO_;
  static constexpr unsigned int kOffCalibHE2_ = kOffCalibHO2_ + nCalibHO2_;
  static constexpr int chanCalibHE2_ = 2;
  static constexpr unsigned int kchanCalibHE2_ = kPhiCalibHE_;
  static constexpr unsigned int nCalibHE2_ = kchanCalibHE2_ * nEtaCalibHE_;
  static constexpr unsigned int kOffCalibHF2_ = kOffCalibHE2_ + nCalibHE2_;
  static constexpr int chanCalibHF2_ = 9, phiCalibHF2_ = 1;
  static constexpr unsigned int kchanCalibHF2_ = 1;
  static constexpr unsigned int nCalibHF2_ = nEtaCalibHF_;
  static constexpr unsigned int kOffCalibHOX_ = kOffCalibHF2_ + nCalibHF2_;
  static constexpr unsigned int nEtaCalibHOX_ = 2;
  static constexpr int etaCalibHOX_[nEtaCalibHOX_] = {4, 15};
  static constexpr unsigned int mPhiCalibHOX_[nEtaCalibHOX_] = {2, 1};
  static constexpr unsigned int nPhiCalibHOX_[nEtaCalibHOX_] = {36, 72};
  static constexpr int nCalibHOX_ = 2 * (nPhiCalibHOX_[0] + nPhiCalibHOX_[1]);
  static constexpr int phiCalibHOX1_ = 4, phiCalibHOX2_ = 12, phiCalibHOX3_ = 6;
  static constexpr unsigned int kOffCalibHBX_ = kOffCalibHOX_ + nCalibHOX_;
  static constexpr unsigned int nEtaCalibHBX_ = 1, mPhiCalibHBX_ = 1;
  static constexpr int etaCalibHBX_ = 16;
  static constexpr unsigned int kPhiCalibHBX_ = maxPhi_ / mPhiCalibHBX_;
  static constexpr unsigned int nCalibHBX_ = 2 * kPhiCalibHBX_ * nEtaCalibHBX_;
  static constexpr unsigned int kOffCalibHEX_ = kOffCalibHBX_ + nCalibHBX_;
  static constexpr unsigned int nEtaCalibHEX_ = 2, mPhiCalibHEX_ = 2;
  static constexpr int etaCalibHEX_[nEtaCalibHEX_] = {25, 27};
  static constexpr unsigned int kPhiCalibHEX_ = maxPhi_ / mPhiCalibHEX_;
  static constexpr unsigned int nCalibHEX_ = 2 * kPhiCalibHEX_ * nEtaCalibHEX_;
  static constexpr unsigned int kOffCalibHFX_ = kOffCalibHEX_ + nCalibHEX_;
};

#endif
