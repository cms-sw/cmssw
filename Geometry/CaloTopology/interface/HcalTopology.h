#ifndef GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H 1

#include <vector>
#include <map>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopologyMode.h"

/** \class HcalTopology

   The HcalTopology class contains a set of constants which represent
   the topology (tower relationship) of the CMS HCAL as built.  These
   constants can be used to determine neighbor relationships and
   existence of cells.

   For use with limited setups (testbeam, cosmic stands, etc), the
   topology can be limited by creating a rejection list -- a list of
   cells which would normally exist in the full CMS HCAL, but are not
   present for the specified topology.
    
   $Date: 2012/11/13 04:53:17 $
   $Revision: 1.18 $
   \author J. Mans - Minnesota
*/
class HcalTopology : public CaloSubdetectorTopology {
public:

  HcalTopology( HcalTopologyMode::Mode mode, int maxDepthHB, int maxDepthHE, HcalTopologyMode::TriggerMode tmode=HcalTopologyMode::tm_LHC_PreLS1);
	
  HcalTopologyMode::Mode mode() const {return mode_;}
  HcalTopologyMode::TriggerMode triggerMode() const { return triggerMode_; }
  /** Add a cell to exclusion list */
  void exclude(const HcalDetId& id);
  /** Exclude an entire subdetector */
  void excludeSubdetector(HcalSubdetector subdet);
  /** Exclude an eta/phi/depth range for a given subdetector */
  int exclude(HcalSubdetector subdet, int ieta1, int ieta2, int iphi1, int iphi2, int depth1=1, int depth2=4);


  /// return a linear packed id
  virtual unsigned int detId2denseId(const DetId& id) const;
  /// return a linear packed id
  virtual DetId denseId2detId(unsigned int /*denseid*/) const;
  /// return a count of valid cells (for dense indexing use)
  virtual unsigned int ncells() const;
  /// return a version which identifies the given topology
  virtual int topoVersion() const;

  /** Is this a valid cell id? */
  virtual bool valid(const DetId& id) const;
  /** Is this a valid cell id? */
  bool validHcal(const HcalDetId& id) const;
  /** Get the neighbors of the given cell in east direction*/
  virtual std::vector<DetId> east(const DetId& id) const;
  /** Get the neighbors of the given cell in west direction*/
  virtual std::vector<DetId> west(const DetId& id) const;
  /** Get the neighbors of the given cell in north direction*/
  virtual std::vector<DetId> north(const DetId& id) const;
  /** Get the neighbors of the given cell in south direction*/
  virtual std::vector<DetId> south(const DetId& id) const;
  /** Get the neighbors of the given cell in up direction (outward)*/
  virtual std::vector<DetId> up(const DetId& id) const;
  /** Get the neighbors of the given cell in down direction (inward)*/
  virtual std::vector<DetId> down(const DetId& id) const;

  /** Get the neighbors of the given cell with higher (signed) ieta */
  int incIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower (signed) ieta */
  int decIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbor (if present) of the given cell with higher iphi */
  bool incIPhi(const HcalDetId& id, HcalDetId &neighbor) const;
  /** Get the neighbor (if present) of the given cell with lower iphi */
  bool decIPhi(const HcalDetId& id, HcalDetId &neighbor) const;
  /** Get the detector behind this one */
  bool incrementDepth(HcalDetId& id) const;

  int firstHBRing() const {return firstHBRing_;}
  int lastHBRing()  const {return lastHBRing_;}
  int firstHERing() const {return firstHERing_;}
  int lastHERing()  const {return lastHERing_;}
  int firstHFRing() const {return firstHFRing_;}
  int lastHFRing()  const {return lastHFRing_;}
  int firstHORing() const {return firstHORing_;}
  int lastHORing()  const {return lastHORing_;}

  int firstHEDoublePhiRing() const {return firstHEDoublePhiRing_;} 
  int firstHFQuadPhiRing() const { return firstHFQuadPhiRing_; }
  int firstHETripleDepthRing() const {return firstHETripleDepthRing_;}
  int singlePhiBins() const {return singlePhiBins_;}
  int doublePhiBins() const {return doublePhiBins_;}

  /// finds the number of depth bins and which is the number to start with
  void depthBinInformation(HcalSubdetector subdet, int etaRing,
                           int & nDepthBins, int & startingBin) const;

  /// how many phi segments in this ring
  int nPhiBins(int etaRing) const;

  /// for each of the ~17 depth segments, specify which readout bin they belong to
  /// if the ring is not found, the first one with a lower ring will be returned.
  void getDepthSegmentation(unsigned ring, std::vector<int> & readoutDepths) const;
  void setDepthSegmentation(unsigned ring, const std::vector<int> & readoutDepths);
  /// returns the boundaries of the depth segmentation, so that the first
  /// result is the first segment, and the second result is the first one
  /// of the next segment.  Used for calculating physical bounds.
  std::pair<int, int> segmentBoundaries(unsigned ring, unsigned depth) const;


  unsigned int getHBSize() const {return HBSize_;}
  unsigned int getHESize() const {return HESize_;}
  unsigned int getHOSize() const {return HOSize_;}
  unsigned int getHFSize() const {return HFSize_;}
  unsigned int getHTSize() const {return HTSize_;}
  unsigned int getCALIBSize() const {return CALIBSize_;}

  int maxDepthHB() const { return maxDepthHB_;}
  int maxDepthHE() const { return maxDepthHE_;}

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

  unsigned int getNumberOfShapes() const { return numberOfShapes_; }
      
private:
  /** Get the neighbors of the given cell with higher absolute ieta */
  int incAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower absolute ieta */
  int decAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;

  /** Is this a valid cell id, ignoring the exclusion list */
  bool validDetIdPreLS1(const HcalDetId& id) const;
  bool validRaw(const HcalDetId& id) const;
  unsigned int detId2denseIdPreLS1 (const DetId& id) const;

  std::vector<HcalDetId> exclusionList_;
  bool excludeHB_, excludeHE_, excludeHO_, excludeHF_;

  HcalTopologyMode::Mode mode_;
  HcalTopologyMode::TriggerMode triggerMode_;
  bool isExcluded(const HcalDetId& id) const;

  const int firstHBRing_;
  const int lastHBRing_;
  const int firstHERing_;
  const int lastHERing_;
  const int firstHFRing_;
  const int lastHFRing_;
  const int firstHORing_;
  const int lastHORing_;

  const int firstHEDoublePhiRing_;
  const int firstHFQuadPhiRing_;
  const int firstHETripleDepthRing_;
  const int singlePhiBins_;
  const int doublePhiBins_;
  const int maxDepthHB_;
  const int maxDepthHE_;

  unsigned int HBSize_;
  unsigned int HESize_;
  unsigned int HOSize_;
  unsigned int HFSize_;
  unsigned int HTSize_;
  unsigned int CALIBSize_;
  const unsigned int numberOfShapes_;

  int topoVersion_;
    
  // index is ring;
  typedef std::map<unsigned, std::vector<int> > SegmentationMap;
  SegmentationMap depthSegmentation_;

  enum { kHBhalf = 1296 ,
	 kHEhalf = 1296 ,
	 kHOhalf = 1080 ,
	 kHFhalf = 864  ,
	 kHThalf = 2088,
	 kZDChalf = 11,
	 kCASTORhalf = 224,
	 kCALIBhalf = 693,
	 kHcalhalf = kHBhalf + kHEhalf + kHOhalf + kHFhalf } ;
  enum { kSizeForDenseIndexingPreLS1 = 2*kHcalhalf } ;
  enum { kHBSizePreLS1 = 2*kHBhalf } ;
  enum { kHESizePreLS1 = 2*kHEhalf } ;
  enum { kHOSizePreLS1 = 2*kHOhalf } ;
  enum { kHFSizePreLS1 = 2*kHFhalf } ;
  enum { kHTSizePreLS1 = 2*kHThalf };
  enum { kCALIBSizePreLS1 = 2*kCALIBhalf };
};


#endif
