#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H 1

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

/** \class CaloTowerTopology
  *  
  * \author J. Mans - Minnesota
  */
class CaloTowerTopology final : public CaloSubdetectorTopology {
public:
  /// standard constructor
  CaloTowerTopology(const HcalTopology * topology);
  /// virtual destructor
  ~CaloTowerTopology() override { }
  /// is this detid present in the Topology?
  bool valid(const DetId& id) const override;
  virtual bool validDetId(const CaloTowerDetId& id) const;
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

  //mimic accessors from HcalTopology, but with continuous ieta
  int firstHBRing() const {return firstHBRing_;}
  int lastHBRing()  const {return lastHBRing_;}
  int firstHERing() const {return firstHERing_;}
  int lastHERing()  const {return lastHERing_;}
  int firstHFRing() const {return firstHFRing_;}
  int lastHFRing()  const {return lastHFRing_;}
  int firstHORing() const {return firstHORing_;}
  int lastHORing()  const {return lastHORing_;}
  int firstHEDoublePhiRing()   const {return firstHEDoublePhiRing_;} 
  int firstHEQuadPhiRing()     const {return firstHEQuadPhiRing_;} 
  int firstHFQuadPhiRing()     const {return firstHFQuadPhiRing_;}
  
  //conversion between CaloTowerTopology ieta and HcalTopology ieta
  int convertCTtoHcal(int ct_ieta) const;
  int convertHcaltoCT(int hcal_ieta, HcalSubdetector subdet) const;

  //dense index functions moved from CaloTowerDetId
  uint32_t denseIndex(const DetId& id) const;
  CaloTowerDetId detIdFromDenseIndex(uint32_t din) const;
  bool validDenseIndex(uint32_t din) const { return ( din < kSizeForDenseIndexing ); }
  uint32_t sizeForDenseIndexing() const { return kSizeForDenseIndexing; }
  
private:
  //member variables
  const HcalTopology * hcaltopo;
  int firstHBRing_, lastHBRing_;
  int firstHERing_, lastHERing_;
  int firstHFRing_, lastHFRing_;
  int firstHORing_, lastHORing_;
  int firstHEDoublePhiRing_, firstHEQuadPhiRing_, firstHFQuadPhiRing_;
  int nSinglePhi_, nDoublePhi_, nQuadPhi_, nEtaHE_;
  uint32_t kSizeForDenseIndexing;

};
#endif
