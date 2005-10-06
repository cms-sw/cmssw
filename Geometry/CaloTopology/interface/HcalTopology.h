#ifndef GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HcalTopology
    
   $Date: 2005/10/03 22:32:17 $
   $Revision: 1.3 $
   \author J. Mans - Minnesota
*/
class HcalTopology {
public:
  HcalTopology();
  void limitDetector(int min_iphi, int max_iphi, int min_ieta, int max_ieta);

  /** Is this a valid cell id? */
  bool valid(const HcalDetId& id) const;
  /** Get the neighbors of the given cell with higher (signed) ieta */
  int incIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower (signed) ieta */
  int decIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbor (if present) of the given cell with higher iphi */
  bool incIPhi(const HcalDetId& id, HcalDetId &neighbor) const;
  /** Get the neighbor (if present) of the given cell with lower iphi */
  bool decIPhi(const HcalDetId& id, HcalDetId &neighbor) const;

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

private:
  /** Get the neighbors of the given cell with higher absolute ieta */
  int incAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;
  /** Get the neighbors of the given cell with lower absolute ieta */
  int decAIEta(const HcalDetId& id, HcalDetId neighbors[2]) const;


  int min_iphi_;
  int max_iphi_;
  int min_ieta_;
  int max_ieta_;

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

};


#endif
