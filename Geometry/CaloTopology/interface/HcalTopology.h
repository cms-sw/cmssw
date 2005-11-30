#ifndef GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_HCALTOPOLOGY_H 1

#include <vector>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HcalTopology

   The HcalTopology class contains a set of hardcoded constants which
   represent the topology (tower relationship) of the CMS HCAL as
   built.  These constants can be used to determine neighbor
   relationships and existence of cells.

   For use with limited setups (testbeam, cosmic stands, etc), the
   topology can be limited by creating a rejection list -- a list of
   cells which would normally exist in the full CMS HCAL, but are not
   present for the specified topology.
    
   $Date: 2005/10/06 00:42:49 $
   $Revision: 1.4 $
   \author J. Mans - Minnesota
*/
class HcalTopology {
public:
  HcalTopology();
  
  /** Add a cell to exclusion list */
  void exclude(const HcalDetId& id);
  /** Exclude an entire subdetector */
  void excludeSubdetector(HcalSubdetector subdet);
  /** Exclude an eta/phi/depth range for a given subdetector */
  int exclude(HcalSubdetector subdet, int ieta1, int ieta2, int iphi1, int iphi2, int depth1=1, int depth2=4);

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

  /** Is this a valid cell id, ignoring the exclusion list */
  bool validRaw(const HcalDetId& id) const;

  std::vector<HcalDetId> exclusionList_;
  bool excludeHB_, excludeHE_, excludeHO_, excludeHF_;

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
};


#endif
