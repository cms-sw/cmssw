#ifndef _RECOMET_METALGORITHMS_HCALHPDRBXMAP_H_
#define _RECOMET_METALGORITHMS_HCALHPDRBXMAP_H_


//
// HcalHPDRBXMap.h
//
//   description: Algorithm which isomorphically maps HPD/RBX locations to
//                integers ranging from 0 to NUM_HPDS-1/NUM_RBXS-1.  The HPDs/RBXs
//                are ordered from lowest to highest: HB+, HB-, HE+, HE-.
//                This is used extensively by the various HcalNoise container
//                classes.  The constructor and destructor are hidden, since
//                the only methods of interest are static.  All the methods
//                here are O(1).
//
//   author: J.P. Chou, Brown
//

#include "boost/array.hpp"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>

class HcalHPDRBXMap {
 public:
  
  // "magic numbers"
  // total number of HPDs in the HB and HE
  const static int NUM_HPDS=288;
  // total number of HPDs per subdetector (HB+, HB-, HE+, HE-)
  const static int NUM_HPDS_PER_SUBDET=72;
  // number of HPDs per RBX
  const static int NUM_HPDS_PER_RBX = 4;
  // total number of RBXs in the HB and HE
  const static int NUM_RBXS=72;
  // total number of RBXs per subdetector (e.g. HB+, HB-, HE+, HE-)
  const static int NUM_RBXS_PER_SUBDET=18;

  // access magic numbers by inline function
  inline int static numHPDs(void) { return NUM_HPDS; }
  inline int static numHPDsPerSubdet(void) { return NUM_HPDS_PER_SUBDET; }
  inline int static numHPDsPerRBX(void) { return NUM_HPDS_PER_RBX; }
  inline int static numRBXs(void) { return NUM_RBXS; }
  inline int static numRBXsPerSubdet(void) { return NUM_RBXS_PER_SUBDET; }

  // determines whether an HPD or RBX index is valid
  // HPDs run from [0,NUM_HPDS-1], and RBXs run from [0,NUM_RBXS-1]
  bool static isValidHPD(int index);
  bool static isValidRBX(int index);
  
  // determines whether a HcalDetId corresponds to a valid HPD/RBX
  // this requires that the HcalDetId be in the HB or HE, does not check depth
  bool static isValid(const HcalDetId&);

  // determines whether the ieta, iphi coordinate corresponds to a valid HPD/RBX
  bool static isValid(int ieta, int iphi);

  // location of the HPD/RBX in the detector based on the HPD/RBX index
  // exception is thrown if index is invalid
  HcalSubdetector static subdetHPD(int index);
  HcalSubdetector static subdetRBX(int index);
  int static zsideHPD(int index);
  int static zsideRBX(int index);
  int static iphiloHPD(int index);
  int static iphiloRBX(int index);
  int static iphihiHPD(int index);
  int static iphihiRBX(int index);

  // returns a list of HPD indices found in a given RBX
  // exception is thrown if rbxindex is invalid
  // HPD indices are ordered in phi-space
  void static indicesHPDfromRBX(int rbxindex, boost::array<int, NUM_HPDS_PER_RBX>& hpdindices);

  // returns the RBX index given an HPD index
  // exception is thrown if hpdindex is invalid
  int static indexRBXfromHPD(int hpdindex);

  // get the HPD/RBX index from an HcalDetector id
  // throws an exception if the HcalDetID does not correspond to a valid HPD/RBX
  int static indexHPD(const HcalDetId&);
  int static indexRBX(const HcalDetId&);

  // get the HPD/RBX indices corresponding to an ieta, iphi coordinate
  // throws an exception if the ieta and iphi do not correspond to a valid HPD/RBX
  void static indexHPDfromEtaPhi(int ieta, int iphi, std::vector<int>& hpdindices);
  void static indexRBXfromEtaPhi(int ieta, int iphi, std::vector<int>& rbxindices);

 private:
  HcalHPDRBXMap();
  ~HcalHPDRBXMap();

};

#endif
