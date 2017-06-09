#ifndef Alignment_HIPAlignmentAlgorithm_HIPAlignableSpecificParameters_h
#define Alignment_HIPAlignmentAlgorithm_HIPAlignableSpecificParameters_h

#include <vector>
#include "Alignment/CommonAlignment/interface/Alignable.h"

class HIPAlignableSpecificParameters{
protected:
  // Use the pointer to match
  const Alignable* aliObj;

public:
  // These are the actual parameters
  int minNHits;
  double maxRelParError;
  double maxHitPull;

  HIPAlignableSpecificParameters(const Alignable* aliObj_);
  HIPAlignableSpecificParameters(const HIPAlignableSpecificParameters& other);
  ~HIPAlignableSpecificParameters(){}

  align::ID id()const;
  align::StructureType objId()const;
  bool matchAlignable(const Alignable* ali)const;

};

#endif
