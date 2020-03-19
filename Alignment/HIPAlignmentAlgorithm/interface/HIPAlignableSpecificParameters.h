#ifndef Alignment_HIPAlignmentAlgorithm_HIPAlignableSpecificParameters_h
#define Alignment_HIPAlignmentAlgorithm_HIPAlignableSpecificParameters_h

#include <vector>
#include "Alignment/CommonAlignment/interface/Alignable.h"

class HIPAlignableSpecificParameters {
protected:
  // Use the pointer to match
  const Alignable* aliObj;
  const bool defaultFlag;

public:
  // These are the actual parameters
  double minRelParError;
  double maxRelParError;
  int minNHits;
  double maxHitPull;
  bool applyPixelProbCut;
  bool usePixelProbXYOrProbQ;
  double minPixelProbXY;
  double maxPixelProbXY;
  double minPixelProbQ;
  double maxPixelProbQ;

  HIPAlignableSpecificParameters(const Alignable* aliObj_, bool defaultFlag_ = false);
  HIPAlignableSpecificParameters(const HIPAlignableSpecificParameters& other);
  ~HIPAlignableSpecificParameters() {}

  bool isDefault() const;

  align::ID id() const;
  align::StructureType objId() const;
  bool matchAlignable(const Alignable* ali) const;
};

#endif
