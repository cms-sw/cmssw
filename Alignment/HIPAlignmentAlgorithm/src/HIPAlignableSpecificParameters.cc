#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignableSpecificParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const Alignable* aliObj_) :
aliObj(aliObj_),
minNHits(0),
maxRelParError(-1),
maxHitPull(-1)
{}

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const HIPAlignableSpecificParameters& other) :
aliObj(other.aliObj),
minNHits(other.minNHits),
maxRelParError(other.maxRelParError),
maxHitPull(other.maxHitPull)
{}

align::ID HIPAlignableSpecificParameters::id()const{ return aliObj->id(); }
align::StructureType HIPAlignableSpecificParameters::objId()const{ return aliObj->alignableObjectId(); }

bool HIPAlignableSpecificParameters::matchAlignable(const Alignable* ali)const{
  bool result = (aliObj==ali);
  if (!result){ // Check deep components of the alignable for this specification
    for (auto const& alideep : aliObj->deepComponents()){
      if (alideep==ali){ result = true; break; }
    }
    if (result) edm::LogWarning("Alignment") // This is correct. Ideally one should specify the same alignables aligned in the specifications
      << "[HIPAlignableSpecificParameters::matchAlignable] Alignment object with id " << ali->id() << " / " << ali->alignableObjectId()
      << " was found as a component of " << this->id() << " / " << this->objId();
  }
  return result;
}
