#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignableSpecificParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const Alignable* aliObj_, bool defaultFlag_) :
aliObj(aliObj_),
defaultFlag(defaultFlag_),
minNHits(0),
maxRelParError(-1),
maxHitPull(-1)
{}

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const HIPAlignableSpecificParameters& other) :
aliObj(other.aliObj),
defaultFlag(other.defaultFlag),
minNHits(other.minNHits),
maxRelParError(other.maxRelParError),
maxHitPull(other.maxHitPull)
{}

align::ID HIPAlignableSpecificParameters::id()const{ if (aliObj!=0) return aliObj->id(); else return 0; }
align::StructureType HIPAlignableSpecificParameters::objId()const{ if (aliObj!=0) return aliObj->alignableObjectId(); else return align::invalid; }

bool HIPAlignableSpecificParameters::matchAlignable(const Alignable* ali)const{
  if (aliObj==(Alignable*)0) return false;
  bool result = (aliObj==ali);
  if (!result){ // Check deep components of the alignable for this specification
    for (auto const& alideep : aliObj->deepComponents()){
      if (alideep==ali){ result = true; break; }
    }
    //if (result) edm::LogWarning("Alignment") // This is correct. Ideally one should specify the same alignables aligned in the specifications
    //  << "[HIPAlignableSpecificParameters::matchAlignable] Alignment object with id " << ali->id() << " / " << ali->alignableObjectId()
    //  << " was found as a component of " << this->id() << " / " << this->objId();
  }
  return result;
}

bool HIPAlignableSpecificParameters::isDefault()const{ return defaultFlag; }

