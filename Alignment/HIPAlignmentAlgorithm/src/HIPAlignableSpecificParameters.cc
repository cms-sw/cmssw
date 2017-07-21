#include "Alignment/HIPAlignmentAlgorithm/interface/HIPAlignableSpecificParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const Alignable* aliObj_, bool defaultFlag_) :
aliObj(aliObj_),
defaultFlag(defaultFlag_),
minRelParError(0),
maxRelParError(-1),
minNHits(0),
maxHitPull(-1),
applyPixelProbCut(false),
usePixelProbXYOrProbQ(false),
minPixelProbXY(0),
maxPixelProbXY(1),
minPixelProbQ(0),
maxPixelProbQ(1)
{}

HIPAlignableSpecificParameters::HIPAlignableSpecificParameters(const HIPAlignableSpecificParameters& other) :
aliObj(other.aliObj),
defaultFlag(other.defaultFlag),
minRelParError(other.minRelParError),
maxRelParError(other.maxRelParError),
minNHits(other.minNHits),
maxHitPull(other.maxHitPull),
applyPixelProbCut(other.applyPixelProbCut),
usePixelProbXYOrProbQ(other.usePixelProbXYOrProbQ),
minPixelProbXY(other.minPixelProbXY),
maxPixelProbXY(other.maxPixelProbXY),
minPixelProbQ(other.minPixelProbQ),
maxPixelProbQ(other.maxPixelProbQ)
{}

align::ID HIPAlignableSpecificParameters::id()const{ if (aliObj!=nullptr) return aliObj->id(); else return 0; }
align::StructureType HIPAlignableSpecificParameters::objId()const{ if (aliObj!=nullptr) return aliObj->alignableObjectId(); else return align::invalid; }

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

