#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"

#include <iomanip>
#include <ostream>
#include <iostream>

const float BeamCurrentInfo::scaleFactor = 1e10;

float BeamCurrentInfo::getBeam1IntensityBX(int bx) const {
  unpackData();
  return beam1IntensitiesUnpacked_.at(bx);
}

const std::vector<float>& BeamCurrentInfo::getBeam1Intensities() const { 
  unpackData();
  return beam1IntensitiesUnpacked_;

}

float BeamCurrentInfo::getBeam2IntensityBX(int bx) const {
  unpackData();
  return beam2IntensitiesUnpacked_.at(bx); }

const std::vector<float>& BeamCurrentInfo::getBeam2Intensities() const {
  unpackData();
  return beam2IntensitiesUnpacked_;
}

bool BeamCurrentInfo::isProductEqual(BeamCurrentInfo const& next) const {
  return (beam1IntensitiesPacked_ == next.beam1IntensitiesPacked_ &&
	  beam2IntensitiesPacked_ == next.beam2IntensitiesPacked_);
}

void BeamCurrentInfo::fillBeamIntensities(const std::vector<float>& beam1Intensities,
					  const std::vector<float>& beam2Intensities) {
  beam1IntensitiesUnpacked_.assign(beam1Intensities.begin(), beam1Intensities.end());
  beam2IntensitiesUnpacked_.assign(beam2Intensities.begin(), beam2Intensities.end());
  unpackedReady_ = true;
  packData();
}

void BeamCurrentInfo::fill(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities) {
  fillBeamIntensities(beam1Intensities, beam2Intensities);
}

// Convert unpacked data to packed data (when it is filled).
void BeamCurrentInfo::packData(void) {
  beam1IntensitiesPacked_.resize(beam1IntensitiesUnpacked_.size());
  beam2IntensitiesPacked_.resize(beam2IntensitiesUnpacked_.size());

  for (unsigned int i=0; i<beam1IntensitiesUnpacked_.size(); i++) {
    beam1IntensitiesPacked_[i] = MiniFloatConverter::float32to16(beam1IntensitiesUnpacked_[i]/scaleFactor);
  }
  for (unsigned int i=0; i<beam2IntensitiesUnpacked_.size(); i++) {
    beam2IntensitiesPacked_[i] = MiniFloatConverter::float32to16(beam2IntensitiesUnpacked_[i]/scaleFactor);
  }

  // Re-unpack the data so that it matches the packed data.
  unpackedReady_ = false;
  unpackData();
}

// Convert packed data to unpacked data when accessors are called.
void BeamCurrentInfo::unpackData(void) const {
  if (unpackedReady_) return;

  beam1IntensitiesUnpacked_.resize(beam1IntensitiesPacked_.size());
  beam2IntensitiesUnpacked_.resize(beam2IntensitiesPacked_.size());

  for (unsigned int i=0; i<beam1IntensitiesPacked_.size(); i++) {
    beam1IntensitiesUnpacked_[i] = MiniFloatConverter::float16to32(beam1IntensitiesPacked_[i])*scaleFactor;
  }
  for (unsigned int i=0; i<beam2IntensitiesPacked_.size(); i++) {
    beam2IntensitiesUnpacked_[i] = MiniFloatConverter::float16to32(beam2IntensitiesPacked_[i])*scaleFactor;
  }
  unpackedReady_ = true;
}
       

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo) {
  s << std::endl << "Dumping BeamCurrentInfo..." << std::endl;
  s << "  beam1Intensities = ";
  const std::vector<float>& b1int = beamInfo.getBeam1Intensities();
  const std::vector<uint16_t>& b1intPacked = beamInfo.getBeam1IntensitiesPacked();
  for (unsigned int i=0; i<10 && i<b1int.size(); ++i) {
    s << b1int.at(i) << " ";
  }
  s << "..." << std::endl << "     (packed: ";
  for (unsigned int i=0; i<10 && i<b1intPacked.size(); ++i) {
    s << b1intPacked.at(i) << " ";
  }
  s << "...)" << std::endl;
  s << "  beam2Intensities = ";
  const std::vector<float>& b2int = beamInfo.getBeam2Intensities();
  for (unsigned int i=0; i<10 && i<b2int.size(); ++i) {
    s << b2int.at(i) << " ";
  }
  s << " ..." << std::endl;

  return s;
}
