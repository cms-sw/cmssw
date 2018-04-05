#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"

#include <iomanip>
#include <ostream>
#include <iostream>

const float BeamCurrentInfo::scaleFactor = 1e10;

float BeamCurrentInfo::getBeam1IntensityBX(int bx) const {
  return beam1IntensitiesUnpacked_.at(bx);
}

const std::vector<float>& BeamCurrentInfo::getBeam1Intensities() const { 
  return beam1IntensitiesUnpacked_;

}

float BeamCurrentInfo::getBeam2IntensityBX(int bx) const {
  return beam2IntensitiesUnpacked_.at(bx); }

const std::vector<float>& BeamCurrentInfo::getBeam2Intensities() const {
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
  packData();
}

void BeamCurrentInfo::fill(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities) {
  fillBeamIntensities(beam1Intensities, beam2Intensities);
}

// Convert unpacked data to packed data (when it is filled).
void BeamCurrentInfo::packData() {
  beam1IntensitiesPacked_.resize(beam1IntensitiesUnpacked_.size());
  beam2IntensitiesPacked_.resize(beam2IntensitiesUnpacked_.size());

  for (unsigned int i=0; i<beam1IntensitiesUnpacked_.size(); i++) {
    beam1IntensitiesPacked_[i] = MiniFloatConverter::float32to16(beam1IntensitiesUnpacked_[i]/scaleFactor);
  }
  for (unsigned int i=0; i<beam2IntensitiesUnpacked_.size(); i++) {
    beam2IntensitiesPacked_[i] = MiniFloatConverter::float32to16(beam2IntensitiesUnpacked_[i]/scaleFactor);
  }

  // Re-unpack the data so that it matches the packed data.
  unpackData();
}

// Convert packed data to unpacked data when reading back data
void BeamCurrentInfo::unpackData(const std::vector<uint16_t>& packed, std::vector<float>& unpacked ) {
  unpacked.resize(packed.size());

  for (unsigned int i=0; i<packed.size(); i++) {
    unpacked[i] = MiniFloatConverter::float16to32(packed[i])*scaleFactor;
  }
}

void BeamCurrentInfo::unpackData() {

  unpackData(beam1IntensitiesPacked_,beam1IntensitiesUnpacked_);
  unpackData(beam2IntensitiesPacked_,beam2IntensitiesUnpacked_);
}
       

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo) {
  s << std::endl << "Dumping BeamCurrentInfo..." << std::endl;
  s << "  beam1Intensities = ";
  const std::vector<float>& b1int = beamInfo.getBeam1Intensities();
  const std::vector<uint16_t>& b1intPacked = beamInfo.getBeam1IntensitiesPacked();
  for (unsigned int i=0; i<10 && i<b1int.size(); ++i) {
    s << b1int[i] << " ";
  }
  s << "..." << std::endl << "     (packed: ";
  for (unsigned int i=0; i<10 && i<b1intPacked.size(); ++i) {
    s << b1intPacked[i] << " ";
  }
  s << "...)" << std::endl;
  s << "  beam2Intensities = ";
  const std::vector<float>& b2int = beamInfo.getBeam2Intensities();
  for (unsigned int i=0; i<10 && i<b2int.size(); ++i) {
    s << b2int[i] << " ";
  }
  s << " ..." << std::endl;

  return s;
}
