#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"

#include <iomanip>
#include <ostream>
#include <iostream>

static const float BeamCurrentInfo::scaleFactor = 1e10;

float BeamCurrentInfo::getBeam1IntensityBX(int bx) const {
  unpackData();
  return beam1IntensitiesUnpacked_.at(bx);
}

const std::vector<float>& BeamCurrentInfo::getBeam1Intensities() const { 
  unpackData();
  return beam1IntensitiesUnpacked_;

}

float getBeam2IntensityBX(int bx) const {
  unpackData();
  return beam2IntensitiesUnpacked_.at(bx); }

const std::vector<float>& getBeam2Intensities() const {
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
  packData();
}

void BeamCurrentInfo::fill(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities) {
  fillBeamIntensities(beam1Intensities, beam2Intensities);
}

// Convert unpacked data to packed data (when it is filled).
void packData(void) {
  MiniFloatConverter mfc;
  beam1IntensitiesPacked_.resize(beam1IntensitiesUnpacked_);
  beam2IntensitiesPacked_.resize(beam2IntensitiesUnpacked_);

  for (int i=0; i<beam1Intensities.size(); i++) {
    beam1IntensitiesPacked_[i] = mfc.float32to16(beam1IntensitiesUnpacked_[i]/scaleFactor);
  }
  for (int i=0; i<beam2Intensities.size(); i++) {
    beam2IntensitiesPacked_[i] = mfc.float32to16(beam2IntensitiesUnpacked_[i]/scaleFactor);
  }
}

// Convert packed data to unpacked data when accessors are called.
void unpackData(void) {
  if (unpackedReady_) return;

  MiniFloatConverter mfc;
  beam1IntensitiesUnpacked_.resize(beam1IntensitiesPacked_);
  beam2IntensitiesUnpacked_.resize(beam2IntensitiesPacked_);

  for (int i=0; i<beam1Intensities.size(); i++) {
    beam1IntensitiesUnpacked_[i] = mfc.float16to32(beam1IntensitiesPacked_[i]*scaleFactor);
  }
  for (int i=0; i<beam2Intensities.size(); i++) {
    beam2IntensitiesUnpacked_[i] = mfc.float16to32(beam2IntensitiesPacked_[i]*scaleFactor);
  }
  unpackedReady_ = true;
}
       

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo) {
  s << "\nDumping BeamCurrentInfo\n\n";
  s << "  beam1Intensities = ";
  const std::vector<float>& b1int = beamInfo.getBeam1IntensitiesUnpacked();
  for (unsigned int i=0; i<10 && i<b1int.size(); ++i) {
    s << b1int.at(i) << " ";
  }
  s << " ...\n";
  s << "  beam2Intensities = ";
  const std::vector<float>& b2int = beamInfo.getBeam2IntensitiesUnpacked();
  for (unsigned int i=0; i<10 && i<b2int.size(); ++i) {
    s << b2int.at(i) << " ";
  }
  s << " ...\n";

  return s << "\n";
}
