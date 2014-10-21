#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"

#include <iomanip>
#include <ostream>
#include <iostream>

bool BeamCurrentInfo::isProductEqual(BeamCurrentInfo const& next) const {
  return (beam1Intensities_ == next.beam1Intensities_ &&
	  beam2Intensities_ == next.beam2Intensities_);
}

void BeamCurrentInfo::fillBeamIntensities(const std::vector<float>& beam1Intensities,
					  const std::vector<float>& beam2Intensities) {
  beam1Intensities_.assign(beam1Intensities.begin(), beam1Intensities.end());
  beam2Intensities_.assign(beam2Intensities.begin(), beam2Intensities.end());
}

void BeamCurrentInfo::fill(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities) {
  fillBeamIntensities(beam1Intensities, beam2Intensities);
}

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo) {
  s << "\nDumping BeamCurrentInfo\n\n";
  s << "  beam1Intensities = ";
  const std::vector<float>& b1int = beamInfo.getBeam1Intensities();
  for (unsigned int i=0; i<10 && i<b1int.size(); ++i) {
    s << b1int.at(i) << " ";
  }
  s << " ...\n";
  s << "  beam2Intensities = ";
  const std::vector<float>& b2int = beamInfo.getBeam2Intensities();
  for (unsigned int i=0; i<10 && i<b2int.size(); ++i) {
    s << b2int.at(i) << " ";
  }
  s << " ...\n";

  return s << "\n";
}
