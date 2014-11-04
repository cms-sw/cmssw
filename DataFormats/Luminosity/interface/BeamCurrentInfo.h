#ifndef DataFormats_Luminosity_BeamCurrentInfo_h
#define DataFormats_Luminosity_BeamCurrentInfo_h
 
/** \class BeamCurrentInfo
 *
 *
 * BeamCurrentInfo has been created by splitting off the
 * beam current information from LumiInfo. See LumiInfo
 * for more details.
 *
 * \author Valerie Halyo
 *         David Dagenhart
 *         Zhen Xie
 *         Paul Lujan
 * \version   October 21, 2014
 *
 ************************************************************/

// To preserve space, this class stores information in uint16_t using
// libminifloat. The beamIntensitiesUnpacked variables contain the expanded
// float versions, and the beamIntensitiesPacked variables contain the
// 16-bit versions. The intensities are also divided by 1e10 during packing
// so that the values are near 1 (this is not strictly necessary since the
// values are still ~1e11 and so within the limit of precision, but this
// should keep us safer).

#include <vector>
#include <iosfwd>
#include <string>
#include "DataFormats/PatCandidates/interface/libminifloat.h"

class BeamCurrentInfo {
 public:
  static const int numOrbits_ = 262144; // number of orbits per LS (2^18)
  static const unsigned int numBX_ = 3564; // number of BX per orbit
  static const float scaleFactor; // factor to scale data by when packing/unpacking
  
  /// default constructor
  BeamCurrentInfo() {
    beam1IntensitiesUnpacked_.assign(numBX_, 0.0);
    beam2IntensitiesUnpacked_.assign(numBX_, 0.0);
    beam1IntensitiesPacked_.assign(numBX_, 0.0);
    beam2IntensitiesPacked_.assign(numBX_, 0.0);
    unpackedReady_ = true;
  } 
  
  /// constructor with fill
  BeamCurrentInfo(const std::vector<float>& beam1Intensities,
                  const std::vector<float>& beam2Intensities) {
    beam1IntensitiesUnpacked_.assign(beam1Intensities.begin(), beam1Intensities.end());
    beam2IntensitiesUnpacked_.assign(beam2Intensities.begin(), beam2Intensities.end());
    packData();
  }
  
  /// destructor
  ~BeamCurrentInfo(){}
  
  // Beam intensities by bunch, or all
  float getBeam1IntensityBX(int bx) const;
  const std::vector<float>& getBeam1Intensities() const;
  float getBeam2IntensityBX(int bx) const;
  const std::vector<float>& getBeam2Intensities() const;

  bool isProductEqual(BeamCurrentInfo const& next) const;

  //
  //setters
  //

  // fill beam intensities
  void fillBeamIntensities(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities);
  // synonym for above
  void fill(const std::vector<float>& beam1Intensities,
	    const std::vector<float>& beam2Intensities);
  
 private:
  mutable std::vector<uint16_t> beam1IntensitiesPacked_;
  mutable std::vector<uint16_t> beam2IntensitiesPacked_;
  std::vector<float> beam1IntensitiesUnpacked_;
  std::vector<float> beam2IntensitiesUnpacked_;
  void packData(void);
  void unpackData(void);
  bool unpackedReady_;
}; 

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo);

#endif // DataFormats_Luminosity_BeamCurrentInfo_h
