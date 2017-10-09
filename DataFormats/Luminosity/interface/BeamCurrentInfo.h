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
// so that the values are near 1, to avoid running into the limits of
// this packing.

#include <vector>
#include <iosfwd>
#include <string>
#include <stdint.h>
#include "DataFormats/Luminosity/interface/LumiConstants.h"

class BeamCurrentInfo {
 public:
  static const float scaleFactor; // factor to scale data by when packing/unpacking
  
  /// default constructor
  BeamCurrentInfo() {
    beam1IntensitiesUnpacked_.assign(LumiConstants::numBX, 0.0);
    beam2IntensitiesUnpacked_.assign(LumiConstants::numBX, 0.0);
    beam1IntensitiesPacked_.assign(LumiConstants::numBX, 0);
    beam2IntensitiesPacked_.assign(LumiConstants::numBX, 0);
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

  // Get packed intensities. Only use this if you really know that this is what you want!
  const std::vector<uint16_t>& getBeam1IntensitiesPacked() const { return beam1IntensitiesPacked_; }
  const std::vector<uint16_t>& getBeam2IntensitiesPacked() const { return beam2IntensitiesPacked_; }
  
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
  

  // used by ROOT iorules
  static void unpackData( const std::vector<uint16_t>& packed, std::vector<float>& unpacked);

 private:
  std::vector<uint16_t> beam1IntensitiesPacked_;
  std::vector<uint16_t> beam2IntensitiesPacked_;
  std::vector<float> beam1IntensitiesUnpacked_;
  std::vector<float> beam2IntensitiesUnpacked_;
  void packData();
  void unpackData();
}; 

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo);

#endif // DataFormats_Luminosity_BeamCurrentInfo_h
