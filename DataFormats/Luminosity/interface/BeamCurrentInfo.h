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
 
#include <vector>
#include <iosfwd>
#include <string>
class BeamCurrentInfo {
 public:
  static const int numOrbits_ = 262144; // number of orbits per LS (2^18)
  static const unsigned int numBX_ = 3564; // number of BX per orbit
  
  /// default constructor
  BeamCurrentInfo() {
    beam1Intensities_.assign(numBX_, 0.0);
    beam2Intensities_.assign(numBX_, 0.0);
  } 
  
  /// constructor with fill
  BeamCurrentInfo(const std::vector<float>& beam1Intensities,
                  const std::vector<float>& beam2Intensities) {
    beam1Intensities_.assign(beam1Intensities.begin(), beam1Intensities.end());
    beam2Intensities_.assign(beam2Intensities.begin(), beam2Intensities.end());
  }
  
  /// destructor
  ~BeamCurrentInfo(){}
  
  // Beam intensities by bunch
  float getBeam1IntensityBX(int bx) const { return beam1Intensities_.at(bx); }
  const std::vector<float>& getBeam1Intensities() const { return beam1Intensities_; }
  float getBeam2IntensityBX(int bx) const { return beam2Intensities_.at(bx); }
  const std::vector<float>& getBeam2Intensities() const { return beam2Intensities_; }

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
  std::vector<float> beam1Intensities_;
  std::vector<float> beam2Intensities_;
}; 

std::ostream& operator<<(std::ostream& s, const BeamCurrentInfo& beamInfo);

#endif // DataFormats_Luminosity_BeamCurrentInfo_h
