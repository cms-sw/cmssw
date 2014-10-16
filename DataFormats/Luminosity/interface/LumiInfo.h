#ifndef DataFormats_Luminosity_LumiInfo_h
#define DataFormats_Luminosity_LumiInfo_h
 
/** \class LumiInfo
 *
 *
 * LumiInfo has been created by merging the content of
 * the old LumiSummary and LumiDetails classes to streamline
 * the lumi information. Many old member variables have been
 * removed.
 * 
 *
 * \author Valerie Halyo
 *         David Dagenhart
 *         Zhen Xie
 *         Paul Lujan
 * \version   1st Version June 7 2007, merged September 10 2014
 *
 ************************************************************/
 
#include <vector>
#include <iosfwd>
#include <string>
class LumiInfo {
public:
  static const int numOrbits_ = 262144; // number of orbits per LS (2^18)
  static const unsigned int numBX_ = 3564; // number of BX per orbit

  /// default constructor
  LumiInfo():
    deadtimeFraction_(0)
  { 
    instLumiByBX_.assign(numBX_, 0.0);
    beam1Intensities_.assign(numBX_, 0.0);
    beam2Intensities_.assign(numBX_, 0.0);
  } 
    
  /// constructor with fill
  LumiInfo(float deadtimeFraction,
	   const std::vector<float>& instLumiByBX,
	   const std::vector<float>& beam1Intensities,
	   const std::vector<float>& beam2Intensities):
    deadtimeFraction_(deadtimeFraction),
    instLumiByBX_(instLumiByBX),
    beam1Intensities_(beam1Intensities),
    beam2Intensities_(beam2Intensities)
  { }

  /// destructor
  ~LumiInfo(){}

  // Instantaneous luminosity (in Hz/ub)
  float instLuminosity() const;

  // Integrated (delivered) luminosity (in ub^-1)
  float integLuminosity() const;

  // Recorded (integrated) luminosity (in ub^-1)
  // (==integLuminosity * (1-deadtimeFraction))
  float recordedLuminosity() const;

  // Deadtime/livetime fraction
  float deadFraction() const { return deadtimeFraction_; }
  float liveFraction() const { return 1-deadtimeFraction_; }

  // lumi section length in seconds = numorbits*3564*25e-09
  float lumiSectionLength() const;

  // Inst lumi by bunch
  float getInstLumiBX(int bx) const { return instLumiByBX_.at(bx); }
  const std::vector<float>& getInstLumiAllBX() const { return instLumiByBX_; }

  // Beam intensities by bunch
  float getBeam1IntensityBX(int bx) const { return beam1Intensities_.at(bx); }
  const std::vector<float>& getBeam1Intensities() const { return beam1Intensities_; }
  float getBeam2IntensityBX(int bx) const { return beam2Intensities_.at(bx); }
  const std::vector<float>& getBeam2Intensities() const { return beam2Intensities_; }

  bool isProductEqual(LumiInfo const& next) const;

  //
  //setters
  //

  void setDeadFraction(float deadtimeFraction) { deadtimeFraction_ = deadtimeFraction; }
  // fill all info
  void fill(const std::vector<float>& instLumiByBX,
	    const std::vector<float>& beam1Intensities,
	    const std::vector<float>& beam2Intensities);
  // fill just inst lumi
  void fillInstLumi(const std::vector<float>& instLumiByBX);

  // fill just beam intensities
  void fillBeamIntensities(const std::vector<float>& beam1Intensities,
			   const std::vector<float>& beam2Intensities);

private:
  float deadtimeFraction_;
  std::vector<float> instLumiByBX_;
  std::vector<float> beam1Intensities_;
  std::vector<float> beam2Intensities_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo);

#endif // DataFormats_Luminosity_LumiInfo_h
