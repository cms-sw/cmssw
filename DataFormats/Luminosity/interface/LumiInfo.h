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
#include "DataFormats/Luminosity/interface/LumiConstants.h"

class LumiInfo {
public:
  /// default constructor
  LumiInfo():
  deadtimeFraction_(0)
  { 
    instLumiByBX_.assign(LumiConstants::numBX, 0.0);
    errLumiByBX_.assign(LumiConstants::numBX, 0.0);
    totalLuminosity_=0;
    totalStatError_=0;
  } 
  
  /// constructor with fill
 LumiInfo(float deadtimeFraction,
	  const std::vector<float>& instLumiByBX):
  deadtimeFraction_(deadtimeFraction)
  {
    instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
  }

  /// destructor
  ~LumiInfo(){}

  //Total Luminosity Saved 
  float getTotalLumi() const{return totalLuminosity_;}

  //Statistical Error on total lumi
  float getTotalStatError() const{return totalStatError_;}

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

  void setInstLumi(std::vector<float>& instLumiByBX);

  void setErrLumiBX(std::vector<float>& errLumiByBX);

  // Inst lumi by bunch
  float getInstLumiBX(int bx) const { return instLumiByBX_.at(bx); }
  const std::vector<float>& getInstLumiAllBX() const { return instLumiByBX_; }
  const std::vector<float>& getErrorLumiAllBX() const { return errLumiByBX_; }

  bool isProductEqual(LumiInfo const& next) const;

  //
  //setters
  //

  void setDeadFraction(float deadtimeFraction) { deadtimeFraction_ = deadtimeFraction; }
  //set the total raw luminosity
  void setTotalLumi(float totalLumi){ totalLuminosity_=totalLumi;}
  //set the statistical error
  void setTotalStatError(float statError){ totalStatError_=statError;}

private:
  float totalLuminosity_;
  float totalStatError_;
  float deadtimeFraction_;
  std::vector<float> instLumiByBX_;
  std::vector<float> errLumiByBX_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo);

#endif // DataFormats_Luminosity_LumiInfo_h
