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
 * \update    October 2017 by Chris Palmer and Sam Higginbotham for PCC projects
 * 
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
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    totalInstLuminosity_=0;
    totalInstLumiStatErr_=0;
  } 
  
  /// constructor with fill; if total algo is the same as summing
 LumiInfo(float deadtimeFraction, const std::vector<float>& instLumiByBX):
  deadtimeFraction_(deadtimeFraction)
  {
    instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    setTotalInstToBXSum() ;
    totalInstLumiStatErr_=0;
  }

  /// constructor with fill; if total algo DIFFERS from summing
 LumiInfo(float deadtimeFraction, const std::vector<float>& instLumiByBX, float totalInstLumi):
  deadtimeFraction_(deadtimeFraction)
  {
    instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    totalInstLuminosity_=totalInstLumi;
    totalInstLumiStatErr_=0;
  }

  /// constructor with fill; if total algo DIFFERS from summing and adding including stats
 LumiInfo(float deadtimeFraction, const std::vector<float>& instLumiByBX, float totalInstLumi,
    const std::vector<float>& instLumiErrByBX, float totalInstLumiErr):
  deadtimeFraction_(deadtimeFraction)
  {
    instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
    instLumiStatErrByBX_.assign(instLumiErrByBX.begin(), instLumiErrByBX.end());
    totalInstLuminosity_=totalInstLumi;
    totalInstLumiStatErr_=totalInstLumiErr;
  }

  /// destructor
  ~LumiInfo(){}

  //Total Luminosity Saved 
  float getTotalInstLumi() const{return totalInstLuminosity_;}

  //Statistical Error on total lumi
  float getTotalInstStatError() const{return totalInstLumiStatErr_;}

  // Instantaneous luminosity (in Hz/ub)
  float instLuminosityBXSum() const;
  
  void setTotalInstToBXSum() ;

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
  const std::vector<float>& getErrorLumiAllBX() const { return instLumiStatErrByBX_; }

  bool isProductEqual(LumiInfo const& next) const;

  //
  //setters
  //

  void setDeadFraction(float deadtimeFraction) { deadtimeFraction_ = deadtimeFraction; }
  //set the total raw luminosity
  void setTotalInstLumi(float totalLumi){ totalInstLuminosity_=totalLumi;}
  //set the statistical error
  void setTotalStatError(float statError){ totalInstLumiStatErr_=statError;}

private:
  float totalInstLuminosity_;
  float totalInstLumiStatErr_;
  float deadtimeFraction_;
  std::vector<float> instLumiByBX_;
  std::vector<float> instLumiStatErrByBX_;
}; 

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo);

#endif // DataFormats_Luminosity_LumiInfo_h
