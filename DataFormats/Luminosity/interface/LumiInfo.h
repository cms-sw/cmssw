#ifndef DataFormats_Luminosity_LumiInfo_h
#define DataFormats_Luminosity_LumiInfo_h

/** 
 * \class LumiInfo
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
 * \version    October 2017 by Chris Palmer and Sam Higginbotham for PCC projects
 * 
 *
 */

#include <vector>
#include <iosfwd>
#include <string>
#include "DataFormats/Luminosity/interface/LumiConstants.h"

class LumiInfo {
public:
  /** 
     * default constructor
     */
  LumiInfo() : deadtimeFraction_(0) {
    instLumiByBX_.assign(LumiConstants::numBX, 0.0);
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    totalInstLuminosity_ = 0;
    totalInstLumiStatErr_ = 0;
  }

  /** 
     * constructor with fill; if total algo is the same as summing
     */
  LumiInfo(float deadtimeFraction, const std::vector<float>& instLumiByBX)
      : deadtimeFraction_(deadtimeFraction), instLumiByBX_(instLumiByBX) {
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    setTotalInstToBXSum();
    totalInstLumiStatErr_ = 0;
  }

  /** 
     * constructor with fill; if total algo DIFFERS from summing
     */
  LumiInfo(float deadtimeFraction, const std::vector<float>& instLumiByBX, float totalInstLumi)
      : deadtimeFraction_(deadtimeFraction), totalInstLuminosity_(totalInstLumi), instLumiByBX_(instLumiByBX) {
    instLumiStatErrByBX_.assign(LumiConstants::numBX, 0.0);
    totalInstLumiStatErr_ = 0;
  }

  /** 
     * constructor with fill; if total algo DIFFERS from summing and adding including stats
     */
  LumiInfo(float deadtimeFraction,
           const std::vector<float>& instLumiByBX,
           float totalInstLumi,
           const std::vector<float>& instLumiErrByBX,
           float totalInstLumiErr)
      : deadtimeFraction_(deadtimeFraction),
        totalInstLuminosity_(totalInstLumi),
        totalInstLumiStatErr_(totalInstLumiErr),
        instLumiByBX_(instLumiByBX),
        instLumiStatErrByBX_(instLumiErrByBX) {}

  /** 
     * destructor
     */
  ~LumiInfo() {}

  //
  // all getters
  //

  /** 
     *  Returns total instantanious luminosity in hz/uB
     */
  float getTotalInstLumi() const { return totalInstLuminosity_; }
  /** 
     *  Returns statistical error on total instantanious luminosity in hz/uB
     */
  float getTotalInstStatError() const { return totalInstLumiStatErr_; }
  /** 
     *  Returns instantaneous luminosity of all bunches
     */
  const std::vector<float>& getInstLumiAllBX() const { return instLumiByBX_; }
  /** 
     * Returns statistical error of instantaneous luminosity for all bunches
     */
  const std::vector<float>& getErrorLumiAllBX() const { return instLumiStatErrByBX_; }
  /** 
     *  Returns instantaneous luminosity of one bunch
     */
  float getInstLumiBX(int bx) const { return instLumiByBX_.at(bx); }
  /** 
     * Deadtime fraction
     */
  float getDeadFraction() const { return deadtimeFraction_; }
  /** 
     * Livetime fraction (1-deadtime frac)
     */
  float getLiveFraction() const { return 1 - deadtimeFraction_; }

  //
  // all setters
  //

  /** 
     * Set the deadtime fraction
     */
  void setDeadFraction(float deadtimeFraction) { deadtimeFraction_ = deadtimeFraction; }
  /** 
     * Set total instantanious luminosity in hz/uB
     */
  void setTotalInstLumi(float totalLumi) { totalInstLuminosity_ = totalLumi; }
  /** 
     *  Set statistical error on total instantanious luminosity in hz/uB
     */
  void setTotalInstStatError(float statError) { totalInstLumiStatErr_ = statError; }
  /** 
     *  Set statistical error of instantaneous luminosity for all bunches
     */
  void setInstLumiAllBX(std::vector<float>& instLumiByBX);
  /** 
     *  Set statistical error of instantaneous luminosity for all bunches
     */
  void setErrorLumiAllBX(std::vector<float>& errLumiByBX);

  /** 
     *  Resets totalInstLuminosity_ to be the sum of instantaneous 
     *  luminosity 
     */
  void setTotalInstToBXSum();

  /** 
     *  Returns the sum of the instantaneous luminosity in Hz/uB,
     *  which not always the same as totalInstLuminosity_.
     */
  float instLuminosityBXSum() const;
  /** 
     * Integrated (delivered) luminosity (in ub^-1)
     */
  float integLuminosity() const;
  /** 
     * Recorded (integrated) luminosity (in ub^-1)
     * (==integLuminosity * (1-deadtimeFraction))
     */
  float recordedLuminosity() const;
  /** 
     * lumi section length in seconds = numorbits*3564*25e-09
     */
  float lumiSectionLength() const;
  /** 
     *  This method checks if all the essential values of this LumiInfo are
     *  the same as the ones in the LumiInfo given as an argument.
     */
  bool isProductEqual(LumiInfo const& next) const;

private:
  float deadtimeFraction_;
  float totalInstLuminosity_;
  float totalInstLumiStatErr_;
  std::vector<float> instLumiByBX_;
  std::vector<float> instLumiStatErrByBX_;
};

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo);

#endif  // DataFormats_Luminosity_LumiInfo_h
