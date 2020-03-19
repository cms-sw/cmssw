#ifndef DataFormats_Luminosity_LumiInfoRunHeader_h
#define DataFormats_Luminosity_LumiInfoRunHeader_h

/** \class LumiInfoRunHeader
 *
 * LumiInfoRunHeader contains LumiInfo data which remains valid
 * during the whole run.
 *
 * This is an updated version of LumiSummaryRunHeader which drops
 * the L1/HLT trigger names and adds the filling scheme information.
 *
 * \author Matevz Tadel, updated by Paul Lujan
 * \date   2011-02-22, updated 2014-09-10
 *
 */

#include <string>
#include <bitset>
#include "DataFormats/Luminosity/interface/LumiConstants.h"

class LumiInfoRunHeader {
public:
  //----------------------------------------------------------------

  /// Default constructor.
  LumiInfoRunHeader() {}

  /// Constructor with lumi provider, filling scheme name, and filling scheme.
  LumiInfoRunHeader(std::string& lumiProvider,
                    std::string& fillingSchemeName,
                    std::bitset<LumiConstants::numBX>& fillingScheme);

  /// Destructor.
  ~LumiInfoRunHeader() {}

  /// Product compare function.
  bool isProductEqual(LumiInfoRunHeader const& o) const;

  //----------------------------------------------------------------

  /// Set lumi provider.
  void setLumiProvider(const std::string& lumiProvider) { lumiProvider_ = lumiProvider; }

  /// Set filling scheme name.
  void setFillingSchemeName(const std::string& fillingSchemeName) { fillingSchemeName_ = fillingSchemeName; }

  /// Set filling scheme.
  void setFillingScheme(const std::bitset<LumiConstants::numBX>& fillingScheme);

  //----------------------------------------------------------------

  /// Get lumi provider.
  std::string getLumiProvider() const { return lumiProvider_; }

  /// Get filling scheme name.
  std::string getFillingSchemeName() const { return fillingSchemeName_; }

  /// Get filling scheme for given bunch.
  bool getBunchFilled(unsigned int bunch) const { return fillingScheme_[bunch]; }

  /// Get full filling scheme.
  const std::bitset<LumiConstants::numBX>& getFillingScheme() const { return fillingScheme_; }

  /// Get bunch spacing (in ns).
  int getBunchSpacing() const { return bunchSpacing_; }

  //----------------------------------------------------------------

private:
  std::string lumiProvider_;                         // string with name of lumi provider
  std::string fillingSchemeName_;                    // name of filling scheme
  std::bitset<LumiConstants::numBX> fillingScheme_;  // filling scheme
  int bunchSpacing_;

  void setBunchSpacing();
};

#endif
