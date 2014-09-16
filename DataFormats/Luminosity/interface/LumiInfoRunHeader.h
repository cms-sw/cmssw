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

#include <vector>
#include <string>

class LumiInfoRunHeader
{
public:
  typedef std::vector<bool> vbool_t;

  //----------------------------------------------------------------

  /// Default constructor.
  LumiInfoRunHeader() {}

  /// Constructor with lumi provider, filling scheme name, and filling scheme.
  LumiInfoRunHeader(std::string& lumiProvider, std::string& fillingSchemeName, vbool_t& fillingScheme);

  /// Destructor.
  ~LumiInfoRunHeader() {}

  /// Product compare function.
  bool isProductEqual(LumiInfoRunHeader const& o) const;

  //----------------------------------------------------------------

  /// Set lumi provider.
  void setLumiProvider(const std::string& lumiProvider) { m_lumiProvider = lumiProvider; }

  /// Set filling scheme name.
  void setFillingSchemeName(const std::string& fillingSchemeName) { m_fillingSchemeName = fillingSchemeName; }

  /// Set filling scheme.
  void setFillingScheme(const vbool_t& fillingScheme);

  //----------------------------------------------------------------

  /// Get lumi provider.
  std::string getLumiProvider() const { return m_lumiProvider; }

  /// Get filling scheme name.
  std::string getFillingSchemeName() const { return m_fillingSchemeName; }

  /// Get filling scheme for given bunch.
  bool getBunchIsFilled(unsigned int bunch) const { return m_fillingScheme.at(bunch); }

  /// Get full filling scheme.
  const vbool_t& getFillingScheme() const { return m_fillingScheme; }

  //----------------------------------------------------------------

private:
  std::string m_lumiProvider;       // string with name of lumi provider
  std::string m_fillingSchemeName;  // name of filling scheme
  vbool_t m_fillingScheme;          // filling scheme
};

#endif
