#ifndef DataFormats_Luminosity_LumiSummaryRunHeader_h
#define DataFormats_Luminosity_LumiSummaryRunHeader_h
 
/** \class LumiSummaryRunHeader
 *
 * LumiSummaryRunHeader contains LumiSummary data which remains valid
 * during the whole run.

 * 1. Vectors of L1 and HLT trigger / path names. LumiSummary uses
 * integer indices into these two vectors to minimize disk-usage in
 * highly selective skim files.
 *
 * \author Matevz Tadel
 * \date   2011-02-22
 *
 * $Id: LumiSummaryRunHeader.h,v 1.1 2011/02/22 16:23:57 matevz Exp $
 */

#include <vector>
#include <string>

class LumiSummaryRunHeader
{
public:
  typedef std::vector<std::string> vstring_t;

  //----------------------------------------------------------------

  /// Default constructor.
  LumiSummaryRunHeader() {}

  /// Constructor with names.
  /// Vectors are swapped so they are empty on return.
  LumiSummaryRunHeader(vstring_t& l1names, vstring_t& hltnames);

  /// Destructor.
  ~LumiSummaryRunHeader() {}

  /// Product compare function.
  bool isProductEqual(LumiSummaryRunHeader const& o) const;

  //----------------------------------------------------------------

  /// Set L1 name vector.
  void setL1Names(const vstring_t& l1names);

  /// Set HLT name vector.
  void setHLTNames(const vstring_t& hltnames);

  /// Swap L1 name vector.
  void swapL1Names(vstring_t& l1names);

  /// Swap HLT name vector.
  void swapHLTNames(vstring_t& hltnames);

  //----------------------------------------------------------------

  /// Get L1 name at given position.
  std::string getL1Name(unsigned int idx) const { return m_l1Names.at(idx); }

  /// Get HLT name at given position.
  std::string getHLTName(unsigned int idx) const { return m_hltNames.at(idx); }

  /// Get L1 name vector.
  const vstring_t& getL1Names(vstring_t& l1names) const { return m_l1Names; }

  /// Get HLT name vector.
  const vstring_t& getHLTNames(vstring_t& hltnames) const { return m_hltNames; }

  /// Get index of given L1 trigger-name. Returns -1 if not found.
  unsigned int getL1Index(const std::string& name) const;

  /// Get index of given HLT path-name. Returns -1 if not found.
  unsigned int getHLTIndex(const std::string& name) const;

  //----------------------------------------------------------------

private:
  vstring_t m_l1Names;  // L1 trigger-name vector.
  vstring_t m_hltNames; // HLT path-name vector.
};

#endif
