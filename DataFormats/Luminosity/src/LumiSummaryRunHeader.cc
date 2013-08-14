// $Id: LumiSummaryRunHeader.cc,v 1.1 2011/02/22 16:23:57 matevz Exp $

#include "DataFormats/Luminosity/interface/LumiSummaryRunHeader.h"

LumiSummaryRunHeader::LumiSummaryRunHeader(vstring_t& l1names, vstring_t& hltnames)
{
  m_l1Names.swap(l1names);
  m_hltNames.swap(hltnames);
}

bool LumiSummaryRunHeader::isProductEqual(LumiSummaryRunHeader const& o) const
{
  return m_l1Names == o.m_l1Names && m_hltNames == o.m_hltNames;
}

//==============================================================================

void LumiSummaryRunHeader::setL1Names(const vstring_t& l1names)
{
  m_l1Names.assign(l1names.begin(), l1names.end());
}

void LumiSummaryRunHeader::setHLTNames(const vstring_t& hltnames)
{
  m_hltNames.assign(hltnames.begin(), hltnames.end());
}

void LumiSummaryRunHeader::swapL1Names(vstring_t& l1names)
{
  m_l1Names.swap(l1names);
}

void LumiSummaryRunHeader::swapHLTNames(vstring_t& hltnames)
{
  m_hltNames.swap(hltnames);
}

//==============================================================================

unsigned int LumiSummaryRunHeader::getL1Index(const std::string& name) const
{
  const unsigned int size = m_l1Names.size();
  for (unsigned int i = 0; i < size; ++i)
  {
    if (m_l1Names[i] == name) return i;
  }
  return -1;
}

unsigned int LumiSummaryRunHeader::getHLTIndex(const std::string& name) const
{
  const unsigned int size = m_hltNames.size();
  for (unsigned int i = 0; i < size; ++i)
  {
    if (m_hltNames[i] == name) return i;
  }
  return -1;
}
