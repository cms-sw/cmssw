#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include <map>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/AlpgenInterface/interface/AlpgenHeader.h"

namespace {
  struct AlpgenParTokens {
    unsigned int	        index;
    std::vector<double>		values;
    std::vector<std::string>	comments;

    bool parse(const std::string &line, bool withIndex);
  };
}

bool AlpgenParTokens::parse(const std::string &line, bool withIndex)
{
  std::string::size_type pos = line.find('!');
  if (pos == std::string::npos)
    return false;

  std::string tmp = boost::trim_copy(line.substr(0, pos));
  boost::split(comments, tmp, boost::algorithm::is_space(),
	       boost::token_compress_on);
  if (comments.empty())
    return false;

  unsigned int i = 0, n = comments.size();
  if (withIndex) {
    std::istringstream ss(comments[i++]);
    ss >> index;
    if (ss.bad() ||
	ss.peek() != std::istringstream::traits_type::eof())
      return false;
  }

  values.clear();
  while(i < n) {
    std::istringstream ss(comments[i++]);
    double value;
    ss >> value;
    if (ss.bad() ||
	ss.peek() != std::istringstream::traits_type::eof())
      return false;

    values.push_back(value);
  }

  tmp = boost::trim_copy(line.substr(pos + 1));
  boost::split(comments, tmp, boost::algorithm::is_space(),
	       boost::token_compress_on);

  return true;
}

bool AlpgenHeader::parse(const std::vector<std::string>::const_iterator &begin,
                         const std::vector<std::string>::const_iterator &end)
{
  std::vector<std::string>::const_iterator line = begin;
  
  // Mimicking Alpgen - read the _unw.par file until you find the "****".
  while(line != end)
    if ((line++)->find("****") != std::string::npos)
      break;
  
  AlpgenParTokens tokens;
  
  // hard process code
  if (line == end || !tokens.parse(*line++, true) ||
      !tokens.values.empty())
    return false;
  ihrd = tokens.index;

  // mc,mb,mt,mw,mz,mh
  if (line == end || !tokens.parse(*line++, false) ||
      tokens.values.size() < 6)
    return false;

  std::copy(tokens.values.begin(), tokens.values.begin() + 6, masses);

  // key - value pairs
  params.clear();
  while(line != end && line->find("****") == std::string::npos) {
    if (!tokens.parse(*line++, true) ||
	tokens.values.size() != 1)
      return false;
    params[(Parameter)tokens.index] = tokens.values[0];
  }
  if (line == end)
    return false;
  else
    line++;

  // cross-section
  if (line == end || !tokens.parse(*line++, false) ||
      tokens.values.size() != 2)
    return false;

  xsec = tokens.values[0];
  xsecErr = tokens.values[1];

  // unweighted events, luminosity
  if (line == end || !tokens.parse(*line++, true) ||
      tokens.values.size() != 1)
    return false;

  nEvents = tokens.index;
  lumi = tokens.values[0];

  return true;
}

// create human-readable representation for all Alpgen parameter indices

#define DEFINE_ALPGEN_PARAMETER(x) { AlpgenHeader::x, #x }

namespace {
  struct AlpgenParameterName {
    AlpgenHeader::Parameter	index;
    const char*                 name;

    inline bool operator == (AlpgenHeader::Parameter index) const
    { return this->index == index; }
  } 
  
  static const alpgenParameterNames[] = {
    DEFINE_ALPGEN_PARAMETER(ih2),
    DEFINE_ALPGEN_PARAMETER(ebeam),
    DEFINE_ALPGEN_PARAMETER(ndns),
    DEFINE_ALPGEN_PARAMETER(iqopt),
    DEFINE_ALPGEN_PARAMETER(qfac),
    DEFINE_ALPGEN_PARAMETER(ickkw),
    DEFINE_ALPGEN_PARAMETER(ktfac),
    DEFINE_ALPGEN_PARAMETER(njets),
    DEFINE_ALPGEN_PARAMETER(ihvy),
    DEFINE_ALPGEN_PARAMETER(ihvy2),
    DEFINE_ALPGEN_PARAMETER(nw),
    DEFINE_ALPGEN_PARAMETER(nz),
    DEFINE_ALPGEN_PARAMETER(nh),
    DEFINE_ALPGEN_PARAMETER(nph),
    DEFINE_ALPGEN_PARAMETER(ptjmin),
    DEFINE_ALPGEN_PARAMETER(ptbmin),
    DEFINE_ALPGEN_PARAMETER(ptcmin),
    DEFINE_ALPGEN_PARAMETER(ptlmin),
    DEFINE_ALPGEN_PARAMETER(metmin),
    DEFINE_ALPGEN_PARAMETER(ptphmin),
    DEFINE_ALPGEN_PARAMETER(etajmax),
    DEFINE_ALPGEN_PARAMETER(etabmax),
    DEFINE_ALPGEN_PARAMETER(etacmax),
    DEFINE_ALPGEN_PARAMETER(etalmax),
    DEFINE_ALPGEN_PARAMETER(etaphmax),
    DEFINE_ALPGEN_PARAMETER(drjmin),
    DEFINE_ALPGEN_PARAMETER(drbmin),
    DEFINE_ALPGEN_PARAMETER(drcmin),
    DEFINE_ALPGEN_PARAMETER(drlmin),
    DEFINE_ALPGEN_PARAMETER(drphjmin),
    DEFINE_ALPGEN_PARAMETER(drphlmin),
    DEFINE_ALPGEN_PARAMETER(drphmin),
    DEFINE_ALPGEN_PARAMETER(mllmin),
    DEFINE_ALPGEN_PARAMETER(mllmax),
    DEFINE_ALPGEN_PARAMETER(iseed1),
    DEFINE_ALPGEN_PARAMETER(iseed2),
    DEFINE_ALPGEN_PARAMETER(itopprc),
    DEFINE_ALPGEN_PARAMETER(cluopt),
    DEFINE_ALPGEN_PARAMETER(iseed3),
    DEFINE_ALPGEN_PARAMETER(iseed4)
  };
} // anonymous namespace

std::string AlpgenHeader::parameterName(Parameter index)
{
  static const unsigned int size = sizeof alpgenParameterNames /
    sizeof alpgenParameterNames[0];

  const AlpgenParameterName *pos =
    std::find(alpgenParameterNames,
	      alpgenParameterNames + size, index);

  if (pos != alpgenParameterNames + size)
    return pos->name;

  std::ostringstream ss;
  ss << "unknown " << (int)index;
  return ss.str();
}
