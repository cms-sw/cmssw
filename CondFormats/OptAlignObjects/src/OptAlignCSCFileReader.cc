#include "CondFormats/OptAlignObjects/interface/OptAlignCSCFileReader.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"

// Boost parser, spirit, for parsing the std::vector elements.
#include "boost/spirit/core.hpp"
#include "boost/any.hpp"

#include <string>
#include <vector>
#include <algorithm>
//#include <fstream>
#include <iostream>
//#include <sstream>
#include <stdexcept>

namespace boost { namespace spirit {} } using namespace boost::spirit;

  
struct CSVMakeString
{
  void operator() (char const* str, char const* end) const
  {
    oas_->do_makeString(str, end);
  }
  
  CSVMakeString() : oas_(0) {}
  explicit CSVMakeString ( const OptAlignCSCFileReader* oas ) : oas_(oas) {}
  
  OptAlignCSCFileReader * oas_;
  
};

OptAlignCSCFileReader::OptAlignCSCFileReader ( const std::string& fname ) : fileName_(fname), tf_(fname.c_str()) 
  
{ 
  // extract the "type" from the filename.
    size_t fit = fname.rfind("/");
    size_t eit = fname.rfind(".");
    size_t j = 0;
    for ( size_t i = fit; i < eit; i++ ) {
      type_[j++] = fname[i];
    }
    std::cout << type_ << std::endl;
}
  
OptAlignCSCFileReader::~OptAlignCSCFileReader () { }


bool OptAlignCSCFileReader::next ( ) {

  bool toReturn(false);
  strVec_.clear();
  line_.clear();
  if ( !tf_.eof() ) {
    toReturn = getline(tf_, line_);
    //    std::cout << "raw line [" << line_ << "]" << std::endl;
    //    std::cout << "line_.size()=" << line_.size() << std::endl;
    if ( line_.size() > 0 ) {
      toReturn = parse_strings(line_.c_str());
    }
  }
  if ( toReturn ) std::cout << "about to return true from next() " << std::endl;
  return toReturn;
}


bool OptAlignCSCFileReader::parse_strings(char const* str) const
{
  CSVMakeString makeString(this);
  return parse(str,
	       ((+(anychar_p - ','))[makeString] 
		>> *(',' >> (+(anychar_p - ','))[makeString]))
	       , space_p).full;
}

void OptAlignCSCFileReader::do_makeString(char const* str, char const* end)
{
  std::string ts(str, end);
  strVec_.push_back(ts);
}


bool OptAlignCSCFileReader::getData( std::vector<std::string>& vecStr) {
  vecStr = strVec_;
  return true;
}
