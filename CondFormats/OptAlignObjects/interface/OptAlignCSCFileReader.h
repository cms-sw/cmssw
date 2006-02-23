#ifndef Guard_OptAlignCSCFileReader_h
#define Guard_OptAlignCSCFileReader_h

#include <iosfwd>
#include <vector>
#include <string>
#include <fstream>
#include "boost/any.hpp"

// -*- C++ -*-
//
// Package:    AlignmentTools/OptAlignCSCFileReader
// Class:      OptAlignCSCFileReader.h
// 
/**\class OptAlignCSCFileReader OptAlignCSCFileReader.cc AlignmentTools/OptAlignASCIISource/interface/OptAlignCSCFileReader.h

 Description: test access to the OpticalAlignments as read from the file provided
    by Marcus Hohlmann

 Implementation:

*/
//
// Original Author:  Mike Case
//         Created:  Tue 07 Feb 2006
// $Id: OptAlignCSCFileReader.h,v 1.1 2006/02/16 10:38:56 case Exp $
//
//
class CSVMakeDouble;
class CSVMakeString;

class OptAlignCSCFileReader 
{
  friend class CSVMakeDouble;
  friend class CSVMakeString;

 public:

  OptAlignCSCFileReader ( const std::string& fname );

  virtual ~OptAlignCSCFileReader();

  bool getNames();
  bool getTypes();
  bool next();
  bool getData( std::vector<std::string>& vecStr);

 private:

  std::string type_;
  std::string fileName_;
  std::ifstream tf_;

  void do_makeString(char const* str, char const* end);
  bool parse_strings(char const* str) const;
  std::vector<std::string> strVec_;
  std::string line_;

};

#endif
