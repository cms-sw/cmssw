//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprStringParser.hh,v 1.3 2006/11/26 02:04:30 narsky Exp $
//
// Description:
//      Class SprStringParser :
//         Parses an input string of names separated by commas into 
//         a vector, with vectors separated from each other by colons.
//         For example, '1,2:3:4,5' is parsed into 3 vectors:
//            v1 = {1,2}    v2 = {3}    v3 = {4,5}
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2004              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprStringParser_HH
#define _SprStringParser_HH

#include <vector>
#include <string>

class SprStringParser
{
public:
  virtual ~SprStringParser() {}

  SprStringParser() {}

  static void parseToStrings(const char* input, 
			     std::vector<std::vector<std::string> >& output);

  static void parseToInts(const char* input, 
			  std::vector<std::vector<int> >& output);

  static void parseToDoubles(const char* input, 
			     std::vector<std::vector<double> >& output);
};

#endif
