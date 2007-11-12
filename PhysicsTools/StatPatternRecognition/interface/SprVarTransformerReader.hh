// File and Version Information:
//      $Id: SprVarTransformerReader.hh,v 1.1 2007/11/07 00:56:14 narsky Exp $
//
// Description:
//      Class SprVarTransformerReader :
//          Reads trained variable transformers.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2007              California Institute of Technology
//------------------------------------------------------------------------
 
#ifndef _SprVarTransformerReader_HH
#define _SprVarTransformerReader_HH

#include <iostream>
#include <vector>
#include <string>

class SprAbsVarTransformer;
class SprPCATransformer;


class SprVarTransformerReader
{
public:
  virtual ~SprVarTransformerReader() {}

  SprVarTransformerReader() {}

  static SprAbsVarTransformer* read(std::istream& is);

  static SprAbsVarTransformer* read(const char* filename);

private:
  // read vars
  static bool readVars(std::istream& is, unsigned& nLine, 
		       std::vector<std::string>& oldVars,
		       std::vector<std::string>& newVars);

  // specific transformers
  static SprPCATransformer* readPCATransformer(std::istream& is, 
					       unsigned& nLine);
};

#endif
