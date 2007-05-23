//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprAbsReader.hh,v 1.2 2006/10/19 21:27:52 narsky Exp $
//
// Description:
//      Class SprAbsReader :
//         read in data info from a file and compose SprData
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
 
#ifndef _SprAbsReader_HH
#define _SprAbsReader_HH

#include <set>
#include <string>

class SprAbsFilter;

class SprAbsReader
{
public:
  virtual ~SprAbsReader() {}

  SprAbsReader() {}

  // Choose variables (by name) to read in. By default all variables
  // are chosen. chooseAllBut() chooses all variables except those specified.
  // chooseAll() method should be used to reset the choice
  // made by chooseVars() and chooseAllBut(). 
  // chooseVars() must be called before read().
  virtual bool chooseVars(const std::set<std::string>& vars) = 0;
  virtual bool chooseAllBut(const std::set<std::string>& vars) = 0;
  virtual void chooseAll() = 0;

  // Read data from a file.  Returns a newly-allocated filter, which the
  // caller is responsible for deleting, or NULL on error.
  virtual SprAbsFilter* read(const char* filename) = 0;
};

#endif
