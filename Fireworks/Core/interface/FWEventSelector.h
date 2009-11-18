// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: FWEventSelector.h,v 1.1 2009/09/29 19:20:18 dmytro Exp $
//

// system include files
#include <string>

struct FWEventSelector{
  FWEventSelector(const char* iSelection, const char* iTitle, bool enable):
    selection(iSelection), title(iTitle), enabled(enable), removed(false){}
  FWEventSelector():
    enabled(false), removed(false){}

  std::string selection;
  std::string title;
  bool enabled;
  bool removed;
};
#endif
