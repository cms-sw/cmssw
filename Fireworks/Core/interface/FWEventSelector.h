// -*- C++ -*-
#ifndef Fireworks_Core_FWEventSelector_h
#define Fireworks_Core_FWEventSelector_h
//
// Package:     newVersion
// Class  :     FWEventSelector
// $Id: CmsShowNavigator.h,v 1.21 2009/08/18 19:03:30 amraktad Exp $
//

// system include files
#include <string>
#include <vector>

struct FWEventSelector{
  FWEventSelector(const char* iSelection, const char* iTitle, bool enable):
    selection(iSelection), title(iTitle), enabled(enable), removed(false){}
  FWEventSelector():
    enabled(false), removed(false){}
  void enable(bool flag){
    enabled = flag;
  }
  void remove(){
    removed = true;
  }
  std::string selection;
  std::string title;
  bool enabled;
  bool removed;
};
#endif
