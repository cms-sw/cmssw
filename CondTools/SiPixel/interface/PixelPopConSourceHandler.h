#ifndef PIXELPOPCONSOURCEHANDLER_H
#define PIXELPOPCONSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class: PixelPopConSourceHandler
/** \class PixelPopConSourceHandler PixelPopConSourceHandler.cc CondTools/SiPixel/src/PixelPopConSourceHandler.cc

 Description: PopCon source handler base class for all pixel popcon applications

 Implementation: 
   <Notes on implementation>
*/
//
// Original Author:  Michael Eads
//         Created:  19 Apr 2008
//
//

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <class T> 
class PixelPopConSourceHandler : public popcon::PopConSourceHandler<T> {

 public:
  //PixelPopConSourceHandler(edm::ParameterSet const &) {;}

  virtual void getNewObjects() {;}
  virtual std::string id() const {return std::string("PixelPopConSourceHandler");}

};








#endif
