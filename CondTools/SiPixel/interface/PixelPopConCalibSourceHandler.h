#ifndef PIXELPOPCONCALIBSOURCEHANDLER_H
#define PIXELPOPCONCALIBSOURCEHANDLER_H

// Package: CondTools/SiPixel
// Class: PixelPopConSourceHandler
/** \class PixelPopConSourceHandler PixelPopConSourceHandler.cc CondTools/SiPixel/src/PixelPopConSourceHandler.cc

 Description: The PopCon source handler class to transfer pixel calibration 
objects from OMDS to ORCON.

 Implementation: 
   <Notes on implementation>
*/
//
// Original Author:  Michael Eads
//         Created:  8 Feb 2008
// $Id: PixelPopConCalibSourceHandler.h,v 1.5 2010/06/04 09:18:01 innocent Exp $
//
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondTools/SiPixel/interface/PixelPopConSourceHandler.h"


// class definition
class PixelPopConCalibSourceHandler : public PixelPopConSourceHandler<SiPixelCalibConfiguration> {
  
 public:
  // specific implementations of getNewObjects
  void getNewObjects_coral();
  void getNewObjects_file();
  ~PixelPopConCalibSourceHandler();
  PixelPopConCalibSourceHandler(edm::ParameterSet const &);
  virtual std::string id() const;

 private:

};


#endif
