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
// $Id: PixelPopConCalibSourceHandler.h,v 1.4 2010/01/21 21:11:45 meads Exp $
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
  void getNewObjects_coral() override;
  void getNewObjects_file() override;
  ~PixelPopConCalibSourceHandler() override;
  PixelPopConCalibSourceHandler(edm::ParameterSet const &);
  std::string id() const override;

 private:

};


#endif
