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
// $Id$
//
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"

#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"

// class definition
class PixelPopConCalibSourceHandler : public popcon::PopConSourceHandler<SiPixelCalibConfiguration> {
  
 public:
  void getNewObjects();
  // specific implementations of getNewObjects
  void getNewObjects_coral();
  void getNewObjects_file();
  ~PixelPopConCalibSourceHandler();
  PixelPopConCalibSourceHandler(edm::ParameterSet const &);
  virtual std::string id() const;

 private:
  std::string _connectString;
  std::string _schemaName;
  std::string _viewName;
  std::string _configKeyName;
  int _runNumber;
  unsigned int _sinceIOV;

};


#endif
