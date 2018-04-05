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
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template <class T> 
class PixelPopConSourceHandler : public popcon::PopConSourceHandler<T> {

 public:
  //PixelPopConSourceHandler(edm::ParameterSet const &) {;}

  void getNewObjects() override {
    // look at _connectString to see which method to call
    if (_connectString.find("oracle") == 0)
      getNewObjects_coral();
    else if (_connectString.find("file") == 0)
      getNewObjects_file();
    else {
      std::cout << "  PixelPopConSourceHandler::getNewObjects() - unknown connect string:" << _connectString << std::endl;
      std::cout << "    connect string must begin with \"oracle\" or \"file\"" << std::endl;
    }
    
  } // virtual void getNewObjects()

  virtual void getNewObjects_coral() {;}
  virtual void getNewObjects_file() {;}
  std::string id() const override {return std::string("PixelPopConSourceHandler");}

 protected:
  std::string _connectString;
  std::string _schemaName;
  std::string _viewName;
  std::string _configKeyName;
  int _runNumber;
  unsigned int _sinceIOV;

};








#endif
