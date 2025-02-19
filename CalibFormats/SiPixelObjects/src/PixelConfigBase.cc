//
// Base class for pixel configuration data
// provide a place to implement common interfaces
// for these objects. Any configuration data
// object that is to be accessed from the database
// should derive from this class
//

#include <iostream>
#include <fstream>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"


using namespace pos;

PixelConfigBase::PixelConfigBase(std::string description,
				 std::string creator,
				 std::string date):
  description_(description),
  creator_(creator),
  date_(date){
}
