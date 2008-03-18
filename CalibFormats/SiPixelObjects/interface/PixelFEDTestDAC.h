#ifndef _PixelFEDTestDAC_h_
#define _PixelFEDTestDAC_h_
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelFEDTestDAC.h
*   \brief This class implements..
*
*   This class specifies which FED boards
*   are used and how they are addressed
*/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include "CalibFormats/SiPixelObjects/interface/PixelCalibBase.h"

namespace pos{

/*! \ingroup CalibrationObjects
*
*   @{
*
*   \class PixelFEDTestDAC PixelFEDTestDAC.h "interface/PixelFEDTestDAC.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelFEDTestDAC : public PixelCalibBase{

  public:
    PixelFEDTestDAC(std::string filename);
    std::string mode() {return mode_;}
    std::vector<unsigned int> dacs() {return dacs_;}

  private:

    unsigned int levelEncoder(int level);
    std::vector<unsigned int> decimalToBaseX(unsigned int a, unsigned int x, unsigned int length);
    std::vector<unsigned int> dacs_;

  };
}
/* @} */
#endif




	
	
