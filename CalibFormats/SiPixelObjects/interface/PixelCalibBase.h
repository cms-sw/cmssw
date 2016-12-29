#ifndef PixelCalibBase_h
#define PixelCalibBase_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelCalibBase.h
*   \brief Base class for pixel calibration procedures
*
*   A longer explanation will be placed here later
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigKey.h"
#include <string>
#include <fstream>


namespace pos{
/*!  \defgroup CalibrationObjects "Calibration Objects"
*    \brief Base class for pixel calibration procedures
*
*  @{
*
*   \class PixelCalibBase PixelCalibBase.h "interface/PixelCalibBase.h"
*
*   A longer explanation will be placed here later
*/
  class PixelCalibBase {

  public:

    PixelCalibBase();
    virtual ~PixelCalibBase();
    virtual std::string mode() const {return mode_;}
    virtual void writeXMLHeader(  pos::PixelConfigKey &key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const {;}
    virtual void writeXML( 	  std::ofstream *out,			     	   			    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const {;}
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const {;}

  protected:

    std::string mode_;

  };
}
/* @} */

#endif
