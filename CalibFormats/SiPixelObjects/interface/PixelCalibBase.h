#ifndef PixelCalibBase_h
#define PixelCalibBase_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelCalibBase.h
*   \brief Base class for pixel calibration procedures
*
*   A longer explanation will be placed here later
*/

#include <string>


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
    virtual std::string mode() = 0;

  protected:

    std::string mode_;

  };
}
/* @} */

#endif
