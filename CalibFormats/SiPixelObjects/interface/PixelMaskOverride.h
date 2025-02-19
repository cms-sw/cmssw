#ifndef PixelMaskOverride_h
#define PixelMaskOverride_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelMaskOverride.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include <string>
#include "PixelROCMaskBits.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaskOverride PixelMaskOverride.h
*  \brief This is the documentation about PixelMaskOverride...
*
*/
  class PixelMaskOverride: public PixelMaskOverrideBase {

  public:

    PixelMaskOverride(std::string filename);

    PixelROCMaskBits getMaskBits(int ROCId );

  private:

    //need to store the input here....

  };
}
/* @} */
#endif
