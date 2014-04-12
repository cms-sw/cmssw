#ifndef PixelMaskOverrideBase_h
#define PixelMaskOverrideBase_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelMaskOverrideBase.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaskOverrideBase PixelMaskOverrideBase.h
*  \brief This is the documentation about PixelMaskOverrideBase...
*
*/
  class PixelMaskOverrideBase: public PixelConfigBase {

  public:

    PixelMaskOverrideBase(std::string description, 
			  std::string creator,
			  std::string date);

    virtual ~PixelMaskOverrideBase();

    virtual PixelROCMaskBits getMaskBits(int ROCId)=0;

  private:

  };
}
/* @} */
#endif
