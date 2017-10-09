#ifndef PixelMaskBase_h
#define PixelMaskBase_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelMaskBase.h
*
*  This class provide a base class for the
*  pixel mask data for the pixel FEC configuration
*  This is a pure interface (abstract class) that
*  needs to have an implementation.
*/

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelMaskOverrideBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"
#include <string>
#include <iostream>

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaskBase PixelMaskBase.h
*  \brief This is the documentation about PixelMaskBase...
*
*  This class provide a base class for the
*  pixel mask data for the pixel FEC configuration
*  This is a pure interface (abstract class) that
*  needs to have an implementation.
* 
*  All applications should just use this 
*  interface and not care about the specific
*  implementation
*/
  class PixelMaskBase: public PixelConfigBase {

  public:

    PixelMaskBase(std::string description, 
		  std::string creator,
		  std::string date);

    virtual ~PixelMaskBase();

    void setOverride(PixelMaskOverrideBase*);

    virtual const PixelROCMaskBits& getMaskBits(int ROCId) const =0;

    virtual PixelROCMaskBits* getMaskBits(PixelROCName name) =0;


    virtual void writeBinary(std::string filename) const =0;

    virtual void writeASCII(std::string filename) const =0;
    virtual void writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, 
				int version, 
				std::string path, 
				std::ofstream *out,
				std::ofstream *out1 = NULL,
				std::ofstream *out2 = NULL
				) const {;}
    virtual void writeXML( std::ofstream *out,                                                              
			   std::ofstream *out1 = NULL ,
			   std::ofstream *out2 = NULL ) const {;}
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 =NULL
				  ) const {;}

    friend std::ostream& operator<<(std::ostream& s, const PixelMaskBase& mask);
    
  private:

    //Hold pointer to the mask override information.
    PixelMaskOverrideBase* maskOverride_;


  };
}
/* @} */
#endif
