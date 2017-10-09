#ifndef PixelTrimAllPixels_h
#define PixelTrimAllPixels_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTrimAllPixels.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include <string>
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
/*!  \ingroup TrimObjects "Trim Objects"
*    \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelTrimBase PixelTrimBase.h
*  \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/
  class PixelTrimAllPixels: public PixelTrimBase {

  public:

    PixelTrimAllPixels(std::string filename);
    PixelTrimAllPixels(std::vector<std::vector<std::string> > &tableMat);  

    //Build the commands needed to configure ROCs
    //on control link

    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
			       PixelNameTranslation* trans,
			       const PixelMaskBase& pixelMask) const;

    void writeBinary(std::string filename) const;

    void 	 writeASCII(std::string filename) const;
    void 	 writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, 
				int version, 
				std::string path, 
				std::ofstream *out,
				std::ofstream *out1 = NULL,
				std::ofstream *out2 = NULL
				) const ;
    virtual void writeXML( std::ofstream *out,                                                              
			   std::ofstream *out1 = NULL ,
			   std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 =NULL
				  ) const ;

    PixelROCTrimBits getTrimBits(int ROCId) const;

    PixelROCTrimBits* getTrimBits(PixelROCName name);


  private:

    std::vector<std::string> rocname_;
    std::vector<PixelROCTrimBits> trimbits_;

  };
}
/* @} */
#endif
