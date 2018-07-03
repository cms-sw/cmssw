#ifndef PixelMaskAllPixels_h
#define PixelMaskAllPixels_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels..h
*   \brief This clss implements..
*
*   A longer explanation will be placed here later
*/
#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaskAllPixels PixelMaskAllPixels.h
*  \brief This is the documentation about PixelMaskAllPixels...
*
*/
  class PixelMaskAllPixels: public PixelMaskBase {

  public:

    PixelMaskAllPixels(std::string filename);
    PixelMaskAllPixels(std::vector< std::vector<std::string> >& tableMat);
// modified by MR on 18-04-2008 10:05:04
    PixelMaskAllPixels() ;
    void addROCMaskBits(const PixelROCMaskBits&);
    

    void writeBinary(std::string filename) const override;

    void 	 writeASCII(std::string dir) const override;
    void 	 writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const override {;}
    void writeXMLHeader(pos::PixelConfigKey key, 
				int version, 
				std::string path, 
				std::ofstream *out,
				std::ofstream *out1 = nullptr,
				std::ofstream *out2 = nullptr
				) const override ;
    void writeXML( std::ofstream *out,                                                              
			   std::ofstream *out1 = nullptr ,
			   std::ofstream *out2 = nullptr ) const override ;
    void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = nullptr,
				  std::ofstream *out2 =nullptr
				  ) const override ;

    const PixelROCMaskBits& getMaskBits(int ROCId) const override;

    PixelROCMaskBits* getMaskBits(PixelROCName name) override;

  private:

    std::vector<PixelROCMaskBits> maskbits_;  
 
  };
}
/* @} */
#endif
