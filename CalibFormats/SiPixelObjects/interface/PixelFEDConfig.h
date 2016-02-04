#ifndef PixelFEDConfig_h
#define PixelFEDConfig_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelFEDConfig.h
*   \brief This class implements..
*
*   This class specifies which FED boards
*   are used and how they are addressed
*/

#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFEDParameters.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelFEDConfig PixelFEDConfig.h
*  \brief This is the documentation about PixelFEDConfig...
*
*  This class specifies which FED boards
*  are used and how they are addressed
*/
  class PixelFEDConfig: public PixelConfigBase {

  public:

    PixelFEDConfig(std::string filename);  //  <---- Modified for the conversion from parallel vectors to object that contain the configuration
   
    PixelFEDConfig(std::vector<std::vector<std::string> >& tableMat ); 

    virtual ~PixelFEDConfig();

    unsigned int getNFEDBoards() const;

    unsigned int getFEDNumber(unsigned int i) const;
    unsigned int getCrate(unsigned int i) const;
    unsigned int getVMEBaseAddress(unsigned int i) const;
    unsigned int crateFromFEDNumber(unsigned int fednumber) const;
    unsigned int VMEBaseAddressFromFEDNumber(unsigned int fednumber) const;
    
    unsigned int FEDNumberFromCrateAndVMEBaseAddress(unsigned int crate, unsigned int vmebaseaddress) const;

    virtual void writeASCII(std::string dir) const;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path) const {;}
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
    virtual void writeXML( 	  std::ofstream *out,			   	 			    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;

    //friend std::ostream& operator<<(std::ostream& s, const PixelDetectorconfig& config);

  private:

    //Already fixed from parallel vectors to vector of objects .... the object that contains the FED config is PixelFEDParameters 
    
    //    std::vector<unsigned int> fednumber_; 
    //    std::vector<unsigned int> crate_;
    //    std::vector<unsigned int> vmebaseaddress_;
    
    std::vector< PixelFEDParameters > fedconfig_;
 
  };
}
/* @} */
#endif
