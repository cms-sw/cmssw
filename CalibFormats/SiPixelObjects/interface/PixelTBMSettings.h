#ifndef PixelTBMSettings_h
#define PixelTBMSettings_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTBMSettings.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelTBMSettings PixelTBMSettings.h
*  \brief This is the documentation about PixelTBMSettings...
*
*   A longer explanation will be placed here later
*
*/
  class PixelTBMSettings: public PixelConfigBase {

  public:

    PixelTBMSettings(std::vector < std::vector< std::string> > &tableMat);
    PixelTBMSettings(std::string filename);
    // modified by MR on 29-04-2008 16:43:30
  PixelTBMSettings():PixelConfigBase("", "", "") {;}

    virtual ~PixelTBMSettings(){}

    //Generate the DAC settings
    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
	                       PixelNameTranslation* trans,
			       bool physics=false, bool doResets=true) const; 

    void writeBinary(std::string filename) const;

    void 	 writeASCII(std::string dir) const;
    void 	 writeXML(         pos::PixelConfigKey key, int version, std::string path) const {;}
    virtual void writeXMLHeader(   pos::PixelConfigKey key, 
				   int version, 
				   std::string path, 
				   std::ofstream *out,
				   std::ofstream *out1 = NULL,
				   std::ofstream *out2 = NULL
				   ) const ;
    virtual void writeXML(        std::ofstream *out,			                                    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;

    friend std::ostream& operator<<(std::ostream& s, const PixelTBMSettings& mask);

    unsigned char getAnalogInputBias() {return analogInputBias_;}
    void setAnalogInputBias(unsigned char analogInputBias) {analogInputBias_=analogInputBias;}
    
    unsigned char getAnalogOutputBias() {return analogOutputBias_;}
    void setAnalogOutputBias(unsigned char analogOutputBias) {analogOutputBias_=analogOutputBias;}
    
    unsigned char getAnalogOutputGain() {return analogOutputGain_;}
    void setAnalogOutputGain(unsigned char analogOutputGain) {analogOutputGain_=analogOutputGain;}
    
    // Added by Dario (Apr 2008)
    bool getMode(void)      {return singlemode_;}
    void setMode(bool mode) {singlemode_ = mode;}
    void setROCName(std::string rocname){
      	PixelROCName tmp(rocname);
	rocid_=tmp;
    }
    void setTBMGenericValue(std::string, int) ;
    
  private:

    PixelROCName rocid_;
    PixelModuleName moduleId_ ;

    unsigned char analogInputBias_;
    unsigned char analogOutputBias_;
    unsigned char analogOutputGain_;
    bool singlemode_;

  };
}
/* @} */
#endif
