#ifndef PixelTBMSettings_h
#define PixelTBMSettings_h
//
// This class provide a base class for the
// pixel ROC dac data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
//
//

#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
  class PixelTBMSettings: public PixelConfigBase {

  public:

    PixelTBMSettings(std::vector < std::vector< std::string> > &tableMat);
    PixelTBMSettings(std::string filename);

    virtual ~PixelTBMSettings(){}

    //Generate the DAC settings
    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
	                       PixelNameTranslation* trans) const; 

    void writeBinary(std::string filename) const;

    void writeASCII(std::string dir) const;

    friend std::ostream& operator<<(std::ostream& s, const PixelTBMSettings& mask);

    unsigned char getAnalogInputBias() {return analogInputBias_;}
    void setAnalogInputBias(unsigned char analogInputBias) {analogInputBias_=analogInputBias;}
    
    unsigned char getAnalogOutputBias() {return analogOutputBias_;}
    void setAnalogOutputBias(unsigned char analogOutputBias) {analogOutputBias_=analogOutputBias;}
    
    unsigned char getAnalogOutputGain() {return analogOutputGain_;}
    void setAnalogOutputGain(unsigned char analogOutputGain) {analogOutputGain_=analogOutputGain;}

  private:

    PixelROCName rocid_;

    unsigned char analogInputBias_;
    unsigned char analogOutputBias_;
    unsigned char analogOutputGain_;
    bool singlemode_;

  };
}
#endif
