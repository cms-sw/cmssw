#ifndef PixelFECConfig_h
#define PixelFECConfig_h
//
// This class specifies which FEC boards
// are used and how they are addressed
// 
// 
// 
//
//
//
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECParameters.h"

namespace pos{
  class PixelFECConfig: public PixelConfigBase {

  public:

    PixelFECConfig(std::string filename);  //  <---- Modified for the conversion from parallel vectors to object that contain the configuration
   
    PixelFECConfig(std::vector<std::vector<std::string> >& tableMat ); 

    unsigned int getNFECBoards() const;

    unsigned int getFECNumber(unsigned int i) const;
    unsigned int getCrate(unsigned int i) const;
    unsigned int getVMEBaseAddress(unsigned int i) const;
    unsigned int crateFromFECNumber(unsigned int fecnumber) const;
    unsigned int VMEBaseAddressFromFECNumber(unsigned int fecnumber) const;
    unsigned int getFECSlot(unsigned int i) {return FECSlotFromVMEBaseAddress(getVMEBaseAddress(i));}
    unsigned int FECSlotFromFECNumber(unsigned int fecnumber) {return FECSlotFromVMEBaseAddress(VMEBaseAddressFromFECNumber(fecnumber));}

    void writeASCII(std::string dir="") const;

    //friend std::ostream& operator<<(std::ostream& s, const PixelDetectorconfig& config);

  private:

    // VMEBaseAddress = (FEC slot)x(0x8000000)
    unsigned int FECSlotFromVMEBaseAddress(unsigned int VMEBaseAddress) {assert(VMEBaseAddress%0x8000000 == 0); return VMEBaseAddress/0x8000000;}

    //Already fixed from parallel vectors to vector of objects .... the object that contains the FEC config is PixelFECParameters 	
   
    //    std::vector<unsigned int> fecnumber_;   
    //    std::vector<unsigned int> crate_;   
    //    std::vector<unsigned int> vmebaseaddress_;
    
    std::vector< PixelFECParameters > fecconfig_;
    
    
 
  };
}
#endif
