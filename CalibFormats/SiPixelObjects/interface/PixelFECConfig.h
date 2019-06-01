#ifndef PixelFECConfig_h
#define PixelFECConfig_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelFECConfig.h
*   \brief This class specifies which FEC boards are used and how they are addressed
*
*   A longer explanation will be placed here later
*/
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECParameters.h"

namespace pos {
  /*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelFECConfig PixelFECConfig.h
*  \brief This class specifies which FEC boards are used and how they are addressed
*/
  class PixelFECConfig : public PixelConfigBase {
  public:
    PixelFECConfig(
        std::string
            filename);  //  <---- Modified for the conversion from parallel vectors to object that contain the configuration

    PixelFECConfig(std::vector<std::vector<std::string> > &tableMat);

    unsigned int getNFECBoards() const;

    unsigned int getFECNumber(unsigned int i) const;
    unsigned int getCrate(unsigned int i) const;
    unsigned int getVMEBaseAddress(unsigned int i) const;
    unsigned int crateFromFECNumber(unsigned int fecnumber) const;
    unsigned int VMEBaseAddressFromFECNumber(unsigned int fecnumber) const;
    unsigned int getFECSlot(unsigned int i) { return FECSlotFromVMEBaseAddress(getVMEBaseAddress(i)); }
    unsigned int FECSlotFromFECNumber(unsigned int fecnumber) {
      return FECSlotFromVMEBaseAddress(VMEBaseAddressFromFECNumber(fecnumber));
    }

    void writeASCII(std::string dir = "") const override;
    void writeXML(pos::PixelConfigKey key, int version, std::string path) const override { ; }
    void writeXMLHeader(pos::PixelConfigKey key,
                        int version,
                        std::string path,
                        std::ofstream *out,
                        std::ofstream *out1 = nullptr,
                        std::ofstream *out2 = nullptr) const override;
    void writeXML(std::ofstream *out, std::ofstream *out1 = nullptr, std::ofstream *out2 = nullptr) const override;
    void writeXMLTrailer(std::ofstream *out,
                         std::ofstream *out1 = nullptr,
                         std::ofstream *out2 = nullptr) const override;

    //friend std::ostream& operator<<(std::ostream& s, const PixelDetectorconfig& config);

  private:
    // VMEBaseAddress = (FEC slot)x(0x8000000)
    unsigned int FECSlotFromVMEBaseAddress(unsigned int VMEBaseAddress) {
      assert(VMEBaseAddress % 0x8000000 == 0);
      return VMEBaseAddress / 0x8000000;
    }

    //Already fixed from parallel vectors to vector of objects .... the object that contains the FEC config is PixelFECParameters

    //    std::vector<unsigned int> fecnumber_;
    //    std::vector<unsigned int> crate_;
    //    std::vector<unsigned int> vmebaseaddress_;

    std::vector<PixelFECParameters> fecconfig_;
  };
}  // namespace pos
/* @} */
#endif
