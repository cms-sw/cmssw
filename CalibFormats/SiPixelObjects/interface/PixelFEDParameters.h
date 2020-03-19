#ifndef PixelFEDParameters_h
#define PixelFEDParameters_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelFEDParameters.h
*   \brief This class implements..
*
*   This class specifies which FED boards
*   are used and how they are addressed
*/

#include <iosfwd>

namespace pos {
  /*! \class PixelFEDParameters PixelFEDParameters.h "interface/PixelFEDParameters.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelFEDParameters;
  std::ostream& operator<<(std::ostream& s, const PixelFEDParameters& pFEDp);

  class PixelFEDParameters {
  public:
    PixelFEDParameters();
    ~PixelFEDParameters();

    unsigned int getFEDNumber() const;
    unsigned int getCrate() const;
    unsigned int getVMEBaseAddress() const;
    void setFEDParameters(unsigned int fednumber, unsigned int crate, unsigned int vmebaseaddress);
    void setFEDNumber(unsigned int fednumber);
    void setCrate(unsigned int crate);
    void setVMEBaseAddress(unsigned int vmebaseaddress);
    friend std::ostream& pos::operator<<(std::ostream& s, const PixelFEDParameters& pFEDp);

  private:
    unsigned int fednumber_;
    unsigned int crate_;
    unsigned int vmebaseaddress_;
  };
}  // namespace pos
#endif
