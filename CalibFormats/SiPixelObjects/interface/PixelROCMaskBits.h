#ifndef PixelROCMaskBits_h
#define PixelROCMaskBits_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/

#include <sstream>
#include <fstream>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

namespace pos{

  class PixelROCMaskBits;
  std::ostream& operator<<(std::ostream& s, const PixelROCMaskBits& maskbits);

/*! \class PixelROCMaskBits PixelROCMaskBits.h "interface/PixelROCMaskBits.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelROCMaskBits {

  public:

    PixelROCMaskBits();
    
    void setROCMaskBits(PixelROCName& rocid ,std::string bits);

    int read(const PixelROCName& rocid, std::string in);
    int read(const PixelROCName& rocid, std::ifstream& in);
    int read(const PixelROCName& rocid, std::istringstream& in);

    int readBinary(const PixelROCName& rocid, std::ifstream& in);

    unsigned int mask(unsigned int col, unsigned int row) const;

    void setMask(unsigned int col, unsigned int row, unsigned int mask);

    void writeBinary(std::ofstream& out) const;

    void writeASCII(std::ofstream&  out) const;
    void writeXML(  std::ofstream * out) const;

    PixelROCName name() const {return rocid_;}

    friend std::ostream& operator<<(std::ostream& s, const PixelROCMaskBits& maskbits);

  private:

    PixelROCName rocid_;
    unsigned char bits_[520];


  };
}

#endif
