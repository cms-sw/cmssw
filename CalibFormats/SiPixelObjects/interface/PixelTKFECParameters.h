#ifndef PixelTKFECParameters_h
#define PixelTKFECParameters_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include <iosfwd>
#include <string>

namespace pos{
/*! \class PixelTKFECParameters PixelTKFECParameters.h "interface/PixelTKFECParameters.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelTKFECParameters {

  public:

    PixelTKFECParameters();
    ~PixelTKFECParameters();

    std::string  getTKFECID() const;
    unsigned int getCrate() const;
    std::string  getType() const;
    unsigned int getAddress() const;
    void setTKFECParameters( std::string TKFECID , unsigned int crate , std::string type, unsigned int address);
    void setTKFECID(std::string TKFECID);
    void setCrate(unsigned int crate);
    void setType(std::string type);
    void setAddress(unsigned int address) ;
    friend std::ostream& operator <<(std::ostream& s,const PixelTKFECParameters &pTKFECp);
  private :

    std::string  TKFECID_;
    unsigned int crate_;
    std::string  type_;
    unsigned int address_;

  };
  std::ostream& operator <<(std::ostream& s ,const PixelTKFECParameters &pTKFECp);

}
/* @} */
#endif
