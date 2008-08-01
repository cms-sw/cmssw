#ifndef PixelTKFECParameters_h
#define PixelTKFECParameters_h

#include <iostream>
#include <vector>
#include <string>

namespace pos{
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
}
#endif
