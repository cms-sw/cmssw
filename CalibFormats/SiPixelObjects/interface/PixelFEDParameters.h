#ifndef PixelFEDParameters_h
#define PixelFEDParameters_h

#include <iostream>
#include <vector>
#include <string>

namespace pos{
  class PixelFEDParameters {



  public:

    PixelFEDParameters();
    ~PixelFEDParameters();

    unsigned int getFEDNumber() const;
    unsigned int getCrate() const;
    unsigned int getVMEBaseAddress() const;
    void setFEDParameters( unsigned int fednumber , unsigned int crate , unsigned int vmebaseaddress);
    void setFEDNumber(unsigned int fednumber);
    void setCrate(unsigned int crate);
    void setVMEBaseAddress(unsigned int vmebaseaddress) ;
    friend std::ostream& pos::operator <<(std::ostream& s,const PixelFEDParameters &pFEDp);
  private :

    unsigned int fednumber_;   
    unsigned int crate_;   
    unsigned int vmebaseaddress_;   

  };
}
#endif
