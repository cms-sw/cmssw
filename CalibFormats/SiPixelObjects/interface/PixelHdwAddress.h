#ifndef PixelHdwAddress_h
#define PixelHdwAddress_h
//
// Store mfec, mfecchannel etc.
//

#include <iostream>
#include <fstream>
#include <string>

namespace pos{
  class PixelHdwAddress {

  public:

    PixelHdwAddress();

    PixelHdwAddress(int fecnumber, int mfec, int mfecchannel,
		    int hubaddress, int portaddress, int rocid,
		    int fednumber, int fedchannel, int fedrocnumber);
    

    unsigned int fecnumber() const { return fecnumber_; }
    unsigned int mfec() const { return mfec_; }
    unsigned int mfecchannel() const { return mfecchannel_;}
    unsigned int hubaddress() const { return hubaddress_; }
    unsigned int portaddress() const { return portaddress_; }
    unsigned int rocid() const { return rocid_; }
    unsigned int fednumber() const { return fednumber_; }
    unsigned int fedchannel() const { return fedchannel_; }
    unsigned int fedrocnumber() const { return fedrocnumber_; }


    friend std::ostream& pos::operator<<(std::ostream& s, const PixelHdwAddress& pixelroc);


    const PixelHdwAddress& operator=(const PixelHdwAddress& aROC);
    
    // Checks for equality of all parts except the ROC numbers and portaddress.
    const bool operator|=(const PixelHdwAddress& aHdwAddress) const{
      return ( fecnumber_    == aHdwAddress.fecnumber_ &&
               mfec_         == aHdwAddress.mfec_ &&
               mfecchannel_  == aHdwAddress.mfecchannel_ &&
               hubaddress_   == aHdwAddress.hubaddress_ &&
               fednumber_    == aHdwAddress.fednumber_ &&
               fedchannel_   == aHdwAddress.fedchannel_ );
    }

  private:


    unsigned int fecnumber_;  
    unsigned int mfec_;  
    unsigned int mfecchannel_;  
    unsigned int portaddress_;  
    unsigned int hubaddress_;  
    unsigned int rocid_;  
    unsigned int fednumber_;  
    unsigned int fedchannel_;  
    unsigned int fedrocnumber_;  

    
  };
}
#endif
