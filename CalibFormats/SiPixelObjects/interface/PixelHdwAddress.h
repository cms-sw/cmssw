#ifndef PixelHdwAddress_h
#define PixelHdwAddress_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h
*   \brief Store mfec, mfecchannel etc.
*
*   A longer explanation will be placed here later
*/

#include <iosfwd>
#include <string>

namespace pos {
  /*! \class PixelHdwAddress PixelHdwAddress.h "interface/PixelHdwAddress.h"
*   \brief Store mfec, mfecchannel etc.
*
*   A longer explanation will be placed here later
*/
  class PixelHdwAddress;

  std::ostream& operator<<(std::ostream& s, const PixelHdwAddress& pixelroc);

  class PixelHdwAddress {
  public:
    PixelHdwAddress();

    PixelHdwAddress(int fecnumber,
                    int mfec,
                    int mfecchannel,
                    int hubaddress,
                    int portaddress,
                    int rocid,
                    int fednumber,
                    int fedchannel,
                    int fedrocnumber);

    unsigned int fecnumber() const { return fecnumber_; }
    unsigned int mfec() const { return mfec_; }
    unsigned int mfecchannel() const { return mfecchannel_; }
    unsigned int hubaddress() const { return hubaddress_; }
    unsigned int portaddress() const { return portaddress_; }
    unsigned int rocid() const { return rocid_; }
    unsigned int fednumber() const { return fednumber_; }
    unsigned int fedchannel() const { return fedchannel_; }
    unsigned int fedrocnumber() const { return fedrocnumber_; }

    friend std::ostream& pos::operator<<(std::ostream& s, const PixelHdwAddress& pixelroc);

    const PixelHdwAddress& operator=(const PixelHdwAddress& aROC);

    bool operator()(const PixelHdwAddress& roc1, const PixelHdwAddress& roc2) const;

    // Checks for equality of all parts except the ROC numbers and portaddress.
    const bool operator|=(const PixelHdwAddress& aHdwAddress) const {
      return (fecnumber_ == aHdwAddress.fecnumber_ && mfec_ == aHdwAddress.mfec_ &&
              mfecchannel_ == aHdwAddress.mfecchannel_ && hubaddress_ == aHdwAddress.hubaddress_ &&
              fednumber_ == aHdwAddress.fednumber_ && fedchannel_ == aHdwAddress.fedchannel_);
    }

    const bool operator<(const PixelHdwAddress& aHdwAddress) const {
      if (fednumber_ < aHdwAddress.fednumber_)
        return true;
      if (fednumber_ > aHdwAddress.fednumber_)
        return false;
      if (fedchannel_ < aHdwAddress.fedchannel_)
        return true;
      if (fedchannel_ > aHdwAddress.fedchannel_)
        return false;
      return (fedrocnumber_ < aHdwAddress.fedrocnumber_);
    }

    void setAddress(std::string what, int value);                                                  // Added by Dario
    void compare(std::string what, bool& changed, unsigned int newValue, unsigned int& oldValue);  // Added by Dario

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

}  // namespace pos
#endif
