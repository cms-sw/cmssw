#ifndef HcalFrontEndId_h
#define HcalFrontEndId_h

#include <string>
#include <cstdint>
#include <iosfwd>

class HcalFrontEndId {
 public:
  HcalFrontEndId() : hcalFrontEndId_(0) {}
  HcalFrontEndId(uint32_t id) {hcalFrontEndId_=id;};
  HcalFrontEndId(const std::string& rbx,int rm,int pixel,int rmfiber,int fiberchannel,int qiecard,int adc);
  ~HcalFrontEndId();
  uint32_t rawId() const {return hcalFrontEndId_;}

  // index which uniquely identifies an RBX within HCAL
  int rbxIndex() const {return (hcalFrontEndId_>>18);}
  static const int maxRbxIndex=0xFF;
  // index which uniquely identifies an RM (e.g. HPD) within HCAL
  int rmIndex() const {return ((rm()-1)&0x3)+(rbxIndex()<<2);}
  static const int maxRmIndex=0x3FF;

  bool null() const { return hcalFrontEndId_==0; }

  std::string rbx() const;
  int rm() const {return ((hcalFrontEndId_>>15)&0x7)+1;}
  int pixel() const {return (hcalFrontEndId_>>10)&0x1F;}
  int rmFiber() const {return ((hcalFrontEndId_>>7)&0x7)+1;}
  int fiberChannel() const {return (hcalFrontEndId_>>5)&0x3;}
  int qieCard() const {return ((hcalFrontEndId_>>3)&0x3)+1;}
  int adc() const {return (hcalFrontEndId_&0x7)-1;}

  int operator==(const HcalFrontEndId& id) const { return id.hcalFrontEndId_==hcalFrontEndId_; }
  int operator!=(const HcalFrontEndId& id) const { return id.hcalFrontEndId_!=hcalFrontEndId_; }
  int operator<(const HcalFrontEndId& id) const { return hcalFrontEndId_<id.hcalFrontEndId_; }

 private:
  uint32_t hcalFrontEndId_;
};

std::ostream& operator<<(std::ostream&,const HcalFrontEndId& id);

#endif

