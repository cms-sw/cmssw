#ifndef YELLOWPARAMS_H
#define YELLOWPARAMS_H

namespace l1t {

  class YellowParams {

  public:
    // constructor
    YellowParams() {}
    
    unsigned firmwareVersion() const {return m_fw_version;}
    unsigned paramA() const {return m_a;}
    unsigned paramB() const {return m_b;}
    unsigned paramC() const {return m_c;}
    
    void setFirmwareVersion(unsigned v) {m_fw_version=v;};
    void setParamA(unsigned v) {m_a = v;}
    void setParamB(unsigned v) {m_b = v;}
    void setParamC(unsigned v) {m_c = v;}
    
  private:
    unsigned m_fw_version, m_a, m_b, m_c;
  };

}// namespace
#endif
