#ifndef RecoLuminosity_LumiProducer_LumiCorrectionParam_h
#define RecoLuminosity_LumiProducer_LumiCorrectionParam_h
#include <iosfwd>
#include <string>
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParamRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
class LumiCorrectionParam {
 public:
  enum LumiType{HF,PIXEL};
  
  /// default constructor
  LumiCorrectionParam();
  explicit LumiCorrectionParam(LumiType lumitype);
  /// destructor
  ~LumiCorrectionParam(){}
  
  ///set ncollidingbunches
  void setNBX(unsigned int nbx);
  ///get ncollidingbunches
  unsigned int ncollidingbunches()const;
  ///
 private :
  LumiType m_lumitype;
  unsigned int m_ncollidingbx;
}; 

std::ostream& operator<<(std::ostream& s, const LumiCorrectionParam& lumicorr);

EVENTSETUP_DATA_DEFAULT_RECORD(LumiCorrectionParam,LumiCorrectionParamRcd)

#endif // RecoLuminosity_LuminosityProducer_LumiCorrectionParam_h
