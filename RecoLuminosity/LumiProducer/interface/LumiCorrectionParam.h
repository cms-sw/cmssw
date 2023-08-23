#ifndef RecoLuminosity_LumiProducer_LumiCorrectionParam_h
#define RecoLuminosity_LumiProducer_LumiCorrectionParam_h
#include <iosfwd>
#include <map>
#include <string>
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParamRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
class LumiCorrectionParam {
public:
  enum LumiType { HF, PIXEL };
  /// default constructor
  LumiCorrectionParam();
  explicit LumiCorrectionParam(LumiType lumitype);
  /// destructor
  ~LumiCorrectionParam() {}

  /*
    getters are used by users
    standard user just needs to apply the final correction
    other methods are for inspection or advanced correction customisation
  */
  ///get the final correction factor
  float getCorrection(float luminonorm) const;
  ///get ncollidingbunches
  unsigned int ncollidingbunches() const;
  ///get current normtag
  std::string normtag() const;
  ///get correction function name
  std::string corrFunc() const;
  ///get correction coefficients
  const std::map<std::string, float>& nonlinearCoeff() const;
  ///get afterglow threshold/value
  const std::map<unsigned int, float>& afterglows() const;
  ///on which amodetag this correction definition should be applied for
  ///information only
  std::string amodetag() const;
  ///on which single beam egev this correction definition should be applied for
  ///information only
  unsigned int beamegev() const;
  ///lumi integrated up to that run. Needed for 2011B correction, otherwise 0.
  float intglumi() const;
  /*
    setters 
  */
  ///set ncollidingbunches
  void setNBX(unsigned int nbx);
  ///set current normtag
  void setNormtag(const std::string& normtag);
  ///set correction function
  void setcorrFunc(const std::string& corrfunc);
  ///set nonlinear constants
  void setnonlinearCoeff(std::map<std::string, float>& coeffmap);
  ///set afterglow thresholds
  void setafterglows(std::map<unsigned int, float>& afterglows);
  ///set description
  void setdescription(const std::string& amodetag, unsigned int beamegev);
  ///set intglumi
  void setintglumi(float intglumi);

private:
  LumiType m_lumitype;
  unsigned int m_ncollidingbx;
  std::string m_normtag;
  std::string m_corrfunc;
  std::map<std::string, float> m_coeffmap;
  std::map<unsigned int, float> m_afterglows;
  std::string m_amodetag;
  float m_beamegev;
  float m_intglumi;
};

std::ostream& operator<<(std::ostream& s, const LumiCorrectionParam& lumicorr);

EVENTSETUP_DATA_DEFAULT_RECORD(LumiCorrectionParam, LumiCorrectionParamRcd)

#endif  // RecoLuminosity_LuminosityProducer_LumiCorrectionParam_h
