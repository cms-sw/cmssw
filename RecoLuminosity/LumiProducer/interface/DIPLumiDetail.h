#ifndef RecoLuminosity_LumiProducer_DIPLumiDetail_h
#define RecoLuminosity_LumiProducer_DIPLumiDetail_h
#include <iosfwd>
#include <string>
#include "RecoLuminosity/LumiProducer/interface/DIPLuminosityRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
class DIPLumiDetail {
public:
  /// default constructor
  DIPLumiDetail();
  typedef std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> ValueRange;
  /// set default constructor
  virtual ~DIPLumiDetail() {}
  bool isNull() const;
  float lumiValue(unsigned int bx) const;
  ValueRange lumiValues() const;
  void filldata(std::vector<float>& lumivalues);
  void fillbxdata(unsigned int bxidx, float bxlumi);
  unsigned int fromRun() const;
  unsigned int fromLS() const;
  void setOrigin(unsigned int runnumber, unsigned int ls);

private:
  unsigned int m_runnum;
  unsigned int m_ls;
  std::vector<float> m_lumiValues;
};

std::ostream& operator<<(std::ostream& s, const DIPLumiDetail&);

EVENTSETUP_DATA_DEFAULT_RECORD(DIPLumiDetail, DIPLuminosityRcd)

#endif  // RecoLuminosity_LuminosityProducer_DIPLumiDetail_h
