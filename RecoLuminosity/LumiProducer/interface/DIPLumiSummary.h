#ifndef RecoLuminosity_LumiProducer_DIPLumiSummary_h
#define RecoLuminosity_LumiProducer_DIPLumiSummary_h
#include <iosfwd>
#include <string>
#include "RecoLuminosity/LumiProducer/interface/DIPLuminosityRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
class DIPLumiSummary {
public:
  /// default constructor
  DIPLumiSummary()
      : m_runnum(0), m_ls(0), m_instlumi(0.0), m_dellumi(0.0), m_reclumi(0.0), m_deadfrac(1.0), m_cmsalive(false) {}

  /// set default constructor
  DIPLumiSummary(float instlumi, float dellumi, float reclumi, unsigned short cmsalive)
      : m_instlumi(instlumi), m_dellumi(dellumi), m_reclumi(reclumi), m_deadfrac(1.0), m_cmsalive(cmsalive) {
    if (m_reclumi > 0.0) {
      m_deadfrac = 1.0 - (m_reclumi / m_dellumi);
    }
  }

  /// destructor
  ~DIPLumiSummary() {}
  bool isNull() const;
  /** 
      average inst lumi,delivered HF, 
      unit Hz/ub, 
  **/
  float instDelLumi() const;
  /**
     delivered luminosity integrated over this LS , 
     unit /ub,  
  **/
  float intgDelLumiByLS() const;
  /**
     recorded luminosity integrated over this LS,this is deduced
     unit /ub,  
  **/
  float intgRecLumiByLS() const;
  /** 
      trigger Deadtime fraction, this is deduced 
      1.0-m_reclumi/m_dellumi
  **/
  float deadtimefraction() const;
  /**
     if cms central daq alive
   **/
  int cmsalive() const;
  //
  //setters
  //
  /**
     from which run data come from
  **/
  unsigned int fromRun() const;
  /**
     from which ls data come from
  **/
  unsigned int fromLS() const;
  void setOrigin(unsigned int runnumber, unsigned int ls);

private:
  unsigned int m_runnum;
  unsigned int m_ls;
  const float m_instlumi;  //avg inst lumi in LS
  const float m_dellumi;   //integrated luminosity of this ls
  const float m_reclumi;
  float m_deadfrac;
  const unsigned short m_cmsalive;
};

std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary);

EVENTSETUP_DATA_DEFAULT_RECORD(DIPLumiSummary, DIPLuminosityRcd)

#endif  // RecoLuminosity_LuminosityProducer_DIPLumiSummary_h
