#ifndef RecoLuminosity_LumiProducer_DIPLumiSummary_h
#define RecoLuminosity_LumiProducer_DIPLumiSummary_h
#include <iosfwd>
#include <string>
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummaryRcd.h"
#include "FWCore/Framework/interface/data_default_record_trait.h"
class DIPLumiSummary {
 public:
  /// default constructor
  DIPLumiSummary(){}
  
  /// set default constructor
  DIPLumiSummary(float instlumi,float dellumi,float reclumi,float deadfrac,unsigned short cmsalive):m_instlumi(instlumi),m_dellumi(dellumi),m_reclumi(reclumi),m_deadfrac(deadfrac),m_cmsalive(cmsalive){}
    
  /// destructor
  ~DIPLumiSummary(){}
  /** 
      average inst lumi,delivered, 
      unit Hz/ub, 
  **/
  float instDelLumi() const;
  /**
     delivered luminosity integrated over LS , 
     unit /ub,  
  **/ 
  float intgDelLumi()const;
  /**
     recorded luminosity integrated over LS , 
     unit /ub,  
  **/ 
  float intgRecLumi()const;
  /** 
      trigger Deadtime fraction
  **/
  float deadtimefraction() const;
  /**
     if cms central daq alive
   **/
  int cmsalive()const;
  //
  //setters
  //
  void setLumiData(float instlumi,float delivlumi,float reclumi);
  void setDeadFraction(float deadfrac);
  void setCMSAlive(unsigned short cmsalive);
 private :
  float m_instlumi;
  float m_dellumi;
  float m_reclumi;
  float m_deadfrac;
  unsigned short m_cmsalive;  
}; 

std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary);

EVENTSETUP_DATA_DEFAULT_RECORD(DIPLumiSummary,DIPLumiSummaryRcd)

#endif // RecoLuminosity_LuminosityProducer_DIPLumiSummary_h
