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
  DIPLumiSummary(float instlumi,float dellumi,float reclumi,unsigned short cmsalive):m_instlumi(instlumi),m_totdellumi(dellumi),m_totreclumi(reclumi),m_deadfrac(1.0),m_cmsalive(cmsalive){}
    
  /// destructor
  ~DIPLumiSummary(){}
  /** 
      average inst lumi,delivered HF, 
      unit Hz/ub, 
  **/
  float instDelLumi() const;
  /**
     delivered luminosity integrated over this LS , 
     instDelLumi*23.31
     unit /ub,  
  **/ 
  float intgDelLumiByLS()const;
  /**
     recorded luminosity integrated over this LS,this is deduced
     intgRecLumi=intgDelLumi*(m_totreclumi/m_totdellumi)
     unit /ub,  
  **/ 
  float intgRecLumiByLS()const;
  /** 
      trigger Deadtime fraction, this is deduced 
      1.0-m_totreclumi/m_totdellumi
  **/
  float deadtimefraction() const;
  /**
     delivered luminosity integrated since the beginning of run
     unit /ub  
   **/
  float intgDelLumiSinceRun()const;
  /**
     recorded luminosity integrated since the beginning of run
     unit /ub
   **/
  float intgRecLumiSinceRun()const;
  /**
     if cms central daq alive
   **/
  int cmsalive()const;
  //
  //setters
  //
 private :
  float m_instlumi;//avg inst lumi in LS
  float m_totdellumi;//total integrated luminosity counting from the beg of run
  float m_totreclumi;
  mutable float m_deadfrac;
  unsigned short m_cmsalive;  
}; 

std::ostream& operator<<(std::ostream& s, const DIPLumiSummary& diplumiSummary);

EVENTSETUP_DATA_DEFAULT_RECORD(DIPLumiSummary,DIPLumiSummaryRcd)

#endif // RecoLuminosity_LuminosityProducer_DIPLumiSummary_h
