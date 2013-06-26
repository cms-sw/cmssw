#ifndef DQM_L1TMONITOR_L1TOMDSHELPER_H
#define DQM_L1TMONITOR_L1TOMDSHELPER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

// ROOT includes
#include "TString.h"

// System includes
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>

// Simplified structure for single object conditions information
class BeamConfiguration{

  public:

    BeamConfiguration(){
      m_valid           = false;
      nCollidingBunches = 0;
    }

    bool bxConfig(int iBx){
      if(beam1[iBx] && beam2[iBx]){return true;}
      else                        {return false;}
    }
    
    bool isValid()          {return m_valid;}
    
    bool m_valid;
    int  nCollidingBunches;
    std::vector<bool> beam1;
    std::vector<bool> beam2;

};

// Simplified structure for single object conditions information
struct WbMTriggerXSecFit{

  TString bitName;         // Bit Name for which the fit refers to
  TString fitFunction;     // Fitting function (hard coded for now...)
  float   bitNumber;       // Bit Number for which the fit refers to
  float   pm1, p0, p1, p2; // Fit parameters f(x)=pm1*x^(-1)+p0+p1*x+p2*x^2
 
};

class L1TOMDSHelper {

  public:

    enum Error{
      NO_ERROR=0,
      WARNING_DB_CONN_FAILED,
      WARNING_DB_QUERY_FAILED,
      WARNING_DB_INCORRECT_NBUNCHES
    };

  public:

    L1TOMDSHelper(); 
    ~L1TOMDSHelper(); // Destructor

    bool                                    connect              (std::string iOracleDB,std::string iPathCondDB,int &error);
    std::map<std::string,WbMTriggerXSecFit> getWbMTriggerXsecFits(std::string iTable,int &error);
    std::map<std::string,WbMTriggerXSecFit> getWbMAlgoXsecFits       (int &error);
    std::map<std::string,WbMTriggerXSecFit> getWbMTechXsecFits       (int &error);
    int                                     getNumberCollidingBunches(int lhcFillNumber,int &error);
    BeamConfiguration                       getBeamConfiguration     (int lhcFillNumber,int &error);
    std::vector<bool>                       getBunchStructure        (int lhcFillNumber,int &error);
    std::vector<float>                      getInitBunchLumi         (int lhcFillNumber,int &error);
    std::vector<double>                     getRelativeBunchLumi     (int lhcFillNumber,int &error);

    std::string                             enumToStringError(int);

  private:


    std::string m_oracleDB;
    std::string m_pathCondDB;

    l1t::OMDSReader *m_omdsReader;

};

#endif
