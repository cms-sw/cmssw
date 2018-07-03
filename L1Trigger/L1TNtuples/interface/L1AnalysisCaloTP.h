#ifndef __L1Analysis_L1AnalysisCaloTP_H__
#define __L1Analysis_L1AnalysisCaloTP_H__

//-------------------------------------------------------------------------------
// Created 06/01/2010 - A.C. Le Bihan
// 
// 
// Original code : L1Trigger/L1TNtuples/L1NtupleProducer
//-------------------------------------------------------------------------------

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1AnalysisCaloTPDataFormat.h"

class L1CaloEcalScale;
class L1CaloHcalScale;

namespace L1Analysis
{
  class L1AnalysisCaloTP
  {
  public:
    L1AnalysisCaloTP();
    L1AnalysisCaloTP(bool verbose);
    ~L1AnalysisCaloTP();
    
    // setup scales
    void setEcalScale(const L1CaloEcalScale* ecalScale) {
      edm::LogInfo("L1NTUPLE") << "Setting ECAL TP scale " << ecalScale << std::endl;
      ecalScale_ = ecalScale;
    }
    
    void setHcalScale(const L1CaloHcalScale* hcalScale) {
      edm::LogInfo("L1NTUPLE") << "Setting HCAL TP scale " << hcalScale << std::endl;
      hcalScale_ = hcalScale;
    }


    void SetHCAL(const HcalTrigPrimDigiCollection& hcalTPs);

    void SetECAL(const EcalTrigPrimDigiCollection& ecalTPs);
                   
    void Reset() {tp_.Reset();}

    L1AnalysisCaloTPDataFormat * getData() {return &tp_;}
 
  private :
    bool verbose_;
    L1AnalysisCaloTPDataFormat tp_;

    const L1CaloEcalScale* ecalScale_;
    const L1CaloHcalScale* hcalScale_;

  }; 
} 
#endif
