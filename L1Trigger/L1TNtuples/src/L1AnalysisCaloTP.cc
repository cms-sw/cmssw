#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisCaloTP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"


L1Analysis::L1AnalysisCaloTP::L1AnalysisCaloTP():verbose_(false)
{
}

L1Analysis::L1AnalysisCaloTP::L1AnalysisCaloTP(bool verbose)
{
  verbose_ = verbose;
}

L1Analysis::L1AnalysisCaloTP::~L1AnalysisCaloTP()
{

}

void L1Analysis::L1AnalysisCaloTP::SetHCAL( const HcalTrigPrimDigiCollection& hcalTPs ) {

  if (verbose_) edm::LogInfo("L1Ntuple") << "HCAL TPs : " << hcalTPs.size() << std::endl;

  for (unsigned i=0; i<hcalTPs.size(); ++i) {

    short ieta = (short) hcalTPs[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    short sign = ieta/absIeta;

    unsigned short cal_iphi = (unsigned short) hcalTPs[i].id().iphi();
    unsigned short iphi = (72 + 18 - cal_iphi) % 72;
    if (absIeta >= 29) {  // special treatment for HF
      iphi = iphi/4;
    }

    unsigned short compEt = hcalTPs[i].SOI_compressedEt();
    double et = 0.;
    if (hcalScale_!=0) et = hcalScale_->et( compEt, absIeta, sign );

    unsigned short fineGrain = (unsigned short) hcalTPs[i].SOI_fineGrain();

    tp_.hcalTPieta.push_back( ieta );
    tp_.hcalTPCaliphi.push_back( cal_iphi );
    tp_.hcalTPiphi.push_back( iphi );
    tp_.hcalTPet.push_back( et );
    tp_.hcalTPcompEt.push_back( compEt );
    tp_.hcalTPfineGrain.push_back( fineGrain );
    tp_.nHCALTP++;

  }

}

void L1Analysis::L1AnalysisCaloTP::SetECAL( const EcalTrigPrimDigiCollection& ecalTPs ) {
  
  if (verbose_) edm::LogInfo("L1Ntuple") << "ECAL TPs : " << ecalTPs.size() << std::endl;

  for (unsigned i=0; i<ecalTPs.size(); ++i) {

    short ieta = (short) ecalTPs[i].id().ieta(); 
    unsigned short absIeta = (unsigned short) abs(ieta);
    short sign = ieta/absIeta;
    
    unsigned short cal_iphi = (unsigned short) ecalTPs[i].id().iphi(); 
    unsigned short iphi = (72 + 18 - cal_iphi) % 72; // transform TOWERS (not regions) into local rct (intuitive) phi bins
    
    unsigned short compEt = ecalTPs[i].compressedEt();
    double et = 0.;
    if (ecalScale_!=0) et = ecalScale_->et( compEt, absIeta, sign );
    
    unsigned short fineGrain = (unsigned short) ecalTPs[i].fineGrain();  // 0 or 1
    
    tp_.ecalTPieta.push_back( ieta );
    tp_.ecalTPCaliphi.push_back( cal_iphi );
    tp_.ecalTPiphi.push_back( iphi );
    tp_.ecalTPet.push_back( et );
    tp_.ecalTPcompEt.push_back( compEt );
    tp_.ecalTPfineGrain.push_back( fineGrain );
    tp_.nECALTP++;
    
  }  

  
}
