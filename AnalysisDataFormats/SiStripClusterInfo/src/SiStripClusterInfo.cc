#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"

void SiStripClusterInfo::print(std::stringstream &ss){
  
  ss << "\nDetId       " << detId_
     << "\nCharge      " << Charge 
     << "\nNoise       " << Noise	
     << "\nPosition    " << Position 
     << "\nWidth       " << Width 
     << "\nMaxCharge   " << MaxCharge  	
     << "\nMaxPosition " << MaxPosition
     << "\nChargeL     " << ChargeL 
     << "\nChargeR     " << ChargeR  	
     << "\nFirstStrip  " << FirstStrip
    ;

  ss << "\nAmplitudes  ";
  for (size_t i=0;i<StripAmplitudes.size();i++)
    ss << StripAmplitudes[i] << " \t ";

  ss << "\nRawDigiAmplitudesL  ";
  for (size_t i=0;i<RawDigiAmplitudesL.size();i++)
    ss << RawDigiAmplitudesL[i] << " \t ";

  ss << "\nRawDigiAmplitudesR  ";
  for (size_t i=0;i<RawDigiAmplitudesR.size();i++)
    ss << RawDigiAmplitudesR[i] << " \t ";

  ss << "\nStripNoises  ";
  for (size_t i=0;i<StripNoises.size();i++)
    ss << StripNoises[i] << " \t ";
}

float SiStripClusterInfo::getGainForStripNb(uint16_t istrip) const{

  short apv_of_istrip = (int) ((FirstStrip+istrip)/128);  
  short apv_of_firstStrip = (int) (FirstStrip/128);  
  return ApvGains[apv_of_istrip-apv_of_istrip];
}
