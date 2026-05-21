#include "ZDCHardCodeHelper.h"

ZDCHardCodeHelper::ZDCHardCodeHelper(){setLowEdges(nbins_, binMin2);};

ZDCHardCodeHelper::~ZDCHardCodeHelper(){};


double ZDCHardCodeHelper::charge( unsigned fAdc, unsigned fCapId){
  unsigned range = range_(fAdc);
  unsigned index = 4*fCapId + range;
  return (center(fAdc) - exact_offsets[index]) / exact_slopes[index];
  // return (center(fAdc) -LUT_array.offsets[0][0][0][index])/ LUT_array.slopes[0][0][0][index];
}

double ZDCHardCodeHelper::charge(const QIE10DataFrame& digi, int ts){
  return charge( digi[ts].adc(), digi[ts].capid());
}


int ZDCHardCodeHelper::rechit_Energy_TriggerBit_EM(const QIE10DataFrame& digi){
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double noise = 0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = EM_Peds[0][channel-1]; width = EM_PedWidths[0][channel-1]; gain = EM_Gains[0][channel-1];}
   else {ped = EM_Peds[1][channel-1]; width = EM_PedWidths[1][channel-1]; gain = EM_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   noise = ootpuFrac*subPedestal(charge(digi, noiseTs_), ped, width)*gain;
   int int_Energy = std::min(std::max(0, int(energy/(50.0))), 1023);
   int int_Noise = std::min(std::max(0, int(noise/(50.0))), 1023);
   return(std::max(0, int_Energy- int_Noise));
}

int ZDCHardCodeHelper::rechit_Energy_TriggerBit_HAD(const QIE10DataFrame& digi){
     
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double noise = 0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = HAD_Peds[0][channel-1]; width = HAD_PedWidths[0][channel-1]; gain = HAD_Gains[0][channel-1];}
   else {ped = HAD_Peds[1][channel-1]; width = HAD_PedWidths[1][channel-1]; gain = HAD_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   noise = ootpuFrac*subPedestal(charge(digi, noiseTs_), ped, width)*gain;
   int int_Energy = std::min(std::max(0, int(energy/(50.0))), 1023);
   int int_Noise = std::min(std::max(0, int(noise/(50.0))), 1023);
   return(std::max(0, int_Energy- int_Noise));
}

double ZDCHardCodeHelper::rechit_Energy_Trigger_EM(const QIE10DataFrame& digi){
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double noise = 0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = EM_Peds[0][channel-1]; width = EM_PedWidths[0][channel-1]; gain = EM_Gains[0][channel-1];}
   else {ped = EM_Peds[1][channel-1]; width = EM_PedWidths[1][channel-1]; gain = EM_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   noise = ootpuFrac*subPedestal(charge(digi, noiseTs_), ped, width)*gain;
   return(std::max(0.0, energy- noise));
}

double ZDCHardCodeHelper::rechit_Energy_Trigger_HAD(const QIE10DataFrame& digi){
     
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double noise = 0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = HAD_Peds[0][channel-1]; width = HAD_PedWidths[0][channel-1]; gain = HAD_Gains[0][channel-1];}
   else {ped = HAD_Peds[1][channel-1]; width = HAD_PedWidths[1][channel-1]; gain = HAD_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   noise = ootpuFrac*subPedestal(charge(digi, noiseTs_), ped, width)*gain;
   return(std::max(0.0, energy- noise));
}


double ZDCHardCodeHelper::rechit_Energy_RPD(const QIE10DataFrame& digi){
     
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = RPD_Peds[0][channel-1]; width = RPD_PedWidths[0][channel-1]; gain = RPD_Gains[0][channel-1];}
   else {ped = RPD_Peds[1][channel-1]; width = RPD_PedWidths[1][channel-1]; gain = RPD_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   return(energy);
}

double ZDCHardCodeHelper::rechit_Time(const QIE10DataFrame& digi){
   double time = -9999.0;
   if(signalTs_ > 0 && signalTs_ <digi.samples() -1 ){
      
      
      double tmp_energy = 0;
      double weight_energy = 0;
      double total_energy = 0;
      for( int i = -1; i <=1 ; ++i){
         tmp_energy = std::max(0.0, charge(digi, signalTs_ +i));
         weight_energy += tmp_energy*(signalTs_ +i);
         total_energy  += tmp_energy;
      
      }
      if(total_energy >0) time = 25.0*(weight_energy / total_energy);
   }
   return(time);
}
double ZDCHardCodeHelper::rechit_TDCtime(const QIE10DataFrame& digi){
    float tmp_tdctime = 0;
    int le_tdc = digi[signalTs_].le_tdc();
    // TDC error codes will be 60=-1, 61 = -2, 62 = -3, 63 = -4
    if (le_tdc >= 60)
      tmp_tdctime = -1 * (le_tdc - 59);
    else
      tmp_tdctime = signalTs_ * 25. + (le_tdc / 2.0);
   return(tmp_tdctime);
}

double ZDCHardCodeHelper::rechit_ChargeWeightedTime(const QIE10DataFrame& digi){
   double time = -99.0;
      
   double tmp_energy = 0;
   double weight_energy = 0;
   double total_energy = 0;
   for( int i = 0; i <digi.samples() ; ++i){
      tmp_energy = std::max(0.0, charge(digi, i));
      weight_energy+= tmp_energy*(i);
      total_energy += tmp_energy;
   }
   if(total_energy >0) time = 25.0*(weight_energy / total_energy);
   return(time);
}

double ZDCHardCodeHelper::rechit_EnergySOIp1(const QIE10DataFrame& digi){
     
   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double energy = 0;
   double gain = 0;
   if(zside < 0) { gain = RPD_Gains[0][channel-1];}
   else { gain = RPD_Gains[1][channel-1];}
   energy = subPedestal(charge(digi, signalTs_ +1), 0.0, 0.0)*gain;
   return(energy);
}
double ZDCHardCodeHelper::rechit_RatioSOIp1(const QIE10DataFrame& digi){

   HcalZDCDetId zdcid = digi.id();
   int zside = zdcid.zside();
   int channel = zdcid.channel();
   
   double ratio = -1.0;
   double ped =0;
   double width =0;
   double gain = 0;
   if(zside < 0) {ped = RPD_Peds[0][channel-1]; width = RPD_PedWidths[0][channel-1]; gain = RPD_Gains[0][channel-1];}
   else {ped = RPD_Peds[1][channel-1]; width = RPD_PedWidths[1][channel-1]; gain = RPD_Gains[0][channel-1];}
   double energy0 = subPedestal(charge(digi, signalTs_), ped, width)*gain;
   double energy1 = subPedestal(charge(digi, signalTs_+1), ped, width)*gain;
   if(energy0 > 0 && energy1 > 0) ratio = energy0/ energy1;
   return(ratio);
}

int ZDCHardCodeHelper::rechit_Saturation(const QIE10DataFrame& digi){
   int isSaturated = 0;
    for (int i =0; i < digi.samples(); i++) {
      if (digi[i].adc() >= maxValue_) {
        isSaturated = 1;
        break;
      }
    }
    return(isSaturated);
}