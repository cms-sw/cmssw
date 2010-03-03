#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"

int CentralityBins::getBin(double value) const {

   if(value < 0) return -1;

   int bin = 0;
   for(int i = 0; i < table_.size(); ++i){
      if(value > table_[i].bin_edge){
	 bin = i;
	 return bin;
      }
   }

   return bin;
}

ClassImp(CBin)
ClassImp(CentralityBins)





