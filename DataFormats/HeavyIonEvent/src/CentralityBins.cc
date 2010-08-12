#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include <iostream>
using namespace std;

int CentralityBins::getBin(double value) const {

   int bin = table_.size() - 1;
   for(unsigned int i = 0; i < table_.size(); ++i){
      if(value >= table_[i].bin_edge){
	 bin = i;
	 return bin;
      }
   }

   return bin;
}

CentralityBins::RunMap getCentralityFromFile(TFile* file, const char* tag, int firstRun, int lastRun){

   CentralityBins::RunMap map;
   for(int run = firstRun; run<= lastRun; ++run){
      const CentralityBins* table = (const CentralityBins*)file->Get(Form("%s/run%d",tag,run));
      if(table) map.insert(std::pair<int,const CentralityBins*>(run,table));
   }
   return map;
}

ClassImp(CBin)
ClassImp(CentralityBins)





