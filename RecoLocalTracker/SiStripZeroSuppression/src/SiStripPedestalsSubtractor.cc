#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"

void SiStripPedestalsSubtractor::subtract(const edm::DetSet<SiStripRawDigi>& input,std::vector<int16_t>& ssrd)
{
     edm::DetSet<SiStripRawDigi>::const_iterator iter=input.data.begin();
     uint32_t c=0;
     for (;iter!=input.data.end();iter++) {
       ssrd[c] = iter->adc() - SiStripPedestalsService_->getPedestal(input.id,c);
       //FIXME for debug, insert messagelogger
       std::cout << "[SiStripPedestalsSubtractor::subtract]: adc before sub= " 
		 << iter->adc() 
		 << "\t pedval= " << SiStripPedestalsService_->getPedestal(input.id,c)
		 << "\t adc pedsub = " << ssrd[c]
		 << std::endl;
       c++;
     }
}
