#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

template<class Item>
void getBarrelErrorSummary(typename std::vector<Item> vItems,long unsigned int & errorCount){
	unsigned int stcode = 0;
    for(typename std::vector<Item>::const_iterator iItems = vItems.begin();
	 iItems != vItems.end(); ++iItems){
	
		stcode=iItems->getStatusCode();
		if(stcode!=0)
			errorCount ++;

	}
}

template <class Item>
void getEndCapErrorSummary(typename std::vector<Item> vItems,long unsigned int & errorCountEE1,
	long unsigned int & errorCountEE2,long unsigned int & totalEE1,long unsigned int & totalEE2){

	unsigned int stcode = 0;
	EEDetId endcapId;
	long unsigned int count=0;

	for(typename std::vector<Item>::const_iterator iItems = vItems.begin();
	 iItems != vItems.end(); ++iItems){
	
		stcode=iItems->getStatusCode();
		if(stcode!=0){
			endcapId = EEDetId::detIdFromDenseIndex(count);
			if(endcapId.zside() == -1) errorCountEE1++;
			if(endcapId.zside() == +1) errorCountEE2++;
		}

		if(endcapId.zside() == -1) totalEE1++;
		if(endcapId.zside() == +1) totalEE2++;
		count++;
    }


}