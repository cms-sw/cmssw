//
// 
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelConfigurationVerifier.h"
#include "CalibFormats/SiPixelObjects/interface/PixelChannel.h"
#include <set>
#include <assert.h>

using namespace pos;
using namespace std;


void PixelConfigurationVerifier::checkChannelEnable(PixelFEDCard *theFEDCard,
						    PixelNameTranslation *theNameTranslation,
						    PixelDetectorConfig *theDetConfig){
  
  set<PixelChannel> channels=theNameTranslation->getChannels(*theDetConfig);

  unsigned int fedid=theFEDCard->fedNumber;

  //use slots 1-36

  vector<bool> usedChannel(37);
  for(unsigned int i=1;i<37;i++){
    usedChannel[i]=false;
  }

  set<PixelChannel>::const_iterator iChannel=channels.begin();


  map <unsigned int, unsigned int> nrocs;
  for(;iChannel!=channels.end();++iChannel){
    PixelHdwAddress hdw=theNameTranslation->getHdwAddress(*iChannel);
    if (fedid==hdw.fednumber()){
      unsigned int fedchannel=hdw.fedchannel();
      assert(fedchannel>0&&fedchannel<37);
      usedChannel[fedchannel]=true;
      nrocs[fedchannel] = theNameTranslation->getROCsFromChannel(*iChannel).size();
    }
  }

  //Now check the channels

  for(unsigned int iChannel=1;iChannel<37;iChannel++){
    bool used=theFEDCard->useChannel(iChannel);
    //if (!used) cout << "Channel="<<iChannel<<" is not used"<<endl;
    if (used) { 
      //      cout << "Channel="<<iChannel<<" is used"<<endl;
      //check that nROCs is the same from theNameTranslation and theFEDCard
      if (nrocs[iChannel] != theFEDCard->NRocs[iChannel-1]) {
	cout<<"[PixelConfigurationVerifier] Warning in FED#"<<fedid<<", channel#"<<iChannel<<": number of ROCs mismatch: theNameTranslation="<<nrocs[iChannel]<<"; theFEDCard="<<theFEDCard->NRocs[iChannel-1]<<endl;
	//	assert(nrocs[iChannel] == theFEDCard->NRocs[iChannel-1]);
      }
    }
    if (used!=usedChannel[iChannel]) {
      cout << "*******************************************************"<<endl;
      cout << "WARNING for fedid="<<fedid<<" and channel="<<iChannel;
      cout << " found that fedcard has channel as"<<endl;
      if (used) cout << "used while configuration not using this channel"
		     << endl;
      if (!used)cout << "not used while configuration uses this channel"
		     << endl;
      cout << "The fedcard will be modifed to agree with configuration"<<endl;
      cout << "*******************************************************"<<endl;
      theFEDCard->setChannel(iChannel,usedChannel[iChannel]);
    }
  }

}
