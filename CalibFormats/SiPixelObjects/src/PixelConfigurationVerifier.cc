//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelConfigurationVerifier.h"
#include "CalibFormats/SiPixelObjects/interface/PixelChannel.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"
#include <set>
#include <cassert>

using namespace pos;
using namespace std;

void PixelConfigurationVerifier::checkChannelEnable(PixelFEDCard *theFEDCard,
                                                    PixelNameTranslation *theNameTranslation,
                                                    PixelDetectorConfig *theDetConfig) {
  std::string mthn = "[PixelConfigurationVerifier::checkChannelEnable()]\t\t    ";
  set<PixelChannel> channels = theNameTranslation->getChannels(*theDetConfig);

  unsigned int fedid = theFEDCard->fedNumber;

  //use slots 1-36

  vector<bool> usedChannel(37);
  for (unsigned int i = 1; i < 37; i++) {
    usedChannel[i] = false;
  }

  set<PixelChannel>::const_iterator iChannel = channels.begin();

  map<int, int> nrocs;
  for (; iChannel != channels.end(); ++iChannel) {
    PixelHdwAddress hdw = theNameTranslation->getHdwAddress(*iChannel);
    if (fedid == hdw.fednumber()) {
      unsigned int fedchannel = hdw.fedchannel();
      assert(fedchannel > 0 && fedchannel < 37);
      usedChannel[fedchannel] = true;
      nrocs[fedchannel] = theNameTranslation->getROCsFromChannel(*iChannel).size();
    }
  }

  map<PixelROCName, PixelROCStatus> roclistcopy = theDetConfig->getROCsList();
  //Now check the channels

  for (unsigned int jChannel = 1; jChannel < 37; jChannel++) {
    bool used = theFEDCard->useChannel(jChannel);
    //    if (!used) cout << "Channel="<<jChannel<<" is not used"<<endl;
    if (used) {
      //            cout << "Channel="<<jChannel<<" is used"<<endl;
      //check that nROCs is the same from theNameTranslation and theFEDCard
      if (nrocs[jChannel] != theFEDCard->NRocs[jChannel - 1]) {
        cout << "[PixelConfigurationVerifier] Warning in FED#" << fedid << ", channel#" << jChannel
             << ": number of ROCs mismatch: theNameTranslation=" << nrocs[jChannel]
             << "; theFEDCard=" << theFEDCard->NRocs[jChannel - 1] << endl;
      }
    }

    //only do these checks if the channel exists
    if (theNameTranslation->FEDChannelExist(fedid, jChannel)) {
      //make sure that all rocs on a channel have the same noAnalogSignal status
      vector<PixelROCName> rocsOnThisChannel = theNameTranslation->getROCsFromFEDChannel(fedid, jChannel);
      bool onehasNAS = false, onedoesnothaveNAS = false;
      vector<PixelROCName>::const_iterator jROC = rocsOnThisChannel.begin();
      for (; jROC != rocsOnThisChannel.end(); ++jROC) {
        PixelROCStatus thisROCstatus = roclistcopy[*jROC];
        if (thisROCstatus.get(PixelROCStatus::noAnalogSignal))
          onehasNAS = true;
        else
          onedoesnothaveNAS = true;
      }
      if (onehasNAS && onedoesnothaveNAS) {
        cout << "[PixelConfigurationVerifier] Error in FED#" << fedid << ", channel#" << jChannel
             << ": not all ROCs have the same noAnalogSignal state." << endl;
        assert(0);
      }

      //now if onehasNAS is true, then all must be noAnalogSignal --> turn off this FED channel!
      if (onehasNAS) {
        cout << "[PixelConfigurationVerifier] FEDid=" << fedid << ", channel=" << jChannel
             << ": Channel disabled because ROCs are set to noAnalogSignal" << endl;
        theFEDCard->setChannel(jChannel, false);  //false should disable the channel
      }

      if (!onehasNAS && (used != usedChannel[jChannel])) {
        cout << __LINE__ << "]\t" << mthn << "*******************************************************" << endl;
        cout << __LINE__ << "]\t" << mthn << "WARNING for fedid=" << fedid << " and channel=" << jChannel
             << " found that fedcard has channel as " << endl;
        if (used)
          cout << __LINE__ << "]\t" << mthn << "used while configuration not using this channel" << endl;
        if (!used)
          cout << __LINE__ << "]\t" << mthn << "not used while configuration uses this channel" << endl;
        cout << __LINE__ << "]\t" << mthn << "The fedcard will be modifed to agree with configuration" << endl;
        cout << __LINE__ << "]\t" << mthn << "*******************************************************" << endl;
        theFEDCard->setChannel(jChannel, usedChannel[jChannel]);
      }
    }
  }
}
