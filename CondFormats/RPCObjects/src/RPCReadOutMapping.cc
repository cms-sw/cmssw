#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <iostream>

using namespace edm;

RPCReadOutMapping::RPCReadOutMapping(const std::string &version) : theVersion(version) {}

const DccSpec *RPCReadOutMapping::dcc(int dccId) const {
  IMAP im = theFeds.find(dccId);
  const DccSpec &ddc = (*im).second;
  return (im != theFeds.end()) ? &ddc : nullptr;
}

void RPCReadOutMapping::add(const DccSpec &dcc) { theFeds[dcc.id()] = dcc; }

std::vector<const DccSpec *> RPCReadOutMapping::dccList() const {
  std::vector<const DccSpec *> result;
  result.reserve(theFeds.size());
  for (IMAP im = theFeds.begin(); im != theFeds.end(); im++) {
    result.push_back(&(im->second));
  }
  return result;
}

std::pair<int, int> RPCReadOutMapping::dccNumberRange() const {
  if (theFeds.empty())
    return std::make_pair(0, -1);
  else {
    IMAP first = theFeds.begin();
    IMAP last = theFeds.end();
    last--;
    return std::make_pair(first->first, last->first);
  }
}

std::vector<std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> > RPCReadOutMapping::rawDataFrame(
    const StripInDetUnit &stripInDetUnit) const {
  std::vector<std::pair<LinkBoardElectronicIndex, LinkBoardPackedStrip> > result;
  LinkBoardElectronicIndex eleIndex = {0, 0, 0, 0};

  const uint32_t &rawDetId = stripInDetUnit.first;
  const int &stripInDU = stripInDetUnit.second;

  for (IMAP im = theFeds.begin(); im != theFeds.end(); im++) {
    const DccSpec &dccSpec = (*im).second;
    const std::vector<TriggerBoardSpec> &triggerBoards = dccSpec.triggerBoards();
    for (std::vector<TriggerBoardSpec>::const_iterator it = triggerBoards.begin(); it != triggerBoards.end(); it++) {
      const TriggerBoardSpec &triggerBoard = (*it);
      typedef std::vector<const LinkConnSpec *> LINKS;
      LINKS linkConns = triggerBoard.enabledLinkConns();
      for (LINKS::const_iterator ic = linkConns.begin(); ic != linkConns.end(); ic++) {
        const LinkConnSpec &link = **ic;
        const std::vector<LinkBoardSpec> &boards = link.linkBoards();
        for (std::vector<LinkBoardSpec>::const_iterator ib = boards.begin(); ib != boards.end(); ib++) {
          const LinkBoardSpec &board = (*ib);

          eleIndex.dccId = dccSpec.id();
          eleIndex.dccInputChannelNum = triggerBoard.dccInputChannelNum();
          eleIndex.tbLinkInputNum = link.triggerBoardInputNumber();
          eleIndex.lbNumInLink = board.linkBoardNumInLink();

          const std::vector<FebConnectorSpec> &febs = board.febs();
          for (std::vector<FebConnectorSpec>::const_iterator ifc = febs.begin(); ifc != febs.end(); ifc++) {
            const FebConnectorSpec &febConnector = (*ifc);
            if (febConnector.rawId() != rawDetId)
              continue;
            int febInLB = febConnector.linkBoardInputNum();
            for (int istrip = 0; istrip < febConnector.nstrips(); istrip++) {
              int stripPinInFeb = febConnector.cablePinNum(istrip);
              if (febConnector.chamberStripNum(istrip) == stripInDU) {
                result.push_back(std::make_pair(eleIndex, LinkBoardPackedStrip(febInLB, stripPinInFeb)));
              }
            }
          }
        }
      }
    }
  }
  return result;
}

const LinkBoardSpec *RPCReadOutMapping::location(const LinkBoardElectronicIndex &ele) const {
  //FIXME after debugging change to dcc(ele.dccId)->triggerBoard(ele.dccInputChannelNum)->...
  const DccSpec *dcc = RPCReadOutMapping::dcc(ele.dccId);
  if (dcc) {
    const TriggerBoardSpec *tb = dcc->triggerBoard(ele.dccInputChannelNum);
    if (tb) {
      const LinkConnSpec *lc = tb->linkConn(ele.tbLinkInputNum);
      if (lc) {
        const LinkBoardSpec *lb = lc->linkBoard(ele.lbNumInLink);
        return lb;
      }
    }
  }
  return nullptr;
}

RPCReadOutMapping::StripInDetUnit RPCReadOutMapping::detUnitFrame(const LinkBoardSpec &location,
                                                                  const LinkBoardPackedStrip &lbstrip) const {
  uint32_t detUnit = 0;
  int stripInDU = 0;
  int febInLB = lbstrip.febInLB();
  int stripPinInFeb = lbstrip.stripPinInFeb();

  const FebConnectorSpec *feb = location.feb(febInLB);
  if (feb) {
    detUnit = feb->rawId();
    const ChamberStripSpec strip = feb->strip(stripPinInFeb);
    if (strip.chamberStripNumber > -1) {
      stripInDU = strip.chamberStripNumber;
    } else {
      // LogWarning("detUnitFrame")<<"problem with stip for febInLB: "<<febInLB
      //                             <<" strip pin: "<< stripPinInFeb
      //                             <<" strip pin: "<< stripPinInFeb;
      LogDebug("") << "problem with stip for febInLB: " << febInLB << " strip pin: " << stripPinInFeb
                   << " strip pin: " << stripPinInFeb << " for linkBoard: " << location.print(3);
    }
  } else {
    // LogWarning("detUnitFrame")<<"problem with detUnit for febInLB: ";
    LogDebug("") << "problem with detUnit for febInLB: " << febInLB << " for linkBoard: " << location.print(1);
  }
  return std::make_pair(detUnit, stripInDU);
}

//
// ALL BELOW IS TEMPORARY, TO BE REMOVED !!!!
//

std::pair<LinkBoardElectronicIndex, int> RPCReadOutMapping::getRAWSpecForCMSChamberSrip(uint32_t detId,
                                                                                        int strip,
                                                                                        int dccInputChannel) const {
  LinkBoardElectronicIndex linkboard;
  linkboard.dccId = 790;
  linkboard.dccInputChannelNum = dccInputChannel;

  for (int k = 0; k < 18; k++) {
    linkboard.tbLinkInputNum = k;
    for (int j = 0; j < 3; j++) {
      linkboard.lbNumInLink = j;
      const LinkBoardSpec *location = this->location(linkboard);
      if (location) {
        for (int i = 1; i < 7; i++) {
          const FebConnectorSpec *feb = location->feb(i);
          if (feb && feb->rawId() == detId) {
            for (int l = 1; l < 17; l++) {
              int pin = l;
              const ChamberStripSpec aStrip = feb->strip(pin);
              if (aStrip.cmsStripNumber == strip) {
                int bitInLink = (i - 1) * 16 + l - 1;
                std::pair<LinkBoardElectronicIndex, int> stripInfo(linkboard, bitInLink);
                return stripInfo;
              }
            }
          }
        }
      }
    }
  }
  RPCDetId aDet(detId);
  std::cout << "Strip: " << strip << " not found for detector: " << aDet << std::endl;
  std::pair<LinkBoardElectronicIndex, int> dummyStripInfo(linkboard, -99);
  return dummyStripInfo;
}

std::vector<const LinkBoardSpec *> RPCReadOutMapping::getLBforChamber(const std::string &name) const {
  std::vector<const LinkBoardSpec *> vLBforChamber;

  LinkBoardElectronicIndex linkboard;
  linkboard.dccId = 790;
  linkboard.dccInputChannelNum = 1;
  linkboard.tbLinkInputNum = 1;
  linkboard.lbNumInLink = 0;
  const LinkBoardSpec *location = this->location(linkboard);

  for (int k = 0; k < 18; k++) {
    linkboard.dccInputChannelNum = 1;
    linkboard.tbLinkInputNum = k;
    for (int j = 0; j < 3; j++) {
      linkboard.lbNumInLink = j;
      int febInputNum = 1;
      location = this->location(linkboard);
      if (location) {
        //location->print();
        for (int j = 0; j < 6; j++) {
          const FebConnectorSpec *feb = location->feb(febInputNum + j);
          if (feb) {
            //feb->print();
            std::string chName = feb->chamber().chamberLocationName();
            if (chName == name) {
              vLBforChamber.push_back(location);
              //feb->chamber().print();
              break;
            }
          }
        }
      }
    }
  }
  return vLBforChamber;
}
