#ifndef RPCEMap_H
#define RPCEMap_H

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <boost/cstdint.hpp>

class RPCEMap {
public:

  RPCEMap(const std::string & version = "")
  : theVersion(version) { }

  virtual ~RPCEMap(){}

  std::string theVersion;

  struct dccItem {
    int theId;
    int nTBs;
  };
  struct tbItem {
    int theNum;
    uint32_t theMaskedLinks;
    int nLinks;
  };
  struct linkItem {
    int theTriggerBoardInputNumber;
    int nLBs;
  };
  struct lbItem {
    bool theMaster;
    int theLinkBoardNumInLink;
    int nFebs;
  };
  struct febItem {
    int theLinkBoardInputNum;
    mutable uint32_t theRawId;
    std::string cmsEtaPartition;
    int positionInCmsEtaPartition;
    std::string localEtaPartition;
    int positionInLocalEtaPartition;
    int diskOrWheel;
    int layer;
    int sector;
    std::string subsector;
    std::string chamberLocationName;
    std::string febZOrnt;
    std::string febZRadOrnt;
    std::string barrelOrEndcap;
    int nStrips;
  };
  struct stripItem {
    int cablePinNumber;
    int chamberStripNumber;
    int cmsStripNumber;
  };

  std::vector<dccItem> theDccs;
  std::vector<tbItem> theTBs;
  std::vector<linkItem> theLinks;
  std::vector<lbItem> theLBs;
  std::vector<febItem> theFebs;
  std::vector<stripItem> theStrips;

  RPCReadOutMapping* convert() const {
    RPCReadOutMapping* cabling = new RPCReadOutMapping(theVersion);
    int lastTB=0;
    int lastLink=0;
    int lastLB=0;
    int lastFeb=0;
    int lastStrip=0;
    for (unsigned int idcc=0; idcc<theDccs.size(); idcc++) {
      DccSpec dcc(theDccs[idcc].theId);
      for (int itb=lastTB; itb<lastTB+theDccs[idcc].nTBs; itb++) {
        TriggerBoardSpec tb(theTBs[itb].theNum);
        for (int ilink=lastLink; ilink<lastLink+theTBs[itb].nLinks; ilink++) {
          LinkConnSpec lc(theLinks[ilink].theTriggerBoardInputNumber);
          for (int ilb=lastLB; ilb<lastLB+theLinks[ilink].nLBs; ilb++) {
            LinkBoardSpec lb(theLBs[ilb].theMaster,theLBs[ilb].theLinkBoardNumInLink);
            for (int ifeb=lastFeb; ifeb<lastFeb+theLBs[ilb].nFebs; ifeb++) {
              ChamberLocationSpec chamber={theFebs[ifeb].diskOrWheel,theFebs[ifeb].layer,theFebs[ifeb].sector,theFebs[ifeb].subsector,theFebs[ifeb].chamberLocationName,theFebs[ifeb].febZOrnt,theFebs[ifeb].febZRadOrnt, theFebs[ifeb].barrelOrEndcap};
              FebLocationSpec afeb={theFebs[ifeb].cmsEtaPartition,theFebs[ifeb].positionInCmsEtaPartition,theFebs[ifeb].localEtaPartition,theFebs[ifeb].positionInLocalEtaPartition};
              FebConnectorSpec febConnector(theFebs[ifeb].theLinkBoardInputNum,chamber,afeb);
              for (int istrip=lastStrip; istrip<lastStrip+theFebs[ifeb].nStrips; istrip++) {
                ChamberStripSpec strip={theStrips[istrip].cablePinNumber,theStrips[istrip].chamberStripNumber,theStrips[istrip].cmsStripNumber};
                febConnector.add(strip);
              }
              lb.add(febConnector);
              lastStrip+=theFebs[ifeb].nStrips;
            }
            lc.add(lb);
            lastFeb+=theLBs[ilb].nFebs;
          }
          tb.add(lc);
          lastLB+=theLinks[ilink].nLBs;
        }
        dcc.add(tb);
        lastLink+=theTBs[itb].nLinks;
      }
      cabling->add(dcc);
      lastTB+=theDccs[idcc].nTBs;
    }
  return cabling;
};
  
private:
  
};

#endif // RPCEMap_H

