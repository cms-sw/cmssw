#ifndef RPCEMap_H
#define RPCEMap_H

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <iostream>
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
    int nLinks;
  };
  struct linkItem {
    int theTriggerBoardInputNumber;
    int nLBs;
  };
  struct lbItem {
    bool theMaster;
    int theLinkBoardNumInLink;
    int theCode;
    int nFebs;
  };
  struct febItem {
    int theLinkBoardInputNum;
    int thePartition;
    int theChamber;
    int theAlgo;
  };

  std::vector<dccItem> theDccs;
  std::vector<tbItem> theTBs;
  std::vector<linkItem> theLinks;
  std::vector<lbItem> theLBs;
  std::vector<febItem> theFebs;

  RPCReadOutMapping* convert() const {
    RPCReadOutMapping* cabling = new RPCReadOutMapping(theVersion);
    int diskOffset=4;
    int year=atoi(theVersion.substr(6,4).c_str());
    int month=atoi(theVersion.substr(3,2).c_str());
    if (year < 2012 || (year==2012 && month<11)) diskOffset=3;
    int lastTB=0;
    int lastLink=0;
    int lastLB=0;
    int lastFeb=0;
    for (unsigned int idcc=0; idcc<theDccs.size(); idcc++) {
      DccSpec dcc(theDccs[idcc].theId);
      for (int itb=lastTB; itb<lastTB+theDccs[idcc].nTBs; itb++) {
        TriggerBoardSpec tb(theTBs[itb].theNum);
        for (int ilink=lastLink; ilink<lastLink+theTBs[itb].nLinks; ilink++) {
          LinkConnSpec lc(theLinks[ilink].theTriggerBoardInputNumber);
          for (int ilb=lastLB; ilb<lastLB+theLinks[ilink].nLBs; ilb++) {
            LinkBoardSpec lb(theLBs[ilb].theMaster,theLBs[ilb].theLinkBoardNumInLink,theLBs[ilb].theCode);
            for (int ifeb=lastFeb; ifeb<lastFeb+theLBs[ilb].nFebs; ifeb++) {
              int sector=(theFebs[ifeb].theChamber)%100;
              char subsector=((theFebs[ifeb].theChamber)/100)%10-2;
              char febZRadOrnt=((theFebs[ifeb].theChamber)/1000)%5;
              char febZOrnt=((theFebs[ifeb].theChamber)/5000)%2;
              char diskOrWheel=((theFebs[ifeb].theChamber)/10000)%10-diskOffset;
              char layer=((theFebs[ifeb].theChamber)/100000)%10;
              char barrelOrEndcap=(theFebs[ifeb].theChamber)/1000000;
              ChamberLocationSpec chamber={diskOrWheel,layer,sector,subsector,febZOrnt,febZRadOrnt,barrelOrEndcap};
              char cmsEtaPartition=(theFebs[ifeb].thePartition)/1000;
              char positionInCmsEtaPartition=((theFebs[ifeb].thePartition)%1000)/100;
              char localEtaPartition=((theFebs[ifeb].thePartition)%100)/10;
              char positionInLocalEtaPartition=(theFebs[ifeb].thePartition)%10;
              FebLocationSpec afeb={cmsEtaPartition,positionInCmsEtaPartition,localEtaPartition,positionInLocalEtaPartition};
              FebConnectorSpec febConnector(theFebs[ifeb].theLinkBoardInputNum,chamber,afeb);
              febConnector.addStrips(theFebs[ifeb].theAlgo);
              lb.add(febConnector);
//              std::cout<<"End of FEB"<<std::endl;
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

