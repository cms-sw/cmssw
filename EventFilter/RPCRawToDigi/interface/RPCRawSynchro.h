#ifndef EventFilter_RPCRawToDigi_RPCRawSynchro_H
#define EventFilter_RPCRawToDigi_RPCRawSynchro_H

#include <map>
#include <vector>
#include <string>
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
class RPCReadOutMapping;
class TH1D;
class TH2D;

class RPCRawSynchro {
public:

  typedef std::vector< std::pair<LinkBoardElectronicIndex, int> > ProdItem;

  RPCRawSynchro() {}

  static int bxDifference(const rpcrawtodigi::EventRecords & event);

  void add(const ProdItem & item);

  struct lessLB { bool operator () (const LinkBoardElectronicIndex& lb1, 
      const LinkBoardElectronicIndex& lb2) const;
  };

  std::string dumpDelays(const RPCReadOutMapping *rm, TH2D * histo) const;
  std::string dumpDccBx(int dcc, TH1D * histo) const;
private:
  typedef std::map<LinkBoardElectronicIndex, std::vector<int>, RPCRawSynchro::lessLB > LBCountMap; 
  LBCountMap theSynchroCounts;
};
#endif

