#ifndef DataFormats_RPCDigi_RPCRawSynchro_H
#define DataFormats_RPCDigi_RPCRawSynchro_H

#include <map>
#include <vector>
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"

class RPCRawSynchro {
public:

  typedef std::vector< std::pair<LinkBoardElectronicIndex, int> > ProdItem;

  RPCRawSynchro() {}

  void add(const ProdItem & item);

  struct lessLB { bool operator () (const LinkBoardElectronicIndex& lb1, 
      const LinkBoardElectronicIndex& lb2) const;
  };

private:
  typedef std::map<LinkBoardElectronicIndex, std::vector<int>, RPCRawSynchro::lessLB > LBCountMap; 
  LBCountMap theSynchroCounts;

};
#endif

