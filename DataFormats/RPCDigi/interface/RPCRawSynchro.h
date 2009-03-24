#ifndef DataFormats_RPCDigi_RPCRawSynchro_H
#define DataFormats_RPCDigi_RPCRawSynchro_H

#include <map>
#include <string>
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

  class LinkSynchroCounts {
  public:
    LinkSynchroCounts() : theCounts(std::vector<int>(8,0)) {}
    void add(int bxDiff);
    double rms() const;
    double mean() const;
    int sum() const { return mom0(); }
    std::string print() const;
    const  std::vector<int> & counts() const { return theCounts; }
  private:
    int  mom0() const;
    double mom1() const;
    std::vector<int> theCounts;
  };


  friend class RPCLinkSynchroHistoMaker;
  typedef std::map<LinkBoardElectronicIndex, LinkSynchroCounts, RPCRawSynchro::lessLB > LBCountMap; 
  LBCountMap theSynchroCounts;

};
#endif

