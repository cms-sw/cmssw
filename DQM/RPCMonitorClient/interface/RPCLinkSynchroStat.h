#ifndef DQM_RPCMonitorClient_RPCLinkSynchroStat_H
#define DQM_RPCMonitorClient_RPCLinkSynchroStat_H

#include <map>
#include <cmath>
#include <vector>
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"

class RPCReadOutMapping;

class RPCLinkSynchroStat {
public:

  RPCLinkSynchroStat(bool useFirstHitOnly);

  virtual ~RPCLinkSynchroStat(){}

  void init(const RPCReadOutMapping* theCabling, bool addChamberInfo);

  void add(const RPCRawSynchro::ProdItem & counts, std::vector<LinkBoardElectronicIndex> & problems);

  void add(const std::string & lbName, const unsigned int *hits);

  std::string dumpDelays();

protected:

  class LinkBoard {
  public:
    LinkBoard(const std::string & n) : theName(n) {}
    const std::string & name() const { return theName; }
    typedef std::pair<std::string,std::string> ChamberAndPartition;
    int add(const ChamberAndPartition & part);
    int add(const LinkBoardElectronicIndex & ele);
    const std::vector<LinkBoardElectronicIndex> & paths() const { return theElePaths; } 
    const std::vector<ChamberAndPartition> & chamberAndPartitions() const { return theChamberAndPartitions; }
    bool operator<(const LinkBoard &o) const {return theName < o.theName; }
    bool operator==(const LinkBoard &o) const { return (theName == o.theName); }
  private:
    std::string theName;
    std::vector<ChamberAndPartition> theChamberAndPartitions;
    std::vector<LinkBoardElectronicIndex> theElePaths;
  };

  class SynchroCounts {
  public:
    SynchroCounts() : theCounts(std::vector<unsigned int>(8,0)) {}
    SynchroCounts(const unsigned int *hits) : theCounts(std::vector<unsigned int>(hits,hits+8)) {}
    
    void increment(unsigned int bxDiff);
    void set(unsigned int bxDiff);
    unsigned int firstHit() const;

    double rms() const;
    double mean() const;
    unsigned int sum() const { return mom0(); }

    std::string print() const;
    const std::vector<unsigned int> & counts() const { return theCounts; }
    bool operator==(const SynchroCounts &) const;
    SynchroCounts & operator+=(const SynchroCounts &rhs);
  private:
    unsigned int  mom0() const;
    double mom1() const;
    std::vector<unsigned int> theCounts;
  };

  typedef std::pair<LinkBoard, SynchroCounts> BoardAndCounts;
  struct LessLinkName { bool operator()(const BoardAndCounts & o1, const BoardAndCounts & o2); };
  struct LessCountSum { bool operator()(const BoardAndCounts & o1, const BoardAndCounts & o2); };
  struct ShortLinkInfo{ unsigned int idx; std::vector<unsigned int> hit_paths; SynchroCounts counts; };

  bool theUseFirstHitOnly;

  static const unsigned int MAXDCCINDEX=2;
  static const unsigned int DCCINDEXSHIFT=790;
  static const unsigned int MAXRBCINDEX=35;
  static const unsigned int MAXLINKINDEX=17;
  static const unsigned int MAXLBINDEX=2; 
  unsigned int theLinkStatNavi[MAXDCCINDEX+1][MAXRBCINDEX+1][MAXLINKINDEX+1][MAXLBINDEX+1];
  std::vector<BoardAndCounts> theLinkStatMap;
  
  friend class RPCLinkSynchroHistoMaker;
};
#endif
