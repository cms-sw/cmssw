#ifndef SiStripObjects_SiStripQuality_h
#define SiStripObjects_SiStripQuality_h
// -*- C++ -*-
// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripQuality
//
/**\class SiStripQuality SiStripQuality.h SiStripQuality.cc

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Domenico Giordano
// Created:     Wed Sep 26 17:42:12 CEST 2007
//


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include <vector>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class SiStripDetCabling;
class SiStripDetInfoFileReader;

class SiStripQuality GCC11_FINAL : public SiStripBadStrip {

 public:

  struct BadComponent{
    uint32_t detid;
    unsigned short BadApvs : 6;
    unsigned short BadFibers :3;
    bool BadModule :1;
  };

  class BadComponentStrictWeakOrdering{
  public:
    bool operator() (const BadComponent& p,const uint32_t i) const {return p.detid < i;}
  };
  
  SiStripQuality(); //takes default file for SiStripDetInfoFileReader
  SiStripQuality(edm::FileInPath&);
  SiStripQuality(const SiStripQuality&); //copy constructor

  ~SiStripQuality();

  void clear(){
    v_badstrips.clear();
    indexes.clear();
    BadComponentVect.clear();
    toCleanUp=false;
  }

  void add(const uint32_t&,const SiStripBadStrip::Range&);
  void add(const SiStripBadStrip*);
  void add(const SiStripDetCabling*);
  void add(const SiStripDetVOff*);
  /**
   * Used to get the cabling from RunInfo. <br>
   * It compares the feds from cabling with those from runInfo and it
   * turns off all the strips associated to feds that are off for the RunInfo
   * but not for FedCabling.
   */
  void add(const RunInfo *);
  void addInvalidConnectionFromCabling();
  void addNotConnectedConnectionFromCabling();

  bool cleanUp(bool force=false);

  void fillBadComponents();

  void ReduceGranularity(double); 

  SiStripQuality& operator +=(const SiStripQuality&); 
  SiStripQuality& operator -=(const SiStripQuality&);
  const SiStripQuality operator -(const SiStripQuality&) const ;
  bool operator ==(const SiStripQuality&) const;
  bool operator !=(const SiStripQuality&) const;
  
  edm::FileInPath getFileInPath() const {return FileInPath_;}

  //------- Interface for the user ----------//
  bool IsModuleUsable(const uint32_t& detid) const;

  bool IsModuleBad(const uint32_t& detid) const;
  bool IsFiberBad(const uint32_t& detid, const short& fiberNb) const;
  bool IsApvBad(const uint32_t& detid, const short& apvNb) const;
  bool IsStripBad(const uint32_t& detid, const short& strip) const;
  bool IsStripBad(const Range& range, const short& strip) const;
  int  nBadStripsOnTheLeft(const Range& range, const short& strip) const; //provides number of consecutive bad strips on the left of strip (including strip)
  int  nBadStripsOnTheRight(const Range& range, const short& strip) const; //provides number of consecutive bad strips on the right of strip (including strip)
  
  short getBadApvs(const uint32_t& detid) const; 
  //each bad apv correspond to a bit to 1: num=
  //0 <-> all good apvs
  //1 <-> only apv 0 bad
  //2<-> only apv 1 bad
  //3<->  apv 0 and 1 bad
  // 4 <-> only apv 2 bad
  //...
  short getBadFibers(const uint32_t& detid) const; 
  //each bad fiber correspond to a bit to 1: num=
  //0 <-> all good fibers
  //1 <-> only fiber 0 bad
  //2<-> only fiber 1 bad
  //3<->  fiber 0 and 1 bad
  // 4 <-> only fiber 2 bad
  //...
  
  const std::vector<BadComponent>& getBadComponentList() const { return BadComponentVect; }   

  void compact(unsigned int&,std::vector<unsigned int>&);

  inline void setPrintDebugOutput(const bool printDebug) { printDebug_ = printDebug; }
  inline void setUseEmptyRunInfo(const bool useEmptyRunInfo) { useEmptyRunInfo_ = useEmptyRunInfo; }


  SiStripDetCabling const * cabling() const { return SiStripDetCabling_;}


 private:

  void compact(std::vector<unsigned int>&,std::vector<unsigned int>&,unsigned short&);
  void subtract(std::vector<unsigned int>&,const std::vector<unsigned int>&);
  void subtraction(std::vector<unsigned int>&,const unsigned int&);
  bool put_replace(const uint32_t& DetId, Range input);

  /**
   * Loop on all the fedIds, take the fedChannels and then the detId.
   * Depending on the value of a bool turn off or not the strips in the list.
   */
  void turnOffFeds(const std::vector<int> & fedsList, const bool turnOffStrips, const bool printDebug);

  /// Prints debug output for a given detId
  void printDetInfo(const uint32_t &detId, const uint32_t &apvPairNumber, std::stringstream &ss);
  /// Prints debug output for the active feds comparing the list in RunInfo and FedCabling
  void printActiveFedsInfo( const std::vector<uint16_t> & activeFedsFromCabling,
                            const std::vector<int> & activeFedsFromRunInfo,
                            const std::vector<int> & differentFeds,
                            const bool printDebug );

  bool toCleanUp;
  edm::FileInPath FileInPath_;
  SiStripDetInfoFileReader* reader;

  std::vector<BadComponent> BadComponentVect;

  const SiStripDetCabling *SiStripDetCabling_;  
  bool printDebug_;
  bool useEmptyRunInfo_;
};

#endif
