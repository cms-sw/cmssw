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
// $Id: SiStripQuality.h,v 1.1 2007/10/08 17:30:47 giordano Exp $
//


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <vector>


class SiStripQuality: public SiStripBadStrip {

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
  
  SiStripQuality();
  SiStripQuality(edm::FileInPath&);
  SiStripQuality(const SiStripBadStrip* );
  SiStripQuality(const SiStripBadStrip*, edm::FileInPath&);


  ~SiStripQuality(){
    delete reader;
  };

  void clear(){
    v_badstrips.clear();
    indexes.clear();
  }
 
  void add(const SiStripBadStrip*);

  bool cleanUp();

  void fillBadComponents();

  //------- Interface for the user ----------//

  bool IsModuleBad(const uint32_t& detid) const;
  bool IsFiberBad(const uint32_t& detid, const short& fiberNb) const;
  bool IsApvBad(const uint32_t& detid, const short& apvNb) const;
  bool IsStripBad(const uint32_t& detid, const short& strip) const;
  
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

 private:

  void compact(std::vector<unsigned int>&,std::vector<unsigned int>&,unsigned short&);
  bool put_replace(const uint32_t& DetId, Range input);

  SiStripDetInfoFileReader* reader;
  edm::FileInPath fileInpath;

  bool toCleanUp;

  std::vector<BadComponent> BadComponentVect;
};

#endif
