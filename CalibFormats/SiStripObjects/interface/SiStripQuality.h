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
// $Id$
//


#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <vector>


class SiStripQuality: public SiStripBadStrip {

 public:

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

  inline std::pair<unsigned short,unsigned short> decode (const unsigned int& value) const {
    return std::make_pair( ((value >> 16)  & 0xFFFF) , (value & 0xFFFF) );
  }

  inline unsigned int encode (const unsigned short& first, const unsigned short& NconsecutiveBadStrips) {
    return   ((first & 0xFFFF) << 16) | ( NconsecutiveBadStrips & 0xFFFF ) ;
  }

  bool cleanUp();

  /*
  bool IsModuleBad(const uint32_t& detid);
  bool IsFiberBad(const uint32_t& detid, const short& fiberNb);
  bool IsApvBad(const uint32_t& detid, const short& apvNb);
  */
  bool IsStripBad(const uint32_t& detid, const short& strip);
  /*
  short getBadApvs(const uint32_t& detid); 
  //each bad apv correspond to a bit to 1: num=
  //0 <-> all good apvs
  //1 <-> only apv 0 bad
  //2<-> only apv 1 bad
  //3<->  apv 0 and 1 bad
  // 4 <-> only apv 2 bad
  //...
  short getBadFibers(const uint32_t& detid); 
  //each bad fiber correspond to a bit to 1: num=
  //0 <-> all good fibers
  //1 <-> only fiber 0 bad
  //2<-> only fiber 1 bad
  //3<->  fiber 0 and 1 bad
  // 4 <-> only fiber 2 bad
  //...

  void getBadModuleList(std::vector<uint32_t>&);
  void getBadFiberList(std::vector< std::pair<uint32_t,short> >&);
  void getBadApvList(std::vector< std::pair<uint32_t,short> >&);
  */

 private:

  void compact(std::vector<unsigned int>&,std::vector<unsigned int>&,unsigned short&);
  bool put_replace(const uint32_t& DetId, Range input);

  SiStripDetInfoFileReader* reader;
  edm::FileInPath fileInpath;

  bool toCleanUp;
};

#endif
