#ifndef RecoLocalTracker_SiStripClusterizer_SiStripApvShotCleaner_H
#define RecoLocalTracker_SiStripClusterizer_SiStripApvShotCleaner_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <vector>


class SiStripApvShotCleaner{
  
 public:
  SiStripApvShotCleaner();
  
  ~SiStripApvShotCleaner(){};

  bool noShots(){return !shots_;}

  bool clean(const edmNew::DetSet<SiStripDigi>& in, edmNew::DetSet<SiStripDigi>::const_iterator& scan, edmNew::DetSet<SiStripDigi>::const_iterator& end){return false;} //FIXME
  bool clean(const    edm::DetSet<SiStripDigi>& in,    edm::DetSet<SiStripDigi>::const_iterator& scan,    edm::DetSet<SiStripDigi>::const_iterator& end);

  bool loop(const edmNew::DetSet<SiStripDigi>& in){return false;} //FIXME
  bool loop(const edm::DetSet<SiStripDigi>& in);
  
  void refresh();

  void reset(edm::DetSet<SiStripDigi>::const_iterator& a, edm::DetSet<SiStripDigi>::const_iterator& b);
  // void reset(edmNew::DetSet<SiStripDigi>::const_iterator& a, edmNew::DetSet<SiStripDigi>::const_iterator& b){;} //FIXME

 private:

  struct orderingByCharge {
    bool operator ()(SiStripDigi const& a, SiStripDigi const& b) {return a.adc() > b.adc();}
  };
  
  void subtractCM();

  void dumpInVector(edm::DetSet<SiStripDigi>::const_iterator* ,size_t );

  uint32_t cacheDetId;
  bool shots_;
  bool shotApv_[25];
  edm::DetSet<SiStripDigi>::const_iterator pFirstDigiOfApv[7];

  std::vector<SiStripDigi> vdigis,apvDigis;  //caches of digis, in case an apvshot is found
  edm::DetSet<SiStripDigi> * pDetSet;
  unsigned short maxNumOfApvs   ;  
  unsigned short stripsPerApv   ; 
  unsigned short stripsForMedian; 
 

};

#endif 
