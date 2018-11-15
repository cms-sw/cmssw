#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripApvShotCleaner.h"
#include <algorithm>
#include <boost/foreach.hpp>

//Uncomment the following #define to have print debug
//#define DEBUGME

SiStripApvShotCleaner::
SiStripApvShotCleaner():
  maxNumOfApvs(6),  //FED Default: 6 (i.e. max num apvs )
  stripsPerApv(128),
  stripsForMedian(64){}



bool SiStripApvShotCleaner::
clean(const edm::DetSet<SiStripDigi>& in, edm::DetSet<SiStripDigi>::const_iterator& scan, edm::DetSet<SiStripDigi>::const_iterator& end){
  if(in.size()<64)
    return false;
  
  if(loop(in)){
    reset(scan,end);
    return true;
  }
  return false;
}

bool SiStripApvShotCleaner::
loop(const edm::DetSet<SiStripDigi>& in){

#ifdef DEBUGME 
  std::stringstream ss;  
  ss << __func__ << " working on detid " << in.detId() << " for a digi.size=" << in.size();
#endif
  
  shots_=false;
  for (auto& val :shotApv_) val=false;
  
  cacheDetId=in.detId();

  //Find the position in the DetSet where the first strip of an apv should be inserted
  // needed to deduce if at least stripsForMedian strips per apv have been fired
  for(size_t i=0;i<=maxNumOfApvs;++i){
    
    SiStripDigi d(i*stripsPerApv,0); //Fake digi, at the edge of the apv
    pFirstDigiOfApv[i] = std::lower_bound(in.begin(),in.end(),d);

    //if satisfied it means that the number of digis in the apv i-1 is above stripsForMedia -> apvShot
    if(i>0 && pFirstDigiOfApv[i]-pFirstDigiOfApv[i-1]>stripsForMedian){
      shots_=true;
      shotApv_[i-1]=true;
#ifdef DEBUGME 
      ss << " found an apv shot of " << pFirstDigiOfApv[i]-pFirstDigiOfApv[i-1] << " digis in detid " << in.detId() << " apv " << i << std::endl;
#endif
    }
    
    //---------------------
    //Just for debug REMOVE
    /*
    if(i>0){
      ss << "detid " << in.detId() << " apv " << i-1 << " number digis " << pFirstDigiOfApv[i]-pFirstDigiOfApv[i-1] << " \t shot " << shotApv_[i-1] << std::endl;
      if(pFirstDigiOfApv[i]-pFirstDigiOfApv[i-1]>stripsForMedian-2){
	edm::DetSet<SiStripDigi>::const_iterator dig=pFirstDigiOfApv[i-1];
	while(dig!=pFirstDigiOfApv[i]){
	  ss << "\t strip " << dig->strip() << " dig.adc " << dig->adc();
	  dig++;
	}
	ss << std::endl;
      }
    }
    */
    //-------------------------------
  }
  
#ifdef DEBUGME 
  edm::LogInfo("ApvShot") << ss.str();
#endif

  if(!shots_)
    return false;

  dumpInVector(pFirstDigiOfApv,maxNumOfApvs);
  return true;
}

void SiStripApvShotCleaner::
dumpInVector(edm::DetSet<SiStripDigi>::const_iterator* pFirstDigiOfApv,size_t maxNumOfApvs){ 
  vdigis.clear();
  //loop on Apvs and remove shots. if an apv doesn't have shots, copy it
  for(size_t i=0;i<maxNumOfApvs;++i){
    apvDigis.clear();

    if(shotApv_[i]){
      apvDigis.insert(apvDigis.end(),pFirstDigiOfApv[i],pFirstDigiOfApv[i+1]);
      subtractCM();
      std::stable_sort(apvDigis.begin(),apvDigis.end());
      vdigis.insert(vdigis.end(),apvDigis.begin(),apvDigis.end());
    }else{
      vdigis.insert(vdigis.end(),pFirstDigiOfApv[i],pFirstDigiOfApv[i+1]);
    }
  }
  
#ifdef DEBUGME 
  std::stringstream ss;
  ss <<"detid " << cacheDetId << " new digi.size " << vdigis.size() << "\n";
  for(size_t i=0;i<vdigis.size();++i)
    ss << "\t " << i << " strip " << vdigis[i].strip() << " adc " << vdigis[i].adc() ;
  edm::LogInfo("ApvShot") << ss.str() << std::endl;
#endif
}

void SiStripApvShotCleaner::subtractCM(){

  //order by charge
  std::stable_sort(apvDigis.begin(),apvDigis.end(),
	    [](SiStripDigi const& a, SiStripDigi const& b) {return a.adc() > b.adc();});

  //ignore case where 64th strip is 0ADC
  if(apvDigis[stripsForMedian].adc()==0){
#ifdef DEBUGME 
        std::stringstream ss;
	ss << "case with strip64=0 --> detid= "<<cacheDetId<< "\n";
        edm::LogInfo("ApvShot") << ss.str();
#endif
     return;
  }

  //Find the Median
  float CM = 0.5f*(apvDigis[stripsForMedian].adc()+apvDigis[stripsForMedian-1].adc());
  
  
  if(CM<=0) 
    return;

  //Subtract the median
  const bool is10bit = apvDigis[0].adc() > 255; // approximation; definitely 10bit in this case
  size_t i=0;
  for(;i<stripsForMedian&&apvDigis[i].adc()>CM;++i){
    const uint16_t adc = ( ( apvDigis[i].adc() > 253 ) && !is10bit ) ? apvDigis[i].adc() : (uint16_t)(apvDigis[i].adc()-CM);
    apvDigis[i]=SiStripDigi(apvDigis[i].strip(),adc);
  }
  apvDigis.resize(i);

#ifdef DEBUGME 
  std::stringstream ss;
  ss << "[subtractCM] detid " << cacheDetId << "  CM is " << CM << " the remaining strips after CM subtraction are  " << i;
  edm::LogInfo("ApvShot") << ss.str();
#endif
}


void SiStripApvShotCleaner::
reset(edm::DetSet<SiStripDigi>::const_iterator& a, edm::DetSet<SiStripDigi>::const_iterator& b){
  pDetSet.reset(new edm::DetSet<SiStripDigi>(cacheDetId));
  pDetSet->data.swap(vdigis);
  a=pDetSet->begin();
  b=pDetSet->end();
}
