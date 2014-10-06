#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <cmath>
#include <numeric>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ThreeThresholdAlgorithm::
ThreeThresholdAlgorithm(float chan, float seed, float cluster, unsigned holes, unsigned bad, unsigned adj, std::string qL, 
			bool setDetId, bool removeApvShots) 
  : ChannelThreshold( chan ), SeedThreshold( seed ), ClusterThresholdSquared( cluster*cluster ),
    MaxSequentialHoles( holes ), MaxSequentialBad( bad ), MaxAdjacentBad( adj ), RemoveApvShots(removeApvShots) {
  _setDetId=setDetId;
  qualityLabel = (qL);
  ADCs.reserve(128);

  // this has to be initialized before the first call to candidateEnded()
  // and this is probably a bad place to initialize it
  lastStrip = 0;
}

template<class digiDetSet>
inline
void ThreeThresholdAlgorithm::
clusterizeDetUnit_(const digiDetSet& digis, output_t::FastFiller& output) {
  if(isModuleBad(digis.detId())) return;
  if (!setDetId( digis.detId() )) return;

#ifdef EDM_ML_DEBUG
  if(!isModuleUsable(digis.detId() )) 
    LogWarning("ThreeThresholdAlgorithm") << " id " << digis.detId() << " not usable???" << std::endl;
#endif

  
  typename digiDetSet::const_iterator  
    scan( digis.begin() ), 
    end(  digis.end() );

  if(RemoveApvShots){
    ApvCleaner.clean(digis,scan,end);
  }

  clearCandidate();
  while( scan != end ) {
    while( scan != end  && !candidateEnded( scan->strip() ) ) 
      addToCandidate(*scan++);
    endCandidate(output);
  }
}

inline 
bool ThreeThresholdAlgorithm::
candidateEnded(const uint16_t& testStrip) const {
  uint16_t holes = testStrip - lastStrip - 1;
  return ( ( (!ADCs.empty())  &                    // a candidate exists, and
	     (holes > MaxSequentialHoles )       // too many holes if not all are bad strips, and
	     ) && 
	   ( holes > MaxSequentialBad ||       // (too many bad strips anyway, or 
	     !allBadBetween( lastStrip, testStrip ) // not all holes are bad strips)
	     )
	   );
}

inline 
void ThreeThresholdAlgorithm::
addToCandidate(uint16_t strip, uint8_t adc) { 
  float Noise = noise( strip );
  if(  adc < static_cast<uint8_t>( Noise * ChannelThreshold) || bad(strip) )
    return;

  if(candidateLacksSeed) candidateLacksSeed  =  adc < static_cast<uint8_t>( Noise * SeedThreshold);
  if(ADCs.empty()) lastStrip = strip - 1; // begin candidate
  while( ++lastStrip < strip ) ADCs.push_back(0); // pad holes

  ADCs.push_back( adc );
  noiseSquared += Noise*Noise;
}

template <class T>
inline
void ThreeThresholdAlgorithm::
endCandidate(T& out) {
  if(candidateAccepted()) {
    applyGains();
    appendBadNeighbors();
    auto siz = std::min(ADCs.size(),(size_t)(SiStripCluster::MAX_SIZE));
    out.push_back(SiStripCluster(firstStrip(), ADCs.begin(), ADCs.begin()+siz));
    splitCluster(out);
  }
  clearCandidate();  
}

inline 
bool ThreeThresholdAlgorithm::
candidateAccepted() const {
  return ( !candidateLacksSeed &&
	   noiseSquared * ClusterThresholdSquared
	   <=  std::pow( float(std::accumulate(ADCs.begin(),ADCs.end(), int(0))), 2.f));
}

inline
void ThreeThresholdAlgorithm::
applyGains() {
  uint16_t strip = firstStrip();
  for( auto &  adc :  ADCs) {
#ifdef EDM_ML_DEBUG
    if(adc > 255) throw InvalidChargeException( SiStripDigi(strip,adc) );
#endif
    // if(adc > 253) continue; //saturated, do not scale
    auto charge = int( float(adc)/gain(strip++) + 0.5f ); //adding 0.5 turns truncation into rounding
    if(adc < 254) adc = ( charge > 1022 ? 255 : 
			  ( charge >  253 ? 254 : charge ));
  }
}

inline 
void ThreeThresholdAlgorithm::
appendBadNeighbors() {
  uint8_t max = MaxAdjacentBad;
  while(0 < max--) {
    if( bad( firstStrip()-1) ) { ADCs.insert( ADCs.begin(), 0);  }
    if( bad(  lastStrip + 1) ) { ADCs.push_back(0); lastStrip++; }
  }
}


void ThreeThresholdAlgorithm::clusterizeDetUnit(const    edm::DetSet<SiStripDigi>& digis, output_t::FastFiller& output) {clusterizeDetUnit_(digis,output);}
void ThreeThresholdAlgorithm::clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>& digis, output_t::FastFiller& output) {clusterizeDetUnit_(digis,output);}

inline
bool ThreeThresholdAlgorithm::
stripByStripBegin(uint32_t id) {
  if (!setDetId( id )) return false;
#ifdef EDM_ML_DEBUG
  assert(isModuleUsable( id ));
#endif
  clearCandidate();
  return true;
}

inline
void ThreeThresholdAlgorithm::
stripByStripAdd(uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) {
  if(candidateEnded(strip)) endCandidate(out);
  addToCandidate(SiStripDigi(strip,adc));
}

inline
void ThreeThresholdAlgorithm::
stripByStripEnd(std::vector<SiStripCluster>& out) { 
  endCandidate(out);
}

#include<atomic>
#include<cstdio>
namespace {
  struct Stat {
    Stat() : tot(0),n4(0),lcharge(0),large(0),second(0),split(0){}
    std::atomic<unsigned int> tot;
    std::atomic<unsigned int> n4;
    std::atomic<unsigned int> lcharge;
    std::atomic<unsigned int> large;
    std::atomic<unsigned int> second;
    std::atomic<unsigned int> split;
    ~Stat() { printf("StripClusters: %d/%d/%d/%d/%d/%d\n",tot.load(),n4.load(),lcharge.load(),large.load(),second.load(),split.load());
              printf("StripClusters: %f/%f\n",double(n4.load())/double(tot.load()), double(lcharge.load())/double(tot.load()) );
            } 

  };

  Stat stat;
}


template<class T> 
void ThreeThresholdAlgorithm::splitCluster(T& out) const {
  const auto & cluster = out.back();
  const auto & strips = cluster.amplitudes();
  
  ++stat.tot;

  int charge = std::accumulate(strips.begin(),strips.end(),0);
  if (charge< 60)  ++stat.lcharge;

  if (strips.size()<15) return;  ++stat.n4;
 
  if (charge<120) return; // ++stat.lcharge;

  auto b = &strips.front(); auto e = b+strips.size();
  auto mx = strips.front(); float mean=0; auto lmx=b; auto p = b+1;
  for (;p!=e;++p) {
    if ( (*p)>mx) { mx=(*p); lmx=p;}
    mean+=(*p)*(p-b);
  }
  mean /=float(charge);

  
  if ( std::abs(float(lmx-b)-mean) <1. ) return; ++stat.large;

  // std::cout << "max,mean,size " << lmx-b << ' ' << mean << ' ' << e-b <<  std::endl;

  auto incr = 1; auto le = e;
  if ( float(lmx-b)-mean > 0 ) { le = b-1; incr=-1;}
  p=lmx+incr;
  auto m = *p;  p+=incr; 
  
  auto d = incr>0 ? le-p : p-le; 
  //std::cout << "start end" << p-b << ' ' << le-b << ' ' << d << std::endl;
  if (d<=0) return;

  for (; p!=le; p+=incr) {
     if ( (*p)<m) { m=(*p); continue;}
     break;
  }

  auto fs = cluster.firstStrip();
  for (; p!=le; p+=incr) {
    if ( (*p) > static_cast<uint8_t>( gain(fs+(p-b))*noise(fs+(p-b))*SeedThreshold) ) break;
  }
  if (p==le) return;
  ++stat.second;
  auto nmx = p;
  
  p=lmx+incr;
  m = *p;  p+=incr; auto lm=p;
  for (; p!=nmx; p+=incr)
     if ( (*p)<m) { m=(*p); lm=p;}

  float charge1 = std::accumulate(b,lm,0); charge1 += (*lm)*charge1/charge;
  float charge2 = charge=charge1;
  if (std::min(charge1,charge2)< 60) return;

  ++stat.split;

  auto newst = strips;
  auto in = lm-b;
  newst[in] = charge1;
  out.back() =  SiStripCluster(fs,newst.begin(),newst.begin()+in+1);
  newst[in] = charge2;
  out.push_back(SiStripCluster(fs+in,newst.begin()+in, newst.end()));


}

