#include "RecoLocalTracker/SiStripClusterizer/test/CompareClusters.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "FWCore/Framework/interface/Event.h"
#include <functional>
#include <numeric>
#include <vector>
#include <iostream>
#include <sstream>

CompareClusters::
CompareClusters(const edm::ParameterSet& conf) 
  : clusters1( conf.getParameter<edm::InputTag>("Clusters1")),
    clusters2( conf.getParameter<edm::InputTag>("Clusters2")),
    digis( conf.getParameter<edm::InputTag>("Digis"))
{}

void CompareClusters::
analyze(const edm::Event& event, const edm::EventSetup& es) {
  event.getByLabel(clusters1, clusterHandle1);  if(!clusterHandle1.isValid()) throw cms::Exception("Input Not found") << clusters1;
  event.getByLabel(clusters2, clusterHandle2);  if(!clusterHandle2.isValid()) throw cms::Exception("Input Not found") << clusters2;
  if( identicalDSV( *clusterHandle1, *clusterHandle2) ) 
    return;

  {//digi access
    event.getByLabel(digis, digiHandle);  if(!digiHandle.isValid()) throw cms::Exception("Input Not found") << digis;
    es.get<SiStripNoisesRcd>().get(noiseHandle);
    es.get<SiStripGainRcd>().get(gainHandle);
    es.get<SiStripQualityRcd>().get(qualityHandle);
  }
  
  input_t::const_iterator 
    set1( clusterHandle1->begin()),  end1( clusterHandle1->end()),
    set2( clusterHandle2->begin()),  end2( clusterHandle2->end());

  message.str("");
  while( set1!=end1 && set2!=end2 ) {
    while( set1!=end1 && set1->id() < set2->id() )  show( (set1++)->id() );
    while( set2!=end2 && set2->id() < set1->id() )  show( (set2++)->id() );
    if( set1!=end1 && set2!=end2 ) {
      if( !identicalDetSet(*set1,*set2) )           show( set1->id() );
      set1++; set2++;
    }
  }
  while( set1!=end1 ) show( (set1++)->id() );
  while( set2!=end2 ) show( (set2++)->id() );
  edm::LogError("Not Identical") << message.str();

  return;
}

void CompareClusters::
show( uint32_t id) {
  message << std::endl << "detId: " << id << std::endl;
  message << "Digis:\n" << printDigis(id);
  edmNew::DetSet<SiStripCluster>::const_iterator c1(0), c2(0), end1(0), end2(0);
  if( clusterHandle1->find(id) != clusterHandle1->end()) { c1 = clusterHandle1->find(id)->begin(); end1 = clusterHandle1->find(id)->end();}
  if( clusterHandle2->find(id) != clusterHandle2->end()) { c2 = clusterHandle2->find(id)->begin(); end2 = clusterHandle2->find(id)->end();}
  message << clusters1.label() << std::endl;   while( c1 != end1 ) message << printCluster(*c1++);
  message << clusters2.label() << std::endl;   while( c2 != end2 ) message << printCluster(*c2++);
}

std::string CompareClusters::
printCluster(const SiStripCluster & cluster) {
  std::stringstream s;
  s  << "\t" << cluster.firstStrip() << " [ ";
  for(unsigned i=0; i<cluster.amplitudes().size(); i++) {
    s << static_cast<int>(cluster.amplitudes()[i]) << " ";
  }
  s << "]" << std::endl;
  return s.str();
}

std::string CompareClusters::
printDigis(uint32_t id) {
  std::stringstream s;
  SiStripApvGain::Range gainRange =  gainHandle->getRange(id);     
  SiStripNoises::Range  noiseRange = noiseHandle->getRange(id);    
  SiStripQuality::Range qualityRange = qualityHandle->getRange(id);
  edm::DetSetVector<SiStripDigi>::const_iterator set = digiHandle->find(id);
  if(set != digiHandle->end()) {
    for(edm::DetSet<SiStripDigi>::const_iterator digi = set->begin(); digi!=set->end(); digi++) {
      s << "( " 
	<< digi->strip() << ", "
	<< digi->adc()   << ", "
	<< noiseHandle->getNoise(digi->strip(),noiseRange) << ", "
	<< gainHandle->getStripGain(digi->strip(),gainRange) << ", "
	<< ( qualityHandle->IsStripBad(qualityRange, digi->strip())  ? "bad" : "good") << ")\n";
    }
  }
  return s.str();
}

bool CompareClusters::
identicalDSV(const input_t& L, const input_t& R) {
  return 
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		  bool(true), std::logical_and<bool>(), identicalDetSet );  
}

bool CompareClusters::
identicalDetSet(const edmNew::DetSet<SiStripCluster>& L, const edmNew::DetSet<SiStripCluster>& R) {
  return 
    L.id() == R.id() &&
    L.size() == R.size() &&
    inner_product(L.begin(), L.end(), R.begin(),
		  bool(true), std::logical_and<bool>(), identicalClusters );
}

bool CompareClusters::
identicalClusters(const SiStripCluster& L, const SiStripCluster& R) {
  return
    L.firstStrip() == R.firstStrip() 
    && L.amplitudes().size() == R.amplitudes().size()
    && inner_product(L.amplitudes().begin(), L.amplitudes().end(), R.amplitudes().begin(), 
		     bool(true), std::logical_and<bool>(), std::equal_to<uint16_t>() );
}
