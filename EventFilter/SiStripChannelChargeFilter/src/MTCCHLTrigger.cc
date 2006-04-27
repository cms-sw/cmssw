#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"


namespace cms
{
MTCCHLTrigger::MTCCHLTrigger(const edm::ParameterSet& ps){

   selOnClusterCharge=ps.getParameter<bool>("SelOnClusterCharge");
   selOnDigiCharge=ps.getParameter<bool>("SelOnDigiCharge");
   clusterChargeThreshold=ps.getParameter<int>("ClusterChargeThreshold");
   digiChargeThreshold=ps.getParameter<int>("DigiChargeThreshold");
}

bool MTCCHLTrigger::filter(edm::Event const& e, edm::EventSetup const& c) {
 

  //get data
  //StripCluster
  edm::Handle<SiStripClusterCollection> h;
  e.getByType(h);
  //StripDigi
  edm::Handle<StripDigiCollection> di;
  e.getByType(di);

  std::vector<unsigned int> DigiIds    = (*di).detIDs();
  std::vector<unsigned int> ClusterIds = (*h).detIDs();

  unsigned int ndigis=0;
  unsigned int nclust=0;
  unsigned int digiadc=0;
  std::vector<unsigned int>::const_iterator detid;
  for (detid=DigiIds.begin();detid!=DigiIds.end();detid++){
    const StripDigiCollection::Range digiRange = (*di).get(*detid);
    StripDigiCollection::ContainerIterator digiRangeIteratorBegin = digiRange.first;
    StripDigiCollection::ContainerIterator digiRangeIteratorEnd   = digiRange.second;
    ndigis+=(digiRangeIteratorEnd-digiRangeIteratorBegin);
    StripDigiCollection::ContainerIterator digiiter;
    for (digiiter=digiRangeIteratorBegin;digiiter!=digiRangeIteratorEnd;digiiter++){
      digiadc+=(*digiiter).adc();
    }
  }
  
  unsigned int amplclus=0;
  for (detid=ClusterIds.begin();detid!=ClusterIds.end();detid++){
    const SiStripClusterCollection::Range clusterRange = (*h).get(*detid);
    SiStripClusterCollection::ContainerIterator clusRangeIteratorBegin = clusterRange.first;
    SiStripClusterCollection::ContainerIterator clusRangeIteratorEnd   = clusterRange.second;


    nclust+=(clusRangeIteratorEnd-clusRangeIteratorBegin);
    SiStripClusterCollection::ContainerIterator clustiter;
    for (clustiter=clusRangeIteratorBegin;clustiter!=clusRangeIteratorEnd;clustiter++){

      std::vector<short>::const_iterator ish;
      std::vector<short>::const_iterator ishb=(*clustiter).amplitudes().begin();
      std::vector<short>::const_iterator ishe=(*clustiter).amplitudes().end();
      for (ish=ishb;ish!=ishe;ish++){

	amplclus+=(*ish);
      }
    }
  }
 
 

  bool allowed=false;
  if ((selOnClusterCharge)&&(amplclus>clusterChargeThreshold))  allowed=true;
  if ((selOnDigiCharge)&&(amplclus>digiChargeThreshold))  allowed=true;

  return allowed;
}
 
}
