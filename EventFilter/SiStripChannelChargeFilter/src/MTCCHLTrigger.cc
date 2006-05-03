#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
//#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
//#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace cms
{
MTCCHLTrigger::MTCCHLTrigger(const edm::ParameterSet& ps){

  //   selOnClusterCharge=ps.getParameter<bool>("SelOnClusterCharge");
   selOnDigiCharge=ps.getParameter<bool>("SelOnDigiCharge");
   ChargeThreshold=ps.getParameter<int>("ChargeThreshold");
   //   digiChargeThreshold=ps.getParameter<int>("DigiChargeThreshold");
// retrieve producer name of input StripDigiCollection
   std::string rawtodigiProducer = ps.getParameter<std::string>("RawToDigiProducer");
   std::string zsdigiProducer = ps.getParameter<std::string>("ZSDigiProducer");
   std::string clusterProducer = ps.getParameter<std::string>("ClusterProducer");

}


bool MTCCHLTrigger::filter(edm::Event & e, edm::EventSetup const& c) {

 
  //  bool allowed=false;

  //get data
  //StripCluster
  edm::Handle< edm::DetSetVector<SiStripCluster> > h;
  e.getByLabel(clusterProducer,"stripcluster",h);
  //StripDigi from RawToDigi and ZeroSuppressor
  std::vector< edm::Handle< edm::DetSetVector<SiStripDigi> > > di;
  e.getManyByType(di);
  //SiStripDigi from ZeroSuppressor
  //   edm::Handle< edm::DetSetVector<SiStripDigi> > diZS;
  //   e.getByLabel(zsdigiProducer,"zsdigi",diZS);


  //  std::vector<unsigned int> DigiIds    = (*di).detIDs();
  //  std::vector<unsigned int> ClusterIds = (*h).detIDs();

  //  unsigned int ndigis=0;
  //  unsigned int nclust=0;


  if (selOnDigiCharge) {

    unsigned int digiadc=0;

    for (std::vector< edm::Handle< edm::DetSetVector<SiStripDigi> > >::const_iterator mi = di.begin(); mi!=di.end(); mi++){

      for (edm::DetSetVector<SiStripDigi>::const_iterator it = (*mi)->begin(); it!= (*mi)->end();it++) {

	for(std::vector<SiStripDigi>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++) digiadc += vit->adc();
	
      }

    }
  
    //     //ZS Digis from ZeroSuppressor
//     for (edm::DetSetVector<SiStripDigi>::const_iterator it=diZS->begin();it!=diZS->end();it++) {

//       for(std::vector<SiStripDigi>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++) digiadc += vit->adc();
      
//     }
  

    return (digiadc>ChargeThreshold) ? true : false;

  }

  else {

    //   std::vector<unsigned int>::const_iterator detid;
    //   for (detid=DigiIds.begin();detid!=DigiIds.end();detid++){
    //     const StripDigiCollection::Range digiRange = (*di).get(*detid);
    //     StripDigiCollection::ContainerIterator digiRangeIteratorBegin = digiRange.first;
    //     StripDigiCollection::ContainerIterator digiRangeIteratorEnd   = digiRange.second;
    //     ndigis+=(digiRangeIteratorEnd-digiRangeIteratorBegin);
    //     StripDigiCollection::ContainerIterator digiiter;
    //     for (digiiter=digiRangeIteratorBegin;digiiter!=digiRangeIteratorEnd;digiiter++){
    //       digiadc+=(*digiiter).adc();
    //     }
    //   }
  




    unsigned int amplclus=0;

    for (edm::DetSetVector<SiStripCluster>::const_iterator it=h->begin();it!=h->end();it++) {
      
      //  for (detid=ClusterIds.begin();detid!=ClusterIds.end();detid++){
      //    const SiStripClusterCollection::Range clusterRange = (*h).get(*detid);
      //    SiStripClusterCollection::ContainerIterator clusRangeIteratorBegin = clusterRange.first;
      //    SiStripClusterCollection::ContainerIterator clusRangeIteratorEnd   = clusterRange.second;
      
      //    nclust+=(clusRangeIteratorEnd-clusRangeIteratorBegin);
      //    SiStripClusterCollection::ContainerIterator clustiter;
      //    for (clustiter=clusRangeIteratorBegin;clustiter!=clusRangeIteratorEnd;clustiter++){
      
      //      std::vector<short>::const_iterator ish;
      //  std::vector<short>::const_iterator ishb=(*clustiter).amplitudes().begin();
      //       std::vector<short>::const_iterator ishe=(*clustiter).amplitudes().end();
      //       for (ish=ishb;ish!=ishe;ish++){
      
      // 	amplclus+=(*ish);
      //       }
      
      for(std::vector<SiStripCluster>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++){
	
	for(std::vector<short>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) amplclus+=(*ia);
	
      }
    }
 
    return (amplclus>ChargeThreshold) ? true : false;

  }
 
}

}
