#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace cms
{
MTCCHLTrigger::MTCCHLTrigger(const edm::ParameterSet& ps){


   selOnDigiCharge=ps.getParameter<bool>("SelOnDigiCharge");
   ChargeThreshold=ps.getParameter<int>("ChargeThreshold");
   clusterProducer = ps.getParameter<std::string>("ClusterProducer");
 
}


bool MTCCHLTrigger::filter(edm::Event & e, edm::EventSetup const& c) {

 
  //  bool allowed=false;

  //get data
  //StripCluster
  edm::Handle< edm::DetSetVector<SiStripCluster> > h;
  e.getByLabel(clusterProducer,h);

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
    
    unsigned int amplclus=0;

    for (edm::DetSetVector<SiStripCluster>::const_iterator it=h->begin();it!=h->end();it++) {
  
      for(std::vector<SiStripCluster>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++){
	
	for(std::vector<short>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) amplclus+=(*ia);
	
      }
    }
 
    return (amplclus>ChargeThreshold) ? true : false;

  }
 
 }

}
