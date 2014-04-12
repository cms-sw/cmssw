#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace cms
{
MTCCHLTrigger::MTCCHLTrigger(const edm::ParameterSet& ps){
   selOnDigiCharge=ps.getParameter<bool>("SelOnDigiCharge");
   ChargeThreshold=ps.getParameter<int>("ChargeThreshold");
   clusterProducer = ps.getParameter<std::string>("ClusterProducer");
   produces <int>();
   produces <unsigned int>();
}

bool MTCCHLTrigger::filter(edm::Event & e, edm::EventSetup const& c) {
  //get data
  //StripCluster
  edm::Handle< edm::DetSetVector<SiStripCluster> > h;
  e.getByLabel(clusterProducer,h);

  //StripDigi from RawToDigi and ZeroSuppressor
  std::vector< edm::Handle< edm::DetSetVector<SiStripDigi> > > di;
  e.getManyByType(di);

  if (selOnDigiCharge) {
    unsigned int digiadc=0;
    for (std::vector< edm::Handle< edm::DetSetVector<SiStripDigi> > >::const_iterator mi = di.begin(); mi!=di.end(); mi++){
      for (edm::DetSetVector<SiStripDigi>::const_iterator it = (*mi)->begin(); it!= (*mi)->end();it++) {
	for(std::vector<SiStripDigi>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++) digiadc += vit->adc();
      }
    }
    return (digiadc>ChargeThreshold) ? true : false;
  } else {
    unsigned int amplclus=0;
    for (edm::DetSetVector<SiStripCluster>::const_iterator it=h->begin();it!=h->end();it++) {
      for(std::vector<SiStripCluster>::const_iterator vit=(it->data).begin(); vit!=(it->data).end(); vit++){
	for(std::vector<uint8_t>::const_iterator ia=vit->amplitudes().begin(); ia!=vit->amplitudes().end(); ia++) 
        {
            if  ((*ia)>0){ amplclus+=(*ia); }
        }
      }
    }
    bool decision= (amplclus>ChargeThreshold) ? true : false;
    std::auto_ptr< unsigned int > output( new unsigned int(amplclus) );
    std::auto_ptr< int > output_dec( new int(decision) );
    e.put(output);
    e.put(output_dec);
    return decision;
  }
 }
}
