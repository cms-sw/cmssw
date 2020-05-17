#include "EventFilter/SiStripChannelChargeFilter/interface/MTCCHLTrigger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace cms {
  MTCCHLTrigger::MTCCHLTrigger(const edm::ParameterSet& ps) {
    selOnDigiCharge = ps.getParameter<bool>("SelOnDigiCharge");
    ChargeThreshold = ps.getParameter<int>("ChargeThreshold");
    clusterProducer = ps.getParameter<std::string>("ClusterProducer");
    produces<int>();
    produces<unsigned int>();
  }

  bool MTCCHLTrigger::filter(edm::Event& e, edm::EventSetup const& c) {
    //get data
    //StripCluster
    edm::Handle<edm::DetSetVector<SiStripCluster> > h;
    e.getByLabel(clusterProducer, h);

    //StripDigi from RawToDigi and ZeroSuppressor
    std::vector<edm::Handle<edm::DetSetVector<SiStripDigi> > > di;
    e.getManyByType(di);

    if (selOnDigiCharge) {
      unsigned int digiadc = 0;
      for (const auto& mi : di) {
        for (auto it = mi->begin(); it != mi->end(); it++) {
          for (auto vit : (it->data))
            digiadc += vit.adc();
        }
      }
      return (digiadc > ChargeThreshold) ? true : false;
    } else {
      unsigned int amplclus = 0;
      for (const auto& it : *h) {
        for (auto vit = (it.data).begin(); vit != (it.data).end(); vit++) {
          for (unsigned char ia : vit->amplitudes()) {
            if (ia > 0) {
              amplclus += ia;
            }
          }
        }
      }
      bool decision = (amplclus > ChargeThreshold) ? true : false;
      e.put(std::make_unique<unsigned int>(amplclus));
      e.put(std::make_unique<int>(decision));
      return decision;
    }
  }
}  // namespace cms
