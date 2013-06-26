#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalRecHitDump
      
  $Date: 2012/09/07 15:47:38 $
  $Revision: 1.13 $
  \author J. Mans - Minnesota
  */
  class HcalRecHitDump : public edm::EDAnalyzer {
  public:
    explicit HcalRecHitDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  private:
    edm::GetterOfProducts<HcalSourcePositionData> getHcalSourcePositionData_;
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) :
    getHcalSourcePositionData_(edm::ProcessMatch("*"), this) {
    callWhenNewProductsRegistered(getHcalSourcePositionData_);
  }
  
  template<typename COLL> void analyzeT(edm::Event const& e, const char * name=0) {
    try {
      std::vector<edm::Handle<COLL> > colls;
      e.getManyByType(colls);
      typename std::vector<edm::Handle<COLL> >::iterator i;
      for (i=colls.begin(); i!=colls.end(); i++) {
        for (typename COLL::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++)
          cout << *j << std::endl;
      }
    } catch (...) {
      if(name) cout << "No " << name << " RecHits." << endl;
    }
  }

  void HcalRecHitDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    analyzeT<HBHERecHitCollection>(e, "HB/HE"); 
    analyzeT<HFRecHitCollection>(e, "HF");
    analyzeT<HORecHitCollection>(e, "HO");
    analyzeT<HcalCalibRecHitCollection>(e);
    analyzeT<ZDCRecHitCollection>(e);
    analyzeT<CastorRecHitCollection>(e);

    std::vector<edm::Handle<HcalSourcePositionData> > handles;
    getHcalSourcePositionData_.fillHandles(e, handles);
    for (auto const& spd : handles){
      cout << *spd << std::endl;
    }
    cout << endl;    
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;


DEFINE_FWK_MODULE(HcalRecHitDump);

