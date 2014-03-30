#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include <string>
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalRecHitDump
      
  \author J. Mans - Minnesota
  */
  class HcalRecHitDump : public edm::EDAnalyzer
  {
  public:
    explicit HcalRecHitDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);

  private:
    edm::GetterOfProducts<HcalSourcePositionData> getHcalSourcePositionData_;

    string hbhePrefix_;
    string hoPrefix_;
    string hfPrefix_;
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) :
    getHcalSourcePositionData_(edm::ProcessMatch("*"), this),
    hbhePrefix_(conf.getUntrackedParameter<string>("hbhePrefix", "")),
    hoPrefix_(conf.getUntrackedParameter<string>("hoPrefix", "")),
    hfPrefix_(conf.getUntrackedParameter<string>("hfPrefix", ""))
  {
    callWhenNewProductsRegistered(getHcalSourcePositionData_);
  }

  template<typename COLL>
  static void analyzeT(edm::Event const& e, const char* name=0, const char* prefix=0)
  {
    const string marker(prefix ? prefix : "");
    try {
      vector<edm::Handle<COLL> > colls;
      e.getManyByType(colls);
      typename std::vector<edm::Handle<COLL> >::iterator i;
      for (i=colls.begin(); i!=colls.end(); i++) {
        for (typename COLL::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++)
          cout << marker << *j << endl;
      }
    } catch (...) {
      if(name) cout << "No " << name << " RecHits." << endl;
    }
  }

  void HcalRecHitDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    analyzeT<HBHERecHitCollection>(e, "HB/HE", hbhePrefix_.c_str()); 
    analyzeT<HFRecHitCollection>(e, "HF", hfPrefix_.c_str());
    analyzeT<HORecHitCollection>(e, "HO", hoPrefix_.c_str());
    analyzeT<HcalCalibRecHitCollection>(e);
    analyzeT<ZDCRecHitCollection>(e);
    analyzeT<CastorRecHitCollection>(e);

    std::vector<edm::Handle<HcalSourcePositionData> > handles;
    getHcalSourcePositionData_.fillHandles(e, handles);
    for (auto const& spd : handles){
      cout << *spd << endl;
    }
    cout << endl;    
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;


DEFINE_FWK_MODULE(HcalRecHitDump);

