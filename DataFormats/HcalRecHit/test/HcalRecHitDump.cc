#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalRecHitDump
      
  $Date: 2010/02/25 00:30:00 $
  $Revision: 1.11 $
  \author J. Mans - Minnesota
  */
  class HcalRecHitDump : public edm::EDAnalyzer {
  public:
    explicit HcalRecHitDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };

  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) {
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
    analyzeT<HcalUpgradeRecHitCollection>(e, "HcalUpgrade");

    edm::Handle<HcalSourcePositionData> spd;
    try {
      e.getByType(spd);
      cout << *spd << std::endl;
    } catch (...) {
//      cout << "No Source Position Data" << endl;
    }

    cout << endl;    
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;


DEFINE_FWK_MODULE(HcalRecHitDump);

