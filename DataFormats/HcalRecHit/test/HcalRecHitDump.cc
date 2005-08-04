#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>

using namespace std;

namespace cms {

  /** \class HcalRecHitDump
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class HcalRecHitDump : public edm::EDAnalyzer {
  public:
    explicit HcalRecHitDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };



 class HRHHappySelector : public edm::Selector {
  public:
    HRHHappySelector() { }
  private:
    virtual bool doMatch(const edm::Provenance& p) const {
      //      cout << p << endl;
      return true;
    }
 };


  HcalRecHitDump::HcalRecHitDump(edm::ParameterSet const& conf) {
  }
  
  void HcalRecHitDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    std::vector<edm::Handle<HBHERecHitCollection> > hbhe;
    std::vector<edm::Handle<HORecHitCollection> > ho;
    std::vector<edm::Handle<HFRecHitCollection> > hf;
    
    HRHHappySelector s;
    
    try {
      e.getMany(s,hbhe);
      //      cout << "Selected " << hbhe.size() << endl;
      std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
      for (i=hbhe.begin(); i!=hbhe.end(); i++) {
	const HBHERecHitCollection& c=*(*i);
	
	for (unsigned int j=0; j<c.size(); j++)
	  cout << c[j] << std::endl;
      }
    } catch (...) {
      cout << "No HB/HE RecHits." << endl;
    }
    
    try {
      e.getMany(s,hf);
      //      cout << "Selected " << hf.size() << endl;
      std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
      for (i=hf.begin(); i!=hf.end(); i++) {
	const HFRecHitCollection& c=*(*i);
	
	for (unsigned int j=0; j<c.size(); j++)
	  cout << c[j] << std::endl;
      }
    } catch (...) {
      cout << "No HF RecHits." << endl;
    }
    
    try {
      e.getMany(s,ho);
      //      cout << "Selected " << ho.size() << endl;
      std::vector<edm::Handle<HORecHitCollection> >::iterator i;
      for (i=ho.begin(); i!=ho.end(); i++) {
	const HORecHitCollection& c=*(*i);
	
	for (unsigned int j=0; j<c.size(); j++)
	  cout << c[j] << std::endl;
      }
    } catch (...) {
      cout << "No HO RecHits." << endl;
    }

    cout << endl;    
  }
}

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace cms;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalRecHitDump)

