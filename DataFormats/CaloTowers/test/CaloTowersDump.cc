#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>

using namespace std;


  /** \class CaloTowersDump
      
  $Date: 2005/10/17 20:25:48 $
  $Revision: 1.1 $
  \author J. Mans - Minnesota
  */
  class CaloTowersDump : public edm::EDAnalyzer {
  public:
    explicit CaloTowersDump(edm::ParameterSet const& conf);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  };


  CaloTowersDump::CaloTowersDump(edm::ParameterSet const& conf) {
  }
  
  void CaloTowersDump::analyze(edm::Event const& e, edm::EventSetup const& c) {
    std::vector<edm::Handle<CaloTowerCollection> > prods;
    
    try {
      e.getManyByType(prods);
      //      cout << "Selected " << hbhe.size() << endl;
      std::vector<edm::Handle<CaloTowerCollection> >::iterator i;
      for (i=prods.begin(); i!=prods.end(); i++) {
	const CaloTowerCollection& c=*(*i);
	
	for (CaloTowerCollection::const_iterator j=c.begin(); j!=c.end(); j++) {
	  cout << *j << std::endl;
	}
      }
    } catch (...) {
      cout << "No CaloTowers." << endl;
    }
  }


#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CaloTowersDump);

