#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <iostream>

using namespace std;


  /** \class CaloTowersDump
      
  $Date: 2010/02/25 00:29:57 $
  $Revision: 1.6 $
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


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_FWK_MODULE(CaloTowersDump);

