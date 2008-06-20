#include "ElectroWeakAnalysis/ZReco/plugins/FiducialRegion.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

#include <iostream>
#include <algorithm>

using namespace reco;
using namespace std;

FiducialRegion::FiducialRegion(const edm::ParameterSet& conf)
{
  src_ = conf.getParameter<edm::InputTag>("src");
}  
  
FiducialRegion::~FiducialRegion()
{
}

void FiducialRegion::select(edm::Handle<reco::PixelMatchGsfElectronCollection> c, const edm::Event& e, const edm::EventSetup& s)
{
  selected_.clear();

  for( reco::PixelMatchGsfElectronCollection::const_iterator el = c->begin();  el != c->end(); el++ ) {
    
    //cutting on detector eta
    const reco::SuperCluster& sc = *(el->superCluster()) ;
    //cout << "eta=" << sc.eta() << " " << el->eta() << endl;

    //if (fabs(el->eta()) < 1.4442 || 1.560 < fabs(el->eta()) ) {
    if (fabs(sc.eta()) < 1.4442 || 1.560 < fabs(sc.eta()) ) {
      selected_.push_back( &(*el) );
    }
  }  

}

