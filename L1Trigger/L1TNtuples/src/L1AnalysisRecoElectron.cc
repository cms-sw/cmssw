#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoElectron.h"

using namespace std;

L1Analysis::L1AnalysisRecoElectron::L1AnalysisRecoElectron()
{
}


L1Analysis::L1AnalysisRecoElectron::~L1AnalysisRecoElectron()
{
}

void L1Analysis::L1AnalysisRecoElectron::SetElectron(const edm::Event& event,
						     const edm::EventSetup& setup,
						     //const edm::Handle<edm::View<reco::GsfElectron>>& electrons,
						     edm::Handle<reco::GsfElectronCollection> electrons,
						     std::vector<edm::Handle<edm::ValueMap<bool> > > eleVIDDecisionHandles,
						     const unsigned& maxElectron)

{

  recoElectron_.nElectrons=0;

  for(reco::GsfElectronCollection::const_iterator el=electrons->begin();
      el!=electrons->end() && recoElectron_.nElectrons < maxElectron;
      ++el) {

    recoElectron_.e.push_back(el->energy());    
    recoElectron_.pt.push_back(el->pt());    
    recoElectron_.et.push_back(el->et());    
    recoElectron_.eta.push_back(el->eta());
    recoElectron_.phi.push_back(el->phi());
    recoElectron_.eta_SC.push_back((el->superClusterPosition()).eta());
    recoElectron_.phi_SC.push_back((el->superClusterPosition()).phi());
    recoElectron_.e_ECAL.push_back(el->ecalEnergy());
    recoElectron_.e_SC.push_back(el->superCluster()->energy());
    recoElectron_.charge.push_back(el->charge());

    edm::Ref<reco::GsfElectronCollection> electronEdmRef(electrons,recoElectron_.nElectrons);
    
    recoElectron_.isVetoElectron.push_back( (*(eleVIDDecisionHandles[0]))[electronEdmRef] );
    recoElectron_.isLooseElectron.push_back( (*(eleVIDDecisionHandles[1]))[electronEdmRef] );
    recoElectron_.isMediumElectron.push_back( (*(eleVIDDecisionHandles[2]))[electronEdmRef] );
    recoElectron_.isTightElectron.push_back( (*(eleVIDDecisionHandles[3]))[electronEdmRef] );

    double iso = (el->pfIsolationVariables().sumChargedHadronPt + max(
	   el->pfIsolationVariables().sumNeutralHadronEt +
           el->pfIsolationVariables().sumPhotonEt - 
           0.5 * el->pfIsolationVariables().sumPUPt, 0.0)) / el->pt();

    recoElectron_.iso.push_back(iso);
    // cout<<ConversionTools::hasMatchedConversion(*el,conversions,theBeamSpot->position())<<endl;
    
    recoElectron_.nElectrons++;

  }
  // for(reco::GsfElectronCollection::const_iterator it=electrons->begin();
  //     it!=electrons->end() && recoElectron_.nElectrons < maxElectron;
  //     ++it) {

  //   recoElectron_.e.push_back(it->energy());    
  //   recoElectron_.pt.push_back(it->pt());    
  //   recoElectron_.et.push_back(it->et());    
  //   recoElectron_.eta.push_back(it->eta());
  //   recoElectron_.phi.push_back(it->phi());

  //   // cout<<it->superCluster().position().eta()<<endl;
  //   isVetoElectronCustom(*it, vertices, conversions, theBeamSpot, Rho);

  //   // cout<<ConversionTools::hasMatchedConversion(*it,conversions,theBeamSpot->position())<<endl;

  //   recoElectron_.nElectrons++;

  // }
}


