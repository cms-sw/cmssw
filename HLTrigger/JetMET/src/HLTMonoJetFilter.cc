/** \class HLTMonoJetFilter
*
*
*  \author Srimanobhas Phat
*
*/

#include "HLTrigger/JetMET/interface/HLTMonoJetFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
//#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>


//
// constructors and destructor
//
HLTMonoJetFilter::HLTMonoJetFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  //Input tag
  inputJetTag_           = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  //PFJets
  max_PtSecondJet_       = iConfig.getParameter<double> ("max_PtSecondJet");
  max_DeltaPhi_          = iConfig.getParameter<double> ("max_DeltaPhi");
}

HLTMonoJetFilter::~HLTMonoJetFilter(){}

void 
HLTMonoJetFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltAntiKT5ConvPFJets"));
  desc.add<bool>("saveTags",false);
  desc.add<double>("max_PtSecondJet",9999.);
  desc.add<double>("max_DeltaPhi",99.);
  descriptions.add("hltMonoJetFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTMonoJetFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  // Get the Candidates
  // CaloJets
  edm::Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 0){ 
    int countJet(0);
    CaloJetRef JetRef1;
    //CaloJetRef JetRef2;
    double calo1Phi=0.;
    double calo2Pt=0.;
    double calo2Phi=0.;
  
    for(CaloJetCollection::const_iterator i = recocalojets->begin();
	i != recocalojets->end(); i++) {
      if(countJet==0){
	JetRef1 = CaloJetRef(recocalojets,distance(recocalojets->begin(),i));
	calo1Phi  = i->phi();
      }
      if(countJet==1){
	//JetRef2 = CaloJetRef(recocalojets,distance(recocalojets->begin(),i)); 
	calo2Pt   = i->pt();
	calo2Phi  = i->phi();
      }
      countJet++;
      if(countJet>=2) break;
    }
  
    if(countJet==1){
      n=1;
      //JetRef2=JetRef1;
    }
    else if(countJet>1 && calo2Pt<max_PtSecondJet_){
      n=1;
    }
    else if(countJet>1 && calo2Pt>=max_PtSecondJet_){
      double Dphi=fabs(deltaPhi(calo1Phi,calo2Phi));
      if(Dphi>=max_DeltaPhi_) n=-1;
      else n=1;
    }
    else{
      n=-1;
    }
  
    if(n==1){
      filterproduct.addObject(TriggerJet,JetRef1);
      //filterproduct.addObject(TriggerJet,JetRef2);
    }
  }

  bool accept(n==1); 

  return accept;
}
