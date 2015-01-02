#include "HLTrigger/JetMET/interface/HLTJetCollForElePlusJets.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "TVector3.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
HLTJetCollForElePlusJets<T>::HLTJetCollForElePlusJets(const edm::ParameterSet& iConfig):
  hltElectronTag(iConfig.getParameter< edm::InputTag > ("HltElectronTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minJetPt_(iConfig.getParameter<double> ("MinJetPt")),
  maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
  minNJets_(iConfig.getParameter<unsigned int> ("MinNJets")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR")),
  //Only for VBF
  minSoftJetPt_(iConfig.getParameter< double > ("MinSoftJetPt")),
  minDeltaEta_(iConfig.getParameter< double > ("MinDeltaEta"))
{
  typedef std::vector<T> TCollection;
  m_theElectronToken = consumes<trigger::TriggerFilterObjectWithRefs>(hltElectronTag);
  m_theJetToken = consumes<TCollection>(sourceJetTag);
  produces<TCollection>();
}


template <typename T>
HLTJetCollForElePlusJets<T>::~HLTJetCollForElePlusJets()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

template <typename T>
void HLTJetCollForElePlusJets<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ("HltElectronTag", edm::InputTag("triggerFilterObjectWithRefs"));
    desc.add<edm::InputTag> ("SourceJetTag", edm::InputTag("jetCollection"));
    desc.add<double> ("MinJetPt", 30.);
    desc.add<double> ("MaxAbsJetEta", 2.6);
    desc.add<unsigned int> ("MinNJets", 1);
    desc.add<double> ("minDeltaR", 0.5);
    //Only for VBF
    desc.add<double> ("MinSoftJetPt", 25.);
    desc.add<double> ("MinDeltaEta", -1.);    
    descriptions.add(defaultModuleLabel<HLTJetCollForElePlusJets<T>>(), desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
template <typename T>
void
HLTJetCollForElePlusJets<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;
  typedef edm::RefVector<TCollection> TRefVector;
  typedef std::vector<edm::RefVector<std::vector<T>,T,edm::refhelper::FindUsingAdvance<std::vector<T>,T> > > TCollectionVector;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(m_theElectronToken,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  // Electrons can be stored as objects of types TriggerCluster, TriggerElectron, or TriggerPhoton
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  trigger::VRphoton photonCands;
  PrevFilterOutput->getObjects(trigger::TriggerPhoton, photonCands);
  
  //prepare the collection of 3-D vector for electron momenta
  std::vector<TVector3> ElePs;

  if(!clusCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      TVector3 positionVector(
			      clusCands[candNr]->superCluster()->position().x(),
			      clusCands[candNr]->superCluster()->position().y(),
			      clusCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }else if(!eleCands.empty()){ // try trigger electrons
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      TVector3 positionVector(
			      eleCands[candNr]->superCluster()->position().x(),
			      eleCands[candNr]->superCluster()->position().y(),
			      eleCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }
  else if(!photonCands.empty()){ // try trigger photons
    for(size_t candNr=0;candNr<photonCands.size();candNr++){
      TVector3 positionVector(
                  photonCands[candNr]->superCluster()->position().x(),
                  photonCands[candNr]->superCluster()->position().y(),
                  photonCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }
  
  edm::Handle<TCollection> theJetCollectionHandle;
  iEvent.getByToken(m_theJetToken, theJetCollectionHandle);
 
  const TCollection & theJetCollection = *theJetCollectionHandle;
  
  std::auto_ptr< TCollection >  theFilteredJetCollection(new TCollection);
  
  std::auto_ptr < TCollectionVector > allSelections(new TCollectionVector());
  
  bool foundSolution(false);

  for (unsigned int i = 0; i < ElePs.size(); i++) {
    
    bool VBFJetPair = false;
    std::vector<int> store_jet;
    TRefVector refVector;
    
    for (unsigned int j = 0; j < theJetCollection.size(); j++) {
      TVector3 JetP(theJetCollection[j].px(), theJetCollection[j].py(),
                    theJetCollection[j].pz());
      double DR = ElePs[i].DeltaR(JetP);
      
      if (JetP.Pt() > minJetPt_ && std::abs(JetP.Eta()) < maxAbsJetEta_ && DR > minDeltaR_) {
	store_jet.push_back(j);
	// The VBF part of the filter
	if ( minDeltaEta_ > 0 ) {
	  for ( unsigned int k = j+1; k < theJetCollection.size(); k++ ) {
	    TVector3 SoftJetP(theJetCollection[k].px(), theJetCollection[k].py(),
			      theJetCollection[k].pz());
	    double softDR = ElePs[i].DeltaR(SoftJetP);
	    
	    if (SoftJetP.Pt() > minSoftJetPt_ && std::abs(SoftJetP.Eta()) < maxAbsJetEta_ && softDR > minDeltaR_)
	      if ( std::abs(SoftJetP.Eta() - JetP.Eta()) > minDeltaEta_ ) {
		store_jet.push_back(k);
		VBFJetPair = true;
	      }
	  }
	}
      }
      
    }
    
    // Now remove duplicates from the jet collection to store
    std::sort( store_jet.begin(), store_jet.end() );
    store_jet.erase( unique( store_jet.begin(), store_jet.end() ), store_jet.end() );
    
    // Now save the cleaned jets
    for ( unsigned int ijet = 0; ijet < store_jet.size(); ijet++ )
      {
	//store all selections
	refVector.push_back(TRef(theJetCollectionHandle, store_jet.at(ijet)));
	//store first selection which matches the criteria
	if(!foundSolution)
	  theFilteredJetCollection->push_back(theJetCollection[store_jet.at(ijet)]);
      }
    //store all selections
    allSelections->push_back(refVector);
    
    if (theFilteredJetCollection->size() >= minNJets_ && minDeltaEta_ < 0)
      foundSolution = true;
    else if (VBFJetPair && minDeltaEta_ > 0)
      foundSolution = true;
    else if (!foundSolution)
      theFilteredJetCollection->clear();
    
    
  }
  
  iEvent.put(theFilteredJetCollection);
  
  return;
  
}
