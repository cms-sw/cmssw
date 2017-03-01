/** \class HLTJetCollectionsVBFFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTJetCollectionsVBFFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"


//
// constructors and destructor
//
template <typename T>
HLTJetCollectionsVBFFilter<T>::HLTJetCollectionsVBFFilter(const edm::ParameterSet& iConfig): HLTFilter(iConfig),
   inputTag_(iConfig.getParameter< edm::InputTag > ("inputTag")),
   originalTag_(iConfig.getParameter< edm::InputTag > ("originalTag")),
   softJetPt_(iConfig.getParameter<double> ("SoftJetPt")),
   hardJetPt_(iConfig.getParameter<double> ("HardJetPt")),
   minDeltaEta_(iConfig.getParameter<double> ("MinDeltaEta")),
   thirdJetPt_(iConfig.getParameter<double> ("ThirdJetPt")),
   maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
   maxAbsThirdJetEta_(iConfig.getParameter<double> ("MaxAbsThirdJetEta")),
   minNJets_(iConfig.getParameter<unsigned int> ("MinNJets")),
   triggerType_(iConfig.getParameter<int> ("TriggerType"))
{
  typedef std::vector<edm::RefVector<std::vector<T>,T,edm::refhelper::FindUsingAdvance<std::vector<T>,T> > > TCollectionVector;
  m_theJetToken = consumes<TCollectionVector>(inputTag_);
}

template <typename T>
HLTJetCollectionsVBFFilter<T>::~HLTJetCollectionsVBFFilter(){}

template <typename T>
void
HLTJetCollectionsVBFFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<edm::InputTag>("originalTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<double>("SoftJetPt",25.0);
  desc.add<double>("HardJetPt",35.0);
  desc.add<double>("MinDeltaEta",3.0);
  desc.add<double>("ThirdJetPt",20.0);
  desc.add<double>("MaxAbsJetEta",9999.);
  desc.add<double>("MaxAbsThirdJetEta",2.6);
  desc.add<unsigned int>("MinNJets",2);
  desc.add<int>("TriggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTJetCollectionsVBFFilter<T>>(), desc);
}

// ------------ method called to produce the data  ------------
template <typename T>
bool
HLTJetCollectionsVBFFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;
  typedef edm::RefVector<TCollection> TRefVector;
  typedef std::vector<edm::RefVector<std::vector<T>,T,edm::refhelper::FindUsingAdvance<std::vector<T>,T> > > TCollectionVector;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(originalTag_);

  Handle<TCollectionVector> theJetCollectionsHandle;
  iEvent.getByToken(m_theJetToken, theJetCollectionsHandle);
  const TCollectionVector & theJetCollections = *theJetCollectionsHandle;
  // filter decision
  bool accept(false);
  std::vector < TRef > goodJetRefs;

  for(unsigned int collection = 0; collection < theJetCollections.size(); ++ collection) {

    const TRefVector & refVector =  theJetCollections[collection];
    if(refVector.size() < minNJets_) continue;

    // VBF decision
    bool thereAreVBFJets(false);
    // 3rd Jet check decision
    bool goodThirdJet(false);
    if ( minNJets_ < 3 ) goodThirdJet = true;

    //empty the good jets collection
    goodJetRefs.clear();

    TRef refOne;
    TRef refTwo;
    typename TRefVector::const_iterator jetOne ( refVector.begin() );
    int firstJetIndex=100, secondJetIndex=100, thirdJetIndex=100;

    // Cycle to look for VBF jets
    for (; jetOne != refVector.end(); jetOne++) {
      TRef jetOneRef(*jetOne);

      if ( thereAreVBFJets ) break;
      if ( jetOneRef->pt() < hardJetPt_ ) break;
      if ( std::abs(jetOneRef->eta()) > maxAbsJetEta_ ) continue;

      typename TRefVector::const_iterator jetTwo = jetOne + 1;
      secondJetIndex = firstJetIndex;
      for (; jetTwo != refVector.end(); jetTwo++) {
        TRef jetTwoRef(*jetTwo);

        if ( jetTwoRef->pt() < softJetPt_ ) break;
        if ( std::abs(jetTwoRef->eta()) > maxAbsJetEta_ ) continue;

        if ( std::abs(jetTwoRef->eta() - jetOneRef->eta()) < minDeltaEta_ ) continue;

        thereAreVBFJets = true;
        refOne = *jetOne;
        goodJetRefs.push_back(refOne);
        refTwo = *jetTwo;
        goodJetRefs.push_back(refTwo);

        firstJetIndex = (int) (jetOne - refVector.begin());
        secondJetIndex= (int) (jetTwo - refVector.begin());

        break;

      }
    }// Close looop on VBF

    // Look for a third jet, if you've found the previous 2
    if ( minNJets_ > 2 && thereAreVBFJets ) {
      TRef refThree;
      typename TRefVector::const_iterator jetThree ( refVector.begin() );
      for (; jetThree != refVector.end(); jetThree++) {
        thirdJetIndex = (int) (jetThree - refVector.begin());

        TRef jetThreeRef(*jetThree);

        if ( thirdJetIndex == firstJetIndex || thirdJetIndex == secondJetIndex ) continue;

        if (jetThreeRef->pt() >= thirdJetPt_ && std::abs(jetThreeRef->eta()) <= maxAbsThirdJetEta_) {
          goodThirdJet = true;
          refThree = *jetThree;
          goodJetRefs.push_back(refThree);
          break;
        }
      }
    }

    if(thereAreVBFJets && goodThirdJet){
      accept = true;
      break;
    }

  }

  //fill the filter object
  for (unsigned int refIndex = 0; refIndex < goodJetRefs.size(); ++refIndex) {
    filterproduct.addObject(triggerType_, goodJetRefs.at(refIndex));
  }

  return accept;
}
