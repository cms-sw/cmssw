/** \class HLTAlphaTFilter
 *
 *
 *  \author Bryn Mathias
 *  \modified Mark Baber, Adam Elwood
 *
 *  Filter for the AlphaT SUSY analysis
 *  Makes a trigger decision based on the event HT
 *  and the AlphaT value. AlphaT cut is chosen
 *  to reject all the QCD background
 *
 */

#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/Handle.h"
#include "HLTrigger/JetMET/interface/HLTAlphaTFilter.h"
#include "HLTrigger/JetMET/interface/AlphaT.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > LorentzV  ;

//
// constructors and destructor
//
template<typename T>
HLTAlphaTFilter<T>::HLTAlphaTFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  inputJetTag_         = iConfig.getParameter< edm::InputTag > ("inputJetTag");
  inputJetTagFastJet_  = iConfig.getParameter< edm::InputTag > ("inputJetTagFastJet");
  minPtJet_            = iConfig.getParameter<std::vector<double> > ("minPtJet");
  etaJet_              = iConfig.getParameter<std::vector<double> > ("etaJet");
  maxNJets_            = iConfig.getParameter<unsigned int> ("maxNJets");
  minHt_               = iConfig.getParameter<double> ("minHt");
  minAlphaT_           = iConfig.getParameter<double> ("minAlphaT");
  triggerType_         = iConfig.getParameter<int>("triggerType");
  dynamicAlphaT_       = iConfig.getParameter<bool>("dynamicAlphaT");
  setDHtZero_          = iConfig.getParameter<bool>("setDHtZero");
  // sanity checks

  if (       (minPtJet_.size()    !=  etaJet_.size())
	     || (  (minPtJet_.size()<1) || (etaJet_.size()<1) )
	     || ( ((minPtJet_.size()<2) || (etaJet_.size()<2))))
    {
      edm::LogError("HLTAlphaTFilter") << "inconsistent module configuration!";
    }

  //register your products
  m_theRecoJetToken = consumes<std::vector<T>>(inputJetTag_);
  m_theFastJetToken = consumes<std::vector<T>>(inputJetTagFastJet_);
}

template<typename T>
HLTAlphaTFilter<T>::~HLTAlphaTFilter(){}

template<typename T>
void HLTAlphaTFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<edm::InputTag>("inputJetTagFastJet",edm::InputTag("hltMCJetCorJetIcone5HF07"));

  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(20.0);
    temp1.push_back(20.0);
    desc.add<std::vector<double> >("minPtJet",temp1);
  }
  desc.add<int>("minNJet",0);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(9999.0);
    temp1.push_back(9999.0);
    desc.add<std::vector<double> >("etaJet",temp1);
  }
  desc.add<unsigned int>("maxNJets",32);
  desc.add<double>("minHt",0.0);
  desc.add<double>("minAlphaT",0.0);
  desc.add<int>("triggerType",trigger::TriggerJet);
  desc.add<bool>("dynamicAlphaT",true); //Set to reproduce old behaviour
  desc.add<bool>("setDHtZero",false); //Set to reproduce old behaviour
  descriptions.add(defaultModuleLabel<HLTAlphaTFilter<T>>(), desc);
}



// ------------ method called to produce the data  ------------
template<typename T>
bool HLTAlphaTFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{

  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  TRef ref;
  // Get the Candidates
  Handle<TCollection> recojets;
  iEvent.getByToken(m_theRecoJetToken,recojets);

  // We have to also look at the L1 FastJet Corrections, at the same time we look at our other jets.
  // We calcualte our HT from the FastJet collection and AlphaT from the standard collection.
  CaloJetRef ref_FastJet;
  // Get the Candidates
  Handle<TCollection> recojetsFastJet;
  iEvent.getByToken(m_theFastJetToken,recojetsFastJet);


  int n(0);

  //OLD - DYNAMIC ALPHAT BEHAVIOUR
  //Produce AlphaT dynamically, first on two jets, then three etc until it does
  //or doesn't pass
  if (dynamicAlphaT_){
    // look at all candidates,  check cuts and add to filter object
    int flag(0);
    double htFast = 0.;
    double aT =0.;
    unsigned int njets(0);

    if(recojets->size() > 1){
      // events with at least two jets, needed for alphaT
      // Make a vector of Lorentz Jets for the AlphaT calcualtion
      std::vector<LorentzV> jets;
      typename TCollection::const_iterator ijet     = recojets->begin();
      typename TCollection::const_iterator ijetFast = recojetsFastJet->begin();
      typename TCollection::const_iterator jjet     = recojets->end();



      for( ; ijet != jjet; ijet++, ijetFast++ ) {
        if( flag == 1) break;
        // Do Some Jet selection!
        if( std::abs(ijet->eta()) > etaJet_.at(0) ) continue;
        if( ijet->et() < minPtJet_.at(0) ) continue;
        njets++;

        if (njets > maxNJets_) //to keep timing reasonable - if too many jets passing pt / eta cuts, just accept the event
          flag = 1;

        else {

          if( std::abs(ijetFast->eta()) < etaJet_.at(1) ){
            if( ijetFast->et() > minPtJet_.at(1) ) {
              // Add to HT
              htFast += ijetFast->et();
            }
          }

          // Add to JetVector
          LorentzV JetLVec(ijet->pt(),ijet->eta(),ijet->phi(),ijet->mass());
          jets.push_back( JetLVec );
          aT = AlphaT(jets,setDHtZero_).value();
          if(htFast > minHt_ && aT > minAlphaT_){
            // set flat to one so that we don't carry on looping though the jets
            flag = 1;
          }
        }

      }
      
      //If passed, add jets to the filter product for the DQM
      if (flag==1) {
        for (typename TCollection::const_iterator recojet = recojets->begin(); recojet!=jjet; recojet++) {
          if (recojet->et() > minPtJet_.at(0)) {
            ref = TRef(recojets,distance(recojets->begin(),recojet));
            filterproduct.addObject(triggerType_,ref);
            n++;
          }
        }
      }
    }// events with at least two jet

    // filter decision
    bool accept(n>0);

    return accept;
  }
  // NEW - STATIC ALPHAT BEHAVIOUR
  // just reproduce
  else{
    // look at all candidates,  check cuts and add to filter object
    int flag(0);
    typename TCollection::const_iterator ijet     = recojets->begin();
    typename TCollection::const_iterator jjet     = recojets->end();


    if( (recojets->size() > 1) && (recojetsFastJet->size() > 1) ){

      // events with at least two jets, needed for alphaT
      double htFast = 0.;
      typename TCollection::const_iterator ijetFast = recojetsFastJet->begin();
      typename TCollection::const_iterator jjetFast = recojetsFastJet->end();
      for( ; ijetFast != jjetFast; ijetFast++ ) {
        if( std::abs(ijetFast->eta()) < etaJet_.at(1) ){
          if( ijetFast->et() > minPtJet_.at(1) ) {
            // Add to HT
            htFast += ijetFast->et();
          }
        }
      }

      if(htFast > minHt_){

        unsigned int njets(0);
        // Make a vector of Lorentz Jets for the AlphaT calcualtion
        std::vector<LorentzV> jets;
        for( ; ijet != jjet; ijet++ ) {
          if( std::abs(ijet->eta()) > etaJet_.at(0) ) continue;
          if( ijet->et() < minPtJet_.at(0) ) continue;
          njets++;

          if (njets > maxNJets_) { //to keep timing reasonable - if too many jets passing pt / eta cuts, just accept the event
            flag = 1;
            break; //Added for efficiency
          }

          // Add to JetVector
          LorentzV JetLVec(ijet->pt(),ijet->eta(),ijet->phi(),ijet->mass());
          jets.push_back( JetLVec );

        }

        if(flag!=1){ //Added for efficiency
          //Calculate the value for alphaT
          float aT = AlphaT(jets,setDHtZero_).value();

          // Trigger decision!
          if(aT > minAlphaT_){
            flag = 1;
          }
        }

        //If passed, add the jets to the filterproduct for DQM
        if (flag==1) {
          for (typename TCollection::const_iterator recojet = recojets->begin(); recojet!=jjet; recojet++) {
            if (recojet->et() > minPtJet_.at(0)) {
              ref = TRef(recojets,distance(recojets->begin(),recojet));
              filterproduct.addObject(triggerType_,ref);
              n++;
            }
          }
        }
      }
    }// events with at least two jet

    // filter decision
    bool accept(n>0);

    return accept;
  }

  // THIS WILL NEVER HAPPEN
  return true;
}
