/** \class HLTMhtHtFilter
*
*
*  \author Gheorghe Lungu
*
*/

#include "HLTrigger/JetMET/interface/HLTMhtHtFilter.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

#include<typeinfo>


//
// constructors and destructor
//
template<typename T>
HLTMhtHtFilter<T>::HLTMhtHtFilter(const edm::ParameterSet& iConfig) : 
  HLTFilter(iConfig),
  inputJetTag_    ( iConfig.template getParameter<edm::InputTag>("inputJetTag") ),
  inputTracksTag_ ( iConfig.template getParameter<edm::InputTag>("inputTracksTag") ),
  minPtJet_       ( iConfig.template getParameter<std::vector<double> >("minPtJet") ),
  etaJet_         ( iConfig.template getParameter<std::vector<double> > ("etaJet") ),
  minPT12_        ( iConfig.template getParameter<double>("minPT12") ),
  minHt_          ( iConfig.template getParameter<double>("minHt") ),
  minMht_         ( iConfig.template getParameter<double>("minMht") ),
  minAlphaT_      ( iConfig.template getParameter<double>("minAlphaT") ),
  minMeff_        ( iConfig.template getParameter<double>("minMeff") ),
  meffSlope_      ( iConfig.template getParameter<double>("meffSlope") ),
  minNJet_        ( iConfig.template getParameter<int>("minNJet") ),
  mode_           ( iConfig.template getParameter<int>("mode") ),
  //----mode=1 for MHT only
  //----mode=2 for Meff
  //----mode=3 for PT12
  //----mode=4 for HT only
  //----mode=5 for HT and AlphaT cross trigger (ALWAYS uses jet ET, not pT)
  usePt_          ( iConfig.template getParameter<bool>("usePt") ),
  useTracks_      ( iConfig.template getParameter<bool>("useTracks") ),
  triggerType_    ( iConfig.template getParameter<int> ("triggerType"))
{
  // sanity checks
  if ( (minPtJet_.size() != etaJet_.size())
       or ( (minPtJet_.size()<1) || (etaJet_.size()<1) )
       or ( ((minPtJet_.size()<2) || (etaJet_.size()<2)) and ( (mode_==1) or (mode_==2) or (mode_ == 5))) 
       ) 
    {
      edm::LogError("HLTMhtHtFilter") << "inconsistent module configuration!";
    }
}

template<typename T>
HLTMhtHtFilter<T>::~HLTMhtHtFilter(){}

template<typename T>
void 
HLTMhtHtFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<double>("minMht",0.0);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(20.0);
    temp1.push_back(20.0);
    desc.add<std::vector<double> >("minPtJet",temp1);
  }
  desc.add<int>("minNJet",0);
  desc.add<int>("mode",2);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(9999.0);
    temp1.push_back(9999.0);
    desc.add<std::vector<double> >("etaJet",temp1);
  }
  desc.add<bool>("usePt",true);
  desc.add<double>("minPT12",0.0);
  desc.add<double>("minMeff",180.0);
  desc.add<double>("meffSlope",1.0);
  desc.add<double>("minHt",0.0);
  desc.add<double>("minAlphaT",0.0);
  desc.add<bool>("useTracks",false);
  desc.add<edm::InputTag>("inputTracksTag",edm::InputTag("hltL3Mouns"));
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(std::string("hlt")+std::string(typeid(HLTMhtHtFilter<T>).name()),desc);
}



// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTMhtHtFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);
  
  // Ref to Candidate object to be recorded in filter object
  TRef ref;

  // get hold of collection of objects
  Handle<TCollection> objects;
  iEvent.getByLabel (inputJetTag_,objects);
  Handle<TrackCollection> tracks;
  if (useTracks_) iEvent.getByLabel(inputTracksTag_,tracks);
  
  // look at all candidates,  check cuts and add to filter object
  int n(0), nj(0), flag(0);
  double ht=0.;
  double mhtx=0., mhty=0.;
  double jetVar;
  double dht = 0.;
  double aT = 0.;
  if(objects->size() > 0){
    // events with at least one jet
    typename TCollection::const_iterator jet ( objects->begin() );
    for (; jet!=objects->end(); jet++) {
      if (flag == 1){break;}
      jetVar = jet->pt();
      if (!usePt_ || mode_==3 ) jetVar = jet->et();

      if (mode_==1 || mode_==2 || mode_ == 5) {//---get MHT
        if (jetVar > minPtJet_.at(1) && std::abs(jet->eta()) < etaJet_.at(1)) {
          mhtx -= jetVar*cos(jet->phi());
          mhty -= jetVar*sin(jet->phi());
          if (mode_==1) ++nj;
        }
      }
      if (mode_==2 || mode_==4 || mode_==5) {//---get HT
        if (jetVar > minPtJet_.at(0) && std::abs(jet->eta()) < etaJet_.at(0)) {
          ht += jetVar;
          nj++;
        }
      }
      if (mode_==3) {//---get PT12
        if (jetVar > minPtJet_.at(0) && std::abs(jet->eta()) < etaJet_.at(0)) {
          nj++;
          mhtx -= jetVar*cos(jet->phi());
          mhty -= jetVar*sin(jet->phi());
          if (nj==2) break;
        }
      }
      if(mode_ == 5){
        double mHT = sqrt( (mhtx*mhtx) + (mhty*mhty) );
	// Make sure to apply jet selection to the jets going into deltaHT as well!!!!!
        if (jetVar > minPtJet_.at(0) && std::abs(jet->eta()) < etaJet_.at(0)) {
	  dht += ( nj < 2 ? jetVar : -1.* jetVar ); //@@ only use for njets < 4
        }
        if ( nj == 2 || nj == 3 ) {
          aT = ( ht - std::abs(dht) ) / ( 2. * sqrt( ( ht*ht ) - ( mHT*mHT  ) ) );
        } else if ( nj > 3 ) {
          aT = ht / ( 2.*sqrt( ( ht*ht ) - ( mHT*mHT  ) ) );
        }
        if(ht > minHt_ && aT > minAlphaT_){
	  // put filter object into the Event
          flag = 1;
        }
      }
    }
    if ( (useTracks_) && (tracks->size()>0) ) {
      for (TrackCollection::const_iterator track = tracks->begin();
           track != tracks->end(); track++) {
        if (mode_==1 || mode_==2 || mode_ == 5) {//---get MHT
          if (track->pt() > minPtJet_.at(1) && std::abs(track->eta()) < etaJet_.at(1)) {
            mhtx -= track->px();
            mhty -= track->py();
	  }
	}
        if (mode_==2 || mode_==4 || mode_==5) {//---get HT
          if (track->pt() > minPtJet_.at(0) && std::abs(track->eta()) < etaJet_.at(0)) {
            ht += track->pt();
            nj++;
	  }
	}
      }
    }

    if( mode_==1 && sqrt(mhtx*mhtx + mhty*mhty) > minMht_ && nj >= minNJet_ ) flag=1;
    if( mode_==2 && sqrt(mhtx*mhtx + mhty*mhty) + meffSlope_*ht > minMeff_) flag=1;
    if( mode_==3 && sqrt(mhtx*mhtx + mhty*mhty) > minPT12_ && nj>1) flag=1;
    if( mode_==4 && ht > minHt_ && nj >= minNJet_ ) flag=1;
    
    if (flag==1) {
      typename TCollection::const_iterator jet ( objects->begin() );
      for (; jet!=objects->end(); jet++) {
	jetVar = jet->pt();
	if (!usePt_ || mode_==3) jetVar = jet->et();
	
	if (jetVar > minPtJet_.at(0)) {
	  ref = TRef(objects,distance(objects->begin(),jet));
	  filterproduct.addObject(triggerType_,ref);
	  n++;
	}
      }
    }
  } // events with at least one jet

  // filter decision
  bool accept(n>0);
  
  return accept;
}
