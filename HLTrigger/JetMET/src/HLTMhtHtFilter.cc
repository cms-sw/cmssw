/** \class HLTMhtHtFilter
*
*
*  \author Gheorghe Lungu
*
*/

#include "HLTrigger/JetMET/interface/HLTMhtHtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>


//
// constructors and destructor
//
HLTMhtHtFilter::HLTMhtHtFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputJetTag_    ( iConfig.getParameter<edm::InputTag>("inputJetTag") ),
  inputTracksTag_ ( iConfig.getParameter<edm::InputTag>("inputTracksTag") ),
  minPtJet_       ( iConfig.getParameter<std::vector<double> >("minPtJet") ),
  etaJet_         ( iConfig.getParameter<std::vector<double> > ("etaJet") ),
  minPT12_        ( iConfig.getParameter<double>("minPT12") ),
  minHt_          ( iConfig.getParameter<double>("minHt") ),
  minMht_         ( iConfig.getParameter<double>("minMht") ),
  minAlphaT_      ( iConfig.getParameter<double>("minAlphaT") ),
  minMeff_        ( iConfig.getParameter<double>("minMeff") ),
  meffSlope_      ( iConfig.getParameter<double>("meffSlope") ),
  minNJet_        ( iConfig.getParameter<int>("minNJet") ),
  mode_           ( iConfig.getParameter<int>("mode") ),
  //----mode=1 for MHT only
  //----mode=2 for Meff
  //----mode=3 for PT12
  //----mode=4 for HT only
  //----mode=5 for HT and AlphaT cross trigger (ALWAYS uses jet ET, not pT)
  usePt_          ( iConfig.getParameter<bool>("usePt") ),
  useTracks_      ( iConfig.getParameter<bool>("useTracks") )
{
  // sanity checks
  if (       (minPtJet_.size()    !=  etaJet_.size())
       or (  (minPtJet_.size()<1) || (etaJet_.size()<1) )
       or ( ((minPtJet_.size()<2) || (etaJet_.size()<2)) and ( (mode_==1) or (mode_==2) or (mode_ == 5))) 
  ) {
    edm::LogError("HLTMhtHtFilter") << "inconsistent module configuration!";
  }
}

HLTMhtHtFilter::~HLTMhtHtFilter(){}

void HLTMhtHtFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<bool>("saveTags",false);
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
  descriptions.add("hltMhtHtFilter",desc);
}



// ------------ method called to produce the data  ------------
bool
  HLTMhtHtFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  CaloJetRef ref;

  // Get the Candidates
  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  Handle<TrackCollection> tracks;
  if (useTracks_) iEvent.getByLabel(inputTracksTag_,tracks);

  // look at all candidates,  check cuts and add to filter object
  int n(0), nj(0), flag(0);
  double ht=0.;
  double mhtx=0., mhty=0.;
  double jetVar;
  double dht = 0.;
  double aT = 0.;
  if(recocalojets->size() > 0){
    // events with at least one jet
    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin();
    recocalojet != recocalojets->end(); recocalojet++) {
      if (flag == 1){break;}
      jetVar = recocalojet->pt();
      if (!usePt_ || mode_==3 ) jetVar = recocalojet->et();

      if (mode_==1 || mode_==2 || mode_ == 5) {//---get MHT
        if (jetVar > minPtJet_.at(1) && fabs(recocalojet->eta()) < etaJet_.at(1)) {
          mhtx -= jetVar*cos(recocalojet->phi());
          mhty -= jetVar*sin(recocalojet->phi());
          if (mode_==1) ++nj;
        }
      }
      if (mode_==2 || mode_==4 || mode_==5) {//---get HT
        if (jetVar > minPtJet_.at(0) && fabs(recocalojet->eta()) < etaJet_.at(0)) {
          ht += jetVar;
          nj++;
        }
      }
      if (mode_==3) {//---get PT12
        if (jetVar > minPtJet_.at(0) && fabs(recocalojet->eta()) < etaJet_.at(0)) {
          nj++;
          mhtx -= jetVar*cos(recocalojet->phi());
          mhty -= jetVar*sin(recocalojet->phi());
          if (nj==2) break;
        }
      }
      if(mode_ == 5){
        double mHT = sqrt( (mhtx*mhtx) + (mhty*mhty) );
	// Make sure to apply jet selection to the jets going into deltaHT as well!!!!!
        if (jetVar > minPtJet_.at(0) && fabs(recocalojet->eta()) < etaJet_.at(0)) {
	  dht += ( nj < 2 ? jetVar : -1.* jetVar ); //@@ only use for njets < 4
        }
        if ( nj == 2 || nj == 3 ) {
          aT = ( ht - fabs(dht) ) / ( 2. * sqrt( ( ht*ht ) - ( mHT*mHT  ) ) );
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
          if (track->pt() > minPtJet_.at(1) && fabs(track->eta()) < etaJet_.at(1)) {
            mhtx -= track->px();
            mhty -= track->py();
	  }
	}
        if (mode_==2 || mode_==4 || mode_==5) {//---get HT
          if (track->pt() > minPtJet_.at(0) && fabs(track->eta()) < etaJet_.at(0)) {
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
    for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); recocalojet!=recocalojets->end(); recocalojet++) {
      jetVar = recocalojet->pt();
      if (!usePt_ || mode_==3) jetVar = recocalojet->et();

      if (jetVar > minPtJet_.at(0)) {
        ref = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
        filterproduct.addObject(TriggerJet,ref);
        n++;
      }
    }
  }
} // events with at least one jet



  // filter decision
bool accept(n>0);

return accept;
}
