#ifndef MUSCLEFITMUONSELECTOR
#define MUSCLEFITMUONSELECTOR

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/GenMuonPair.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Muon.h"

#include "HepMC/GenParticle.h"
#include "HepMC/GenEvent.h"

#include "MuScleFitPlotter.h"

#include <vector>

/**
* This class is used to perform the selection of muon pairs in MuScleFit. <br>
* It receives the event and returns a muon collection containing all the selected muons.
*/
typedef reco::Particle::LorentzVector lorentzVector;

class MuScleFitMuonSelector
{
 public:
  MuScleFitMuonSelector(const edm::InputTag & muonLabel, const int muonType, const bool PATmuons,
			const std::vector<int> & resfind,
			const bool speedup, const std::string & genParticlesName,
			const bool compareToSimTracks, const edm::InputTag & simTracksCollectionName,
			const bool sherpa, const bool debug) :
    muonLabel_(muonLabel),
    muonType_(muonType),
    PATmuons_(PATmuons),
    resfind_(resfind),
    speedup_(speedup),
    genParticlesName_(genParticlesName),
    compareToSimTracks_(compareToSimTracks),
    simTracksCollectionName_(simTracksCollectionName),
    sherpa_(sherpa),
    debug_(debug)
  {}
  ~MuScleFitMuonSelector() {}

  //Method to get the muon after FSR (status 1 muon in PYTHIA6) starting from status 3 muon which is daughter of the Z
  const reco::Candidate* getStatus1Muon(const reco::Candidate* status3Muon);

  //Method to get the muon before FSR (status 3 muon in PYTHIA6) starting from status 3 muon which is daughter of the Z
  const reco::Candidate* getStatus3Muon(const reco::Candidate* status3Muon);
  
  /// Main method used to select muons of type specified by muonType_ from the collection specified by muonLabel_ and PATmuons_
  void selectMuons(const edm::Event & event, std::vector<MuScleFitMuon> & muons,
		   std::vector<GenMuonPair> & genPair,
		   std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
		   MuScleFitPlotter * plotter);


 protected:
  /// Apply the Onia cuts to select globalMuons
  bool selGlobalMuon(const pat::Muon* aMuon);
  /// Apply the Onia cuts to select trackerMuons
  bool selTrackerMuon(const pat::Muon* aMuon);

  // Generator and simulation level information
  GenMuonPair findGenMuFromRes( const reco::GenParticleCollection* genParticles);
  GenMuonPair findGenMuFromRes( const edm::HepMCProduct* evtMC );
  std::pair<lorentzVector, lorentzVector> findSimMuFromRes( const edm::Handle<edm::HepMCProduct> & evtMC,
							    const edm::Handle<edm::SimTrackContainer> & simTracks );
  void selectGeneratedMuons(const edm::Handle<pat::CompositeCandidateCollection> & collAll,
			    const std::vector<const pat::Muon*> & collMuSel,
			    std::vector<GenMuonPair> & genPair,
			    MuScleFitPlotter * plotter);
  void selectGenSimMuons(const edm::Event & event,
			 std::vector<GenMuonPair> & genPair,
			 std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
			 MuScleFitPlotter * plotter);
  // void selectGeneratedMuons(const edm::Event & event, std::vector<std::pair<lorentzVector,lorentzVector> > & genPair);
  void selectSimulatedMuons(const edm::Event & event,
			    const bool ifHepMC, edm::Handle<edm::HepMCProduct> evtMC,
			    std::vector<std::pair<lorentzVector,lorentzVector> > & simPair,
			    MuScleFitPlotter * plotter);

  /// Template function used to convert the muon collection to a vector of reco::LeafCandidate
  template<typename T>
  std::vector<MuScleFitMuon> fillMuonCollection( const std::vector<T>& tracks )
  {
    std::vector<MuScleFitMuon> muons;
    typename std::vector<T>::const_iterator track;
    for( track = tracks.begin(); track != tracks.end(); ++track ) {
      reco::Particle::LorentzVector mu;
      mu = reco::Particle::LorentzVector(track->px(),track->py(),track->pz(),
					 sqrt(track->p()*track->p() + mMu2));
      
      Double_t hitsTk(0), hitsMuon(0), ptError(0);
      if ( const reco::Muon* myMu = dynamic_cast<const reco::Muon*>(&(*track))  ){
	hitsTk =   myMu->innerTrack()->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->innerTrack()->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->innerTrack()->ptError();
      }
      else if ( const pat::Muon* myMu = dynamic_cast<const pat::Muon*>(&(*track)) ) {
	hitsTk =   myMu->innerTrack()->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->innerTrack()->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->innerTrack()->ptError();
      }
      else if (const reco::Track* myMu = dynamic_cast<const reco::Track*>(&(*track))){
	hitsTk =   myMu->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->ptError();
      }
      
      MuScleFitMuon muon(mu,track->charge(),ptError,hitsTk,hitsMuon);

    if (debug_>0) {
      std::cout<<"[MuScleFitMuonSelector::fillMuonCollection] after MuScleFitMuon initialization"<<std::endl;
      std::cout<<"  muon = "<<muon<<std::endl;
    }

    muons.push_back(muon);
    }
    return muons;
  }

  /// Template function used to extract the selected muon type from the muon collection
  template<typename T>
  void takeSelectedMuonType(const T & muon, std::vector<reco::Track> & tracks)
  {
    // std::cout<<"muon "<<muon->isGlobalMuon()<<muon->isStandAloneMuon()<<muon->isTrackerMuon()<<std::endl;
    //NNBB: one muon can be of many kinds at once but with the muonType_ we are sure
    // to avoid double counting of the same muon
    if(muon->isGlobalMuon() && muonType_==1)
      tracks.push_back(*(muon->globalTrack()));
    else if(muon->isStandAloneMuon() && muonType_==2)
      tracks.push_back(*(muon->outerTrack()));
    else if(muon->isTrackerMuon() && muonType_==3)
      tracks.push_back(*(muon->innerTrack()));

    else if( muonType_ == 10 && !(muon->isStandAloneMuon()) ) //particular case!!
      tracks.push_back(*(muon->innerTrack()));
    else if( muonType_ == 11 && muon->isGlobalMuon() )
      tracks.push_back(*(muon->innerTrack()));
    else if( muonType_ == 13 && muon->isTrackerMuon() )
      tracks.push_back(*(muon->innerTrack()));
  }

  const edm::InputTag muonLabel_;
  const int muonType_;
  const bool PATmuons_;
  const std::vector<int> resfind_;
  const bool speedup_;
  const std::string genParticlesName_;
  const bool compareToSimTracks_;
  const edm::InputTag simTracksCollectionName_;
  const bool sherpa_;
  const bool debug_;
  static const double mMu2;
  static const unsigned int motherPdgIdArray[6];
};

#endif
