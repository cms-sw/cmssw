


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//
// class declaration
//

class MuonBadTrackFilter : public edm::global::EDFilter<> {
public:
  explicit MuonBadTrackFilter(const edm::ParameterSet&);
  ~MuonBadTrackFilter();

private:
  virtual bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;
  virtual std::string trackInfo(const reco::TrackRef& trackRef) const;
  virtual void printMuonProperties(const reco::MuonRef& muonRef) const;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection>   tokenPFCandidates_;
  const bool taggingMode_;
  const double          ptMin_;
  const double          chi2Min_;
  const double          p1_;
  const double          p2_;
  const double          p3_;
  const bool debug_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonBadTrackFilter::MuonBadTrackFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode")         )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
  , chi2Min_              ( iConfig.getParameter<double>      ("chi2Min")         )
  , p1_                   ( iConfig.getParameter<double>        ("p1")            )
  , p2_                   ( iConfig.getParameter<double>        ("p2")            )
  , p3_                   ( iConfig.getParameter<double>        ("p3")            )
  , debug_                ( iConfig.getParameter<bool>          ("debug")         )
{
  produces<bool>();
}

MuonBadTrackFilter::~MuonBadTrackFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MuonBadTrackFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundBadTrack = false;

  for ( unsigned i=0; i<pfCandidates->size(); ++i ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];
  
    if ( std::abs(cand.pdgId()) != 13 ) continue;

    if (cand.pt() < ptMin_) continue;

    if (cand.muonRef().isNull()) continue;
    
    const reco::MuonRef       muon  = cand.muonRef();
    if ( debug_ ) printMuonProperties(muon);
    
    if (muon->muonBestTrack().isAvailable()) {
      if (muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() == 0) {
        
        if (muon->globalTrack().isAvailable()) {
          if (muon->globalTrack()->normalizedChi2() > chi2Min_) {
            foundBadTrack = true;
            if ( debug_ ) cout << "globalTrack numberOfValidMuonHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonHits() <<
              " numberOfValidMuonCSCHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonCSCHits() <<
              " numberOfValidMuonDTHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonDTHits() <<
              " normalizedChi2: " << muon->globalTrack()->normalizedChi2() << endl;
            if ( debug_ ) cout << "muonBestTrack numberOfValidMuonHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() <<
              " numberOfValidMuonCSCHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonCSCHits() <<
              " numberOfValidMuonDTHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonDTHits() <<
              " normalizedChi2: " << muon->muonBestTrack()->normalizedChi2() << endl;
          }
        }

      }
    }
    
    // perform same check as for charged hadrons
    if (!cand.trackRef().isNull()) {
    
      const reco::TrackRef trackref = cand.trackRef();
      const double Pt = trackref->pt();
      const double DPt = trackref->ptError();
      const double P = trackref->p();
      const unsigned int LostHits = trackref->numberOfLostHits();

      if ((DPt/Pt) > (p1_ * sqrt(p2_*p2_/P+p3_*p3_) / (1.+LostHits))) {

        foundBadTrack = true;

        if ( debug_ ) {
          cout << cand << endl;
          cout << "muon \t" << "track pT = " << Pt << " +/- " << DPt;
          cout << endl;
        }
      }
    }

    // check if at least one track has good quality
    if (muon->innerTrack().isAvailable()) {
      const double P = muon->innerTrack()->p();
      const double DPt = muon->innerTrack()->ptError();
      if (P != 0) {
        if ( debug_ ) cout << "innerTrack DPt/P: " << DPt/P << endl;
        if (DPt/P < 1) {
          if ( debug_ ) cout << "innerTrack good" << endl;
          continue;
        }
      }
    }
    if (muon->pickyTrack().isAvailable()) {
      const double P = muon->pickyTrack()->p();
      const double DPt = muon->pickyTrack()->ptError();
      if (P != 0) {
        if ( debug_ ) cout << "pickyTrack DPt/P: " << DPt/P << endl;
        if (DPt/P < 1) {
          if ( debug_ ) cout << "pickyTrack good" << endl;
          continue;
        }
      }
    }
    if (muon->globalTrack().isAvailable()) {
      const double P = muon->globalTrack()->p();
      const double DPt = muon->globalTrack()->ptError();
      if (P != 0) {
        if ( debug_ ) cout << "globalTrack DPt/P: " << DPt/P << endl;
        if (DPt/P < 1) {
          if ( debug_ ) cout << "globalTrack good" << endl;
          continue;
        }
      }
    }
    if (muon->tpfmsTrack().isAvailable()) {
      const double P = muon->tpfmsTrack()->p();
      const double DPt = muon->tpfmsTrack()->ptError();
      if (P != 0) {
        if ( debug_ ) cout << "tpfmsTrack DPt/P: " << DPt/P << endl;
        if (DPt/P < 1) {
          if ( debug_ ) cout << "tpfmsTrack good" << endl;
          continue;
        }
      }
    }
    if (muon->dytTrack().isAvailable()) {
      const double P = muon->dytTrack()->p();
      const double DPt = muon->dytTrack()->ptError();
      if (P != 0) {
        if ( debug_ ) cout << "dytTrack DPt/P: " << DPt/P << endl;
        if (DPt/P < 1) {
          if ( debug_ ) cout << "dytTrack good" << endl;
          continue;
        }
      }
    }
    if ( debug_ ) cout << "No tracks are good" << endl;
    foundBadTrack = true;
    
    
  } // end loop over PF candidates


  bool pass = !foundBadTrack;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}


std::string MuonBadTrackFilter::trackInfo(const reco::TrackRef& trackRef) const {
  
  std::ostringstream out;
  
  if(trackRef.isNull()) {
    out << "track ref not set"; 
  }
  else if (! trackRef.isAvailable()) {
    out << "track ref not available"; 
  }
  else {
    const reco::Track& track = *trackRef; 
    out << "pt = " << track.pt() << " +- " << track.ptError()/track.pt()
  << " chi2 = " << track.normalizedChi2()
  << "; Muon Hits: " << track.hitPattern().numberOfValidMuonHits()
  << "/" << track.hitPattern().numberOfLostMuonHits()
  << " (DT: " << track.hitPattern().numberOfValidMuonDTHits()
  << "/" << track.hitPattern().numberOfLostMuonDTHits()
  << " CSC: " << track.hitPattern().numberOfValidMuonCSCHits()
  << "/" << track.hitPattern().numberOfLostMuonCSCHits()
  << " RPC: " << track.hitPattern().numberOfValidMuonRPCHits()
  << "/" << track.hitPattern().numberOfLostMuonRPCHits() << ")"
        << "; Valid inner hits:"
  << " TRK: " << track.hitPattern().numberOfValidTrackerHits()
  << " PIX: " << track.hitPattern().numberOfValidPixelHits();  
  }   
  return out.str();
}


void MuonBadTrackFilter::printMuonProperties(const reco::MuonRef& muonRef) const {
  
  if ( !muonRef.isNonnull() ) return;
  
  bool isGL = muonRef->isGlobalMuon();
  bool isTR = muonRef->isTrackerMuon();
  bool isST = muonRef->isStandAloneMuon();
  bool isTPFMS = muonRef->tpfmsTrack().isNonnull() && muonRef->tpfmsTrack()->pt()>0;
  bool isPicky = muonRef->pickyTrack().isNonnull() && muonRef->pickyTrack()->pt()>0;
  bool isDyt = muonRef->dytTrack().isNonnull() && muonRef->dytTrack()->pt()>0;
 
  reco::Muon::MuonTrackType tunePType = muonRef->tunePMuonBestTrackType();
  std::string tunePTypeStr; 
  switch( tunePType ){
  case reco::Muon::InnerTrack: tunePTypeStr = "Inner"; break;
  case reco::Muon::OuterTrack: tunePTypeStr = "Outer"; break;
  case reco::Muon::CombinedTrack: tunePTypeStr = "Combined"; break;
  case reco::Muon::TPFMS: tunePTypeStr = "TPFMS"; break;
  case reco::Muon::Picky: tunePTypeStr = "Picky"; break;
  case reco::Muon::DYT: tunePTypeStr = "DYT"; break;
  default:tunePTypeStr = "unknow"; break;
  } 

  std::cout<<"pt " << muonRef->pt()
     <<" eta " << muonRef->eta()  
           <<" GL: "<<isGL
     <<" TR: "<<isTR
     <<" ST: "<<isST
     <<" TPFMS: "<<isTPFMS
     <<" Picky: "<<isPicky
     <<" DYT: "<<isDyt
     <<" TuneP: "<<tunePTypeStr
     <<" nMatches "<<muonRef->numberOfMatches()<<std::endl;

  if ( isGL ) {
    std::cout<<"\tCombined "<<trackInfo(muonRef->combinedMuon())<<std::endl;
    std::cout<<"\tInner "<<trackInfo(muonRef->innerTrack())<<std::endl;
  }

  if ( isST ) {  
    std::cout<<"\tOuter "<<trackInfo(muonRef->standAloneMuon())<<std::endl;
  }

  if ( isTR ){
    reco::TrackRef trackerMu = muonRef->innerTrack();
    // const reco::Track& track = *trackerMu;
    std::cout<<"\tInner "<<trackInfo(trackerMu)<<std::endl;
    std::cout<< "\t\tTMLastStationAngLoose               "
  << muon::isGoodMuon(*muonRef,muon::TMLastStationAngLoose) << std::endl       
  << "\t\tTMLastStationAngTight               "
  << muon::isGoodMuon(*muonRef,muon::TMLastStationAngTight) << std::endl          
  << "\t\tTMLastStationLoose               "
  << muon::isGoodMuon(*muonRef,muon::TMLastStationLoose) << std::endl       
  << "\t\tTMLastStationTight               "
  << muon::isGoodMuon(*muonRef,muon::TMLastStationTight) << std::endl          
  << "\t\tTMOneStationLoose                "
  << muon::isGoodMuon(*muonRef,muon::TMOneStationLoose) << std::endl       
  << "\t\tTMOneStationTight                "
  << muon::isGoodMuon(*muonRef,muon::TMOneStationTight) << std::endl       
  << "\t\tTMLastStationOptimizedLowPtLoose " 
  << muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtLoose) << std::endl
  << "\t\tTMLastStationOptimizedLowPtTight " 
  << muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedLowPtTight) << std::endl 
  << "\t\tTMLastStationOptimizedBarrelLowPtLoose " 
  << muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedBarrelLowPtLoose) << std::endl
  << "\t\tTMLastStationOptimizedBarrelLowPtTight " 
  << muon::isGoodMuon(*muonRef,muon::TMLastStationOptimizedBarrelLowPtTight) << std::endl 
  << std::endl;
  }

  if( isPicky ) {
    std::cout<<"\tPicky "<<trackInfo(muonRef->pickyTrack())<<std::endl;
  }

  if( isDyt ) {
    std::cout<<"\tDyt "<<trackInfo(muonRef->dytTrack())<<std::endl;
  }

  if( isTPFMS ) {
    std::cout<<"\tTPFMS "<<trackInfo(muonRef->tpfmsTrack())<<std::endl;
  }

  std::cout<< "TM2DCompatibilityLoose           "
     << muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityLoose) << std::endl 
     << "TM2DCompatibilityTight           "
     << muon::isGoodMuon(*muonRef,muon::TM2DCompatibilityTight) << std::endl;

  if ( muonRef->isGlobalMuon() 
       &&  muonRef->isTrackerMuon() 
       &&  muonRef->isStandAloneMuon() ) {
    reco::TrackRef combinedMu = muonRef->combinedMuon();
    reco::TrackRef trackerMu = muonRef->track();
    reco::TrackRef standAloneMu = muonRef->standAloneMuon();
  
    double sigmaCombined = combinedMu->ptError()/(combinedMu->pt()*combinedMu->pt());    
    double sigmaTracker = trackerMu->ptError()/(trackerMu->pt()*trackerMu->pt());    
    double sigmaStandAlone = standAloneMu->ptError()/(standAloneMu->pt()*standAloneMu->pt());    
  
    bool combined = combinedMu->ptError()/combinedMu->pt() < 0.20;   
    bool tracker = trackerMu->ptError()/trackerMu->pt() < 0.20;    
    bool standAlone = standAloneMu->ptError()/standAloneMu->pt() < 0.20;   

    double delta1 =  combined && tracker ?   
      fabs(1./combinedMu->pt() -1./trackerMu->pt())    
      /sqrt(sigmaCombined*sigmaCombined + sigmaTracker*sigmaTracker) : 100.;   
    double delta2 = combined && standAlone ?   
      fabs(1./combinedMu->pt() -1./standAloneMu->pt())   
      /sqrt(sigmaCombined*sigmaCombined + sigmaStandAlone*sigmaStandAlone) : 100.;

    double delta3 = standAlone && tracker ?    
      fabs(1./standAloneMu->pt() -1./trackerMu->pt())    
      /sqrt(sigmaStandAlone*sigmaStandAlone + sigmaTracker*sigmaTracker) : 100.;  

    double delta =   
      standAloneMu->hitPattern().numberOfValidMuonDTHits()+    
      standAloneMu->hitPattern().numberOfValidMuonCSCHits() > 0 ?    
      std::min(delta3,std::min(delta1,delta2)) : std::max(delta3,std::max(delta1,delta2));   

    std::cout << "delta = " << delta << " delta1 "<<delta1<<" delta2 "<<delta2<<" delta3 "<<delta3<<std::endl;   
  
    double ratio =   
      combinedMu->ptError()/combinedMu->pt()   
      / (trackerMu->ptError()/trackerMu->pt());    
    //if ( ratio > 2. && delta < 3. ) std::cout << "ALARM ! " << ratio << ", " << delta << std::endl;
    std::cout<<" ratio "<<ratio<<" combined mu pt "<<combinedMu->pt()<<std::endl;
    //bool quality3 =  ( combinedMu->pt() < 50. || ratio < 2. ) && delta <  3.;
  }

  double sumPtR03 = muonRef->isolationR03().sumPt;
  double emEtR03 = muonRef->isolationR03().emEt;
  double hadEtR03 = muonRef->isolationR03().hadEt;    
  double relIsoR03 = (sumPtR03 + emEtR03 + hadEtR03)/muonRef->pt();
  double sumPtR05 = muonRef->isolationR05().sumPt;
  double emEtR05 = muonRef->isolationR05().emEt;
  double hadEtR05 = muonRef->isolationR05().hadEt;    
  double relIsoR05 = (sumPtR05 + emEtR05 + hadEtR05)/muonRef->pt();
  std::cout<<" 0.3 Rel Iso: "<<relIsoR03<<" sumPt "<<sumPtR03<<" emEt "<<emEtR03<<" hadEt "<<hadEtR03<<std::endl;
  std::cout<<" 0.5 Rel Iso: "<<relIsoR05<<" sumPt "<<sumPtR05<<" emEt "<<emEtR05<<" hadEt "<<hadEtR05<<std::endl;
  return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(MuonBadTrackFilter);
