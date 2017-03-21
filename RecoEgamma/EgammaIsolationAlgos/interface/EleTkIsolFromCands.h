#ifndef RECOEGAMMA_EGAMMAISOLATIONALGOS_ELETKISOLFROMCANDS_H
#define RECOEGAMMA_EGAMMAISOLATIONALGOS_ELETKISOLFROMCANDS_H

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EleTkIsolFromCands {

private:
  struct TrkCuts {
    float minPt;
    float minDR2;
    float maxDR2;
    float minDEta;
    float maxDZ;
    float minHits;
    float minPixelHits;
    float maxDPtPt;
    std::vector<reco::TrackBase::TrackQuality> allowedQualities;
    std::vector<reco::TrackBase::TrackAlgorithm> algosToReject;
    explicit TrkCuts(const edm::ParameterSet& para);
    static edm::ParameterSetDescription pSetDescript();
  };

  TrkCuts barrelCuts_,endcapCuts_;

public:
  explicit EleTkIsolFromCands(const edm::ParameterSet& para);
  EleTkIsolFromCands(const EleTkIsolFromCands&)=default;
  ~EleTkIsolFromCands()=default;
  EleTkIsolFromCands& operator=(const EleTkIsolFromCands&)=default;

  static edm::ParameterSetDescription pSetDescript();

  std::pair<int,double> calIsol(const reco::TrackBase& trk,const pat::PackedCandidateCollection& cands,const edm::View<reco::GsfElectron>& eles);
  
  std::pair<int,double> calIsol(const double eleEta,const double elePhi,const double eleVZ,
				const pat::PackedCandidateCollection& cands,
				const edm::View<reco::GsfElectron>& eles);
 
  double calIsolPt(const reco::TrackBase& trk,const pat::PackedCandidateCollection& cands,
		   const edm::View<reco::GsfElectron>& eles){
    return calIsol(trk,cands,eles).second;
  }
  
  double calIsolPt(const double eleEta,const double elePhi,const double eleVZ,
		   const pat::PackedCandidateCollection& cands,
		   const edm::View<reco::GsfElectron>& eles){
    return calIsol(eleEta,elePhi,eleVZ,cands,eles).second;
  }

  static bool passTrkSel(const reco::Track& trk,
			 const double trkPt,
			 const TrkCuts& cuts,
			 const double eleEta,const double elePhi,
			 const double eleVZ);
  
private:
  //no qualities specified, accept all, ORed
  static bool passQual(const reco::TrackBase& trk,
		       const std::vector<reco::TrackBase::TrackQuality>& quals);
  static bool passAlgo(const reco::TrackBase& trk,
		       const std::vector<reco::TrackBase::TrackAlgorithm>& algosToRej);
  //for PF electron candidates the "trk pt" is not the track pt
  //its the trk-calo gsfele combination energy * trk sin(theta)
  //so the idea is to match with the gsf electron and get the orginal
  //gsftrack's pt
  double getTrkPt(const reco::TrackBase& trk,
		  const edm::View<reco::GsfElectron>& eles);
};


#endif
