#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

EleTkIsolFromCands::TrkCuts::TrkCuts(const edm::ParameterSet& para)
{
  minPt = para.getParameter<double>("minPt");
  auto sq = [](double val){return val*val;};
  minDR2 = sq(para.getParameter<double>("minDR"));
  maxDR2 = sq(para.getParameter<double>("maxDR"));
  minDEta = para.getParameter<double>("minDEta");
  maxDZ = para.getParameter<double>("maxDZ");
  minHits = para.getParameter<int>("minHits");
  minPixelHits = para.getParameter<int>("minPixelHits");
  maxDPtPt = para.getParameter<double>("maxDPtPt");
  
  auto qualNames = para.getParameter<std::vector<std::string> >("allowedQualities");
  auto algoNames = para.getParameter<std::vector<std::string> >("algosToReject");

  for(auto& qualName : qualNames){
    allowedQualities.push_back(reco::TrackBase::qualityByName(qualName));
  }
  for(auto& algoName : algoNames){
    algosToReject.push_back(reco::TrackBase::algoByName(algoName));
  }
  std::sort(algosToReject.begin(),algosToReject.end());

}

edm::ParameterSetDescription EleTkIsolFromCands::TrkCuts::pSetDescript()
{
  edm::ParameterSetDescription desc;
  desc.add<double>("minPt",1.0);
  desc.add<double>("maxDR",0.3);
  desc.add<double>("minDR",0.000);
  desc.add<double>("minDEta",0.005);
  desc.add<double>("maxDZ",0.1);
  desc.add<double>("maxDPtPt",-1);
  desc.add<int>("minHits",8);
  desc.add<int>("minPixelHits",1);
  desc.add<std::vector<std::string> >("allowedQualities");
  desc.add<std::vector<std::string> >("algosToReject");
  return desc;
}

EleTkIsolFromCands::EleTkIsolFromCands(const edm::ParameterSet& para):
  barrelCuts_(para.getParameter<edm::ParameterSet>("barrelCuts")),
  endcapCuts_(para.getParameter<edm::ParameterSet>("endcapCuts"))          
{
  

}

edm::ParameterSetDescription EleTkIsolFromCands::pSetDescript()
{
  edm::ParameterSetDescription desc;
  desc.add("barrelCuts",TrkCuts::pSetDescript());
  desc.add("endcapCuts",TrkCuts::pSetDescript());
  return desc;
}

std::pair<int,double> 
EleTkIsolFromCands::calIsol(const reco::TrackBase& eleTrk,
			    const pat::PackedCandidateCollection& cands,
			    const PIDVeto pidVeto)const
{
  return calIsol(eleTrk.eta(),eleTrk.phi(),eleTrk.vz(),cands,pidVeto);
}

std::pair<int,double> 
EleTkIsolFromCands::calIsol(const double eleEta,const double elePhi,
			    const double eleVZ,
			    const pat::PackedCandidateCollection& cands,
			    const PIDVeto pidVeto)const
{
  double ptSum=0.;
  int nrTrks=0;

  const TrkCuts& cuts = std::abs(eleEta)<1.5 ? barrelCuts_ : endcapCuts_;
  
  for(auto& cand  : cands){
    if(cand.hasTrackDetails() && cand.charge()!=0 && passPIDVeto(cand.pdgId(),pidVeto)){
      const reco::Track& trk = cand.pseudoTrack();
      if(passTrkSel(trk,trk.pt(),cuts,eleEta,elePhi,eleVZ)){	
	ptSum+=trk.pt();
	nrTrks++;
      }
    }
  }
  return {nrTrks,ptSum};	
}



std::pair<int,double> 
EleTkIsolFromCands::calIsol(const reco::TrackBase& eleTrk,
			    const reco::TrackCollection& tracks)const
{
  return calIsol(eleTrk.eta(),eleTrk.phi(),eleTrk.vz(),tracks);
}

std::pair<int,double> 
EleTkIsolFromCands::calIsol(const double eleEta,const double elePhi,
			    const double eleVZ,
			    const reco::TrackCollection& tracks)const
{
  double ptSum=0.;
  int nrTrks=0;

  const TrkCuts& cuts = std::abs(eleEta)<1.5 ? barrelCuts_ : endcapCuts_;
  
  for(auto& trk  : tracks){
    if(passTrkSel(trk,trk.pt(),cuts,eleEta,elePhi,eleVZ)){	
      ptSum+=trk.pt();
      nrTrks++;
    }
  }
  return {nrTrks,ptSum};	
}	

bool EleTkIsolFromCands::passPIDVeto(const int pdgId,const EleTkIsolFromCands::PIDVeto veto)
{
  int pidAbs = std::abs(pdgId);
  switch (veto){
  case PIDVeto::NONE:
    return true;
  case PIDVeto::ELES:
    if(pidAbs==11) return false;
    else return true;
  case PIDVeto::NONELES:
    if(pidAbs==11) return true;
    else return false;
  }
  throw cms::Exception("CodeError") <<
    "invalid PIDVeto "<<static_cast<int>(veto)<<", "<<
    "this is likely due to some static casting of invalid ints somewhere";
}

EleTkIsolFromCands::PIDVeto EleTkIsolFromCands::pidVetoFromStr(const std::string& vetoStr) 
{
  if(vetoStr=="NONE") return PIDVeto::NONE;
  else if(vetoStr=="ELES") return PIDVeto::ELES;
  else if(vetoStr=="NONELES") return PIDVeto::NONELES;
  else{
    throw cms::Exception("CodeError") <<"unrecognised string "<<vetoStr<<", either a typo or this function needs to be updated";
  }
}

bool EleTkIsolFromCands::passTrkSel(const reco::TrackBase& trk,
				    const double trkPt,const TrkCuts& cuts,
				    const double eleEta,const double elePhi,
				    const double eleVZ)
{
  const float dR2 = reco::deltaR2(eleEta,elePhi,trk.eta(),trk.phi());
  const float dEta = trk.eta()-eleEta;
  const float dZ = eleVZ - trk.vz();

  return dR2>=cuts.minDR2 && dR2<=cuts.maxDR2 && 
    std::abs(dEta)>=cuts.minDEta && 
    std::abs(dZ)<cuts.maxDZ &&
    trk.hitPattern().numberOfValidHits() >= cuts.minHits &&
    trk.hitPattern().numberOfValidPixelHits() >=cuts.minPixelHits &&
    (trk.ptError()/trkPt < cuts.maxDPtPt || cuts.maxDPtPt<0) && 
    passQual(trk,cuts.allowedQualities) &&
    passAlgo(trk,cuts.algosToReject) &&
    trkPt > cuts.minPt;
}
    
 
	
bool EleTkIsolFromCands::
passQual(const reco::TrackBase& trk,
	 const std::vector<reco::TrackBase::TrackQuality>& quals)
{
  if(quals.empty()) return true;

  for(auto qual : quals) {
    if(trk.quality(qual)) return true;
  }

  return false;  
}

bool EleTkIsolFromCands::
passAlgo(const reco::TrackBase& trk,
	 const std::vector<reco::TrackBase::TrackAlgorithm>& algosToRej)
{
  return algosToRej.empty() || !std::binary_search(algosToRej.begin(),algosToRej.end(),trk.algo());
}

