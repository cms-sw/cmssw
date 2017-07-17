#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

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
			    const edm::View<reco::GsfElectron>& eles)
{
  return calIsol(eleTrk.eta(),eleTrk.phi(),eleTrk.vz(),cands,eles);
}

std::pair<int,double> 
EleTkIsolFromCands::calIsol(const double eleEta,const double elePhi,
			    const double eleVZ,
			    const pat::PackedCandidateCollection& cands,
			    const edm::View<reco::GsfElectron>& eles)
{

  double ptSum=0.;
  int nrTrks=0;

  const TrkCuts& cuts = std::abs(eleEta)<1.5 ? barrelCuts_ : endcapCuts_;
  
  for(auto& cand  : cands){
    if(cand.charge()!=0){
      const reco::Track& trk = cand.pseudoTrack(); 
      double trkPt = std::abs(cand.pdgId())!=11 ? trk.pt() : getTrkPt(trk,eles);
      if(passTrkSel(trk,trkPt,cuts,eleEta,elePhi,eleVZ)){	
	ptSum+=trkPt;
	nrTrks++;
      }
    }
  }
  return {nrTrks,ptSum};	
}
	

bool EleTkIsolFromCands::passTrkSel(const reco::Track& trk,
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

//so the working theory here is that the track we have is the electrons gsf track
//if so, lets get the pt of the gsf track before E/p combinations
//if no match found to a gsf ele with a gsftrack, return the pt of the input track
double EleTkIsolFromCands::
getTrkPt(const reco::TrackBase& trk,
	 const edm::View<reco::GsfElectron>& eles)
{
  //note, the trk.eta(),trk.phi() should be identical to the gsf track eta,phi
  //although this may not be the case due to roundings after packing
  auto match=[](const reco::TrackBase& trk,const reco::GsfElectron& ele){
    return std::abs(trk.eta()-ele.gsfTrack()->eta())<0.001 &&
    reco::deltaPhi(trk.phi(),ele.gsfTrack()->phi())<0.001;// && 
  };
  for(auto& ele : eles){
    if(ele.gsfTrack().isNonnull()){
      if(match(trk,ele)){
	return ele.gsfTrack()->pt();
      }
    }
  }
  return trk.pt();

}
