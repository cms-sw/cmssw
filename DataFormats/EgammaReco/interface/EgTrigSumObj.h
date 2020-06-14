#ifndef DataFormats_EgammaReco_EgTrigSumObj_h
#define DataFormats_EgammaReco_EgTrigSumObj_h

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include <unordered_map>
#include <string>

namespace reco{
  class RecoEcalCandidate;
}

namespace reco{
  class EgTrigSumObj {
  public:

    EgTrigSumObj():energy_(0.),pt_(0.),eta_(0.),phi_(0.),hasPixelMatch_(false){}

    EgTrigSumObj(float energy,float pt,float eta,float phi);
    EgTrigSumObj(const reco::RecoEcalCandidate& ecalCand);
  
    float energy()const{return energy_;}
    float et()const{return pt();}
    float pt()const{return pt_;}
    float mass()const{return 0.;}
    float eta()const{return eta_;}
    float phi()const{return phi_;}

    const reco::SuperClusterRef& superCluster()const{return superCluster_;}
    const reco::GsfTrackRefVector& gsfTracks()const{return gsfTracks_;}
    const reco::ElectronSeedRefVector& seeds()const{return seeds_;}
   
    void setSuperCluster(const reco::SuperClusterRef& sc){superCluster_ = sc;}
    void setGsfTracks(const reco::GsfTrackRefVector&& trks){gsfTracks_ = trks;}   
    void setSeeds(const reco::ElectronSeedRefVector&& seeds);

    bool hasVar(const std::string& varName)const; 
    float var(const std::string& varName,bool raiseExcept=true)const; 
    const std::unordered_map<std::string,float>& vars()const{return vars_;}
    //varNames and varNamesStr are reasonably expensive functions and are more
    //intended for debugging than normal use
    std::vector<std::string> varNames()const;
    std::string varNamesStr()const;
    void setVar(std::string name,float value,bool overwrite=true); 
    void clearVars(){vars_.clear();}

  private:
    float energy_;
    float pt_;
    float eta_;
    float phi_; 
    bool hasPixelMatch_;
    std::unordered_map<std::string,float> vars_;
    
    reco::SuperClusterRef superCluster_;
    reco::GsfTrackRefVector gsfTracks_;
    //currently these are pixel seeds but could be tracker seeds...
    reco::ElectronSeedRefVector seeds_;
    
  };

}

#endif
