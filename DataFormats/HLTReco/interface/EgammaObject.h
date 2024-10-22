#ifndef DataFormats_HLTReco_EgammaObject_h
#define DataFormats_HLTReco_EgammaObject_h

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include <vector>
#include <string>

namespace reco {
  class RecoEcalCandidate;
}

namespace trigger {
  class EgammaObject : public TriggerObject {
  public:
    EgammaObject() : hasPixelMatch_(false) {}

    EgammaObject(int id, float pt, float eta, float phi, float mass)
        : TriggerObject(id, pt, eta, phi, mass), hasPixelMatch_(false) {}
    EgammaObject(const reco::RecoEcalCandidate& ecalCand);

    const reco::SuperClusterRef& superCluster() const { return superCluster_; }
    const reco::GsfTrackRefVector& gsfTracks() const { return gsfTracks_; }
    const reco::ElectronSeedRefVector& seeds() const { return seeds_; }

    void setSuperCluster(const reco::SuperClusterRef& sc) { superCluster_ = sc; }
    void setGsfTracks(reco::GsfTrackRefVector trks) { gsfTracks_ = std::move(trks); }
    void setSeeds(reco::ElectronSeedRefVector seeds);

    bool hasVar(const std::string& varName) const;
    float var(const std::string& varName, bool raiseExcept = true) const;
    const std::vector<std::pair<std::string, float>>& vars() const { return vars_; }
    //varNames and varNamesStr are reasonably expensive functions and are more
    //intended for debugging than normal use
    std::vector<std::string> varNames() const;
    std::string varNamesStr() const;
    void setVars(std::vector<std::pair<std::string, float>> vars);
    void clearVars() { vars_.clear(); }

  private:
    struct VarComparer {
      bool operator()(const std::string& lhs, const std::pair<std::string, float>& rhs) const {
        return lhs < rhs.first;
      }
      bool operator()(const std::pair<std::string, float>& lhs, const std::string& rhs) const {
        return lhs.first < rhs;
      }
    };

    bool hasPixelMatch_;
    std::vector<std::pair<std::string, float>> vars_;

    reco::SuperClusterRef superCluster_;
    reco::GsfTrackRefVector gsfTracks_;
    //currently these are pixel seeds but could be tracker seeds...
    reco::ElectronSeedRefVector seeds_;
  };

}  // namespace trigger

#endif
