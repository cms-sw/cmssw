#ifndef HSCParticle_H
#define HSCParticle_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPCaloInfo.h"

namespace susybsm {

  /// define arbitration schemes
  namespace HSCParticleType {
    enum Type { globalMuon, trackerMuon, matchedStandAloneMuon, standAloneMuon, innerTrack, unknown };
  }

  class RPCHit4D {
  public:
    int id;
    int bx;
    GlobalPoint gp;
    bool operator<(const RPCHit4D& other) const { return gp.mag() < other.gp.mag(); }
  };

  class RPCBetaMeasurement {
  public:
    bool isCandidate;
    float beta;

    RPCBetaMeasurement() {
      isCandidate = false;
      beta = -9999;
    }
  };

  class HSCParticle {
  public:
    // constructor
    HSCParticle() {}

    // check available infos
    bool hasTrack() const { return track_.packedCandRef().isNonnull(); }
    bool hasMuon() const { return muon_.isNonnull(); }
    bool hasMuonRef() const { return muonRef_.isNonnull(); }
    bool hasMTMuonRef() const { return MTMuonRef_.isNonnull(); }
    bool hasTrackRef() const { return trackRef_.isNonnull(); }
    bool hasTrackIsoRef() const { return trackIsoRef_.isNonnull(); }
    bool hasRpcInfo() const { return rpc_.beta != -9999; }
    bool hasCaloInfo() const { return caloInfoRef_.isNonnull(); }

    // set infos
    void setDeDxHitInfo(const reco::DeDxHitInfo* data) { dedxHitInfo_ = data; }
    void setTrack(const pat::IsolatedTrack& data) { track_ = data; }
    void setMuon(const pat::MuonRef& data) { muon_ = data; }
    void setMuon(const reco::MuonRef& data) { muonRef_ = data; }
    void setMTMuon(const reco::MuonRef& data) { MTMuonRef_ = data; }
    void setTrack(const reco::TrackRef& data) { trackRef_ = data; }
    void setTrackIso(const reco::TrackRef& data) { trackIsoRef_ = data; }
    void setRpc(const RPCBetaMeasurement& data) { rpc_ = data; }
    void setCaloInfo(const HSCPCaloInfoRef& data) { caloInfoRef_ = data; }

    // get infos
    const reco::DeDxHitInfo* dedxHitInfo() const { return dedxHitInfo_; }
    pat::IsolatedTrack track() const { return track_; }
    pat::MuonRef muon() const { return muon_; }
    reco::TrackRef trackRef() const { return trackRef_; }
    reco::TrackRef trackIsoRef() const { return trackIsoRef_; }
    reco::MuonRef muonRef() const { return muonRef_; }
    reco::MuonRef MTMuonRef() const { return MTMuonRef_; }
    HSCPCaloInfoRef caloInfoRef() const { return caloInfoRef_; }
    const RPCBetaMeasurement& rpc() const { return rpc_; }

    // shortcut of long function
    float p() const;
    float pt() const;
    int type() const;

  private:
    const reco::DeDxHitInfo* dedxHitInfo_ = nullptr;
    pat::IsolatedTrack track_;
    pat::MuonRef muon_;
    reco::TrackRef trackRef_;     //TrackRef from refitted track collection (dE/dx purposes)
    reco::TrackRef trackIsoRef_;  //TrackRef from general track collection (isolation purposes)
    reco::MuonRef muonRef_;
    reco::MuonRef MTMuonRef_;  //Muon reconstructed from MT muon segments.  SA only
    HSCPCaloInfoRef caloInfoRef_;

    RPCBetaMeasurement rpc_;
  };

  typedef std::vector<HSCParticle> HSCParticleCollection;
  typedef edm::Ref<HSCParticleCollection> HSCParticleRef;
  typedef edm::RefProd<HSCParticleCollection> HSCParticleRefProd;
  typedef edm::RefVector<HSCParticleCollection> HSCParticleRefVector;
  typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, EcalRecHitCollection> > TracksEcalRecHitsMap;
}  // namespace susybsm
#endif
