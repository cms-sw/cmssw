// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"


class MuonIDTableProducer : public edm::global::EDProducer<> {
    public:
        explicit MuonIDTableProducer(const edm::ParameterSet &iConfig) :
            name_(iConfig.getParameter<std::string>("name")),
            src_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
            srcVtx_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vertices")))
        {
            produces<nanoaod::FlatTable>();
        }

        ~MuonIDTableProducer() override {};

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("muons")->setComment("input muon collection");
            desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"))->setComment("input vertex collection, for dxy/dz");
            desc.add<std::string>("name")->setComment("name of the muon nanoaod::FlatTable we are extending with IDs");
            descriptions.add("muonIDTable", desc);
        }

    private:
        void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override ;

        std::string name_;
        edm::EDGetTokenT<std::vector<pat::Muon>> src_;
        edm::EDGetTokenT<std::vector<reco::Vertex>> srcVtx_;

        static bool isMediumMuonHIP(const pat::Muon& muon) ;
        static bool isSoftMuonHIP(const pat::Muon& muon, const reco::Vertex& vtx) ;
        static bool isTrackerHighPt(const pat::Muon & mu, const reco::Vertex & primaryVertex) ;
};

// ------------ method called to produce the data  ------------
void
MuonIDTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const 
{

    edm::Handle<std::vector<pat::Muon>> muons;
    iEvent.getByToken(src_, muons);
    edm::Handle<std::vector<reco::Vertex>> vertices;
    iEvent.getByToken(srcVtx_, vertices);

    unsigned int ncand = muons->size();

    const reco::Vertex & pv = vertices->front(); // consistent with IP information in slimmedLeptons.

    bool isRun2016BCDEF = (272007 <= iEvent.run() && iEvent.run() <= 278808);
    std::vector<uint8_t> tight(ncand, 0), highPt(ncand, 0), soft(ncand, 0), medium(ncand, 0);
    for (unsigned int i = 0; i < ncand; ++i) {
        const pat::Muon & mu = (*muons)[i];
        tight[i] = muon::isTightMuon(mu, pv);
        highPt[i] = muon::isHighPtMuon(mu, pv) ? 2 : isTrackerHighPt(mu,pv);
        soft[i] = isRun2016BCDEF ? isSoftMuonHIP(mu,pv) : muon::isSoftMuon(mu, pv);
        medium[i] = isRun2016BCDEF ? isMediumMuonHIP(mu) : muon::isMediumMuon(mu);
    }

    auto tab  = std::make_unique<nanoaod::FlatTable>(ncand, name_, false, true);
    tab->addColumn<uint8_t>("tightId", tight, "POG Tight muon ID", nanoaod::FlatTable::BoolColumn);
    tab->addColumn<uint8_t>("highPtId", highPt, "POG highPt muon ID (1 = tracker high pT, 2 = global high pT, which includes tracker high pT)", nanoaod::FlatTable::UInt8Column);
    tab->addColumn<uint8_t>("softId", soft, "POG Soft muon ID (using the relaxed cuts in the data Run 2016 B-F periods, and standard cuts elsewhere)", nanoaod::FlatTable::BoolColumn);
    tab->addColumn<uint8_t>("mediumId", medium, "POG Medium muon ID (using the relaxed cuts in the data Run 2016 B-F periods, and standard cuts elsewhere)", nanoaod::FlatTable::BoolColumn);

    iEvent.put(std::move(tab));
}

bool MuonIDTableProducer::isMediumMuonHIP(const pat::Muon& mu) {
      bool goodGlob = mu.isGlobalMuon() && 
                      mu.globalTrack()->normalizedChi2() < 3 && 
                      mu.combinedQuality().chi2LocalPosition < 12 && 
                      mu.combinedQuality().trkKink < 20; 
      bool isMedium = muon::isLooseMuon(mu) && 
                      mu.innerTrack()->validFraction() > 0.49 && 
                      muon::segmentCompatibility(mu) > (goodGlob ? 0.303 : 0.451); 
      return isMedium; 
}

bool MuonIDTableProducer::isSoftMuonHIP(const pat::Muon& mu, const reco::Vertex& vtx) {
    return muon::isGoodMuon(mu, muon::TMOneStationTight) && 
           mu.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
           mu.innerTrack()->hitPattern().pixelLayersWithMeasurement() > 0 && 
           std::abs(mu.innerTrack()->dxy(vtx.position())) < 0.3 && 
           std::abs(mu.innerTrack()->dz(vtx.position())) < 20.;
}

bool MuonIDTableProducer::isTrackerHighPt(const pat::Muon & mu, const reco::Vertex & primaryVertex) {
        return ( mu.numberOfMatchedStations() > 1 
                         && (mu.muonBestTrack()->ptError()/mu.muonBestTrack()->pt()) < 0.3 
                         && std::abs(mu.muonBestTrack()->dxy(primaryVertex.position())) < 0.2 
                         && std::abs(mu.muonBestTrack()->dz(primaryVertex.position())) < 0.5 
                         && mu.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 
                         && mu.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 );
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(MuonIDTableProducer);
