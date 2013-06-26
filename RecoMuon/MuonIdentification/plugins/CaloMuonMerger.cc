//
// $Id: CaloMuonMerger.cc,v 1.6 2011/01/06 13:26:53 gpetrucc Exp $
//

/**
  \class    CaloMuonMerger "RecoMuon/MuonIdentification/plugins/CaloMuonMerger.cc"
  \brief    Merges reco::CaloMuons, reco::Muons and optionally reco::Tracks avoiding innerTrack duplications in a single reco::Muon collection
            
  \author   Giovanni Petrucciani
  \version  $Id: CaloMuonMerger.cc,v 1.6 2011/01/06 13:26:53 gpetrucc Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class CaloMuonMerger : public edm::EDProducer {
public:
  explicit CaloMuonMerger(const edm::ParameterSet & iConfig);
  virtual ~CaloMuonMerger() { }

  virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

private:
  edm::InputTag muons_;
  StringCutObjectSelector<reco::Muon, false> muonsCut_;
  bool mergeCaloMuons_;
  edm::InputTag caloMuons_;
  StringCutObjectSelector<reco::CaloMuon, false> caloMuonsCut_;
  double minCaloCompatibility_;
  bool mergeTracks_;
  edm::InputTag tracks_;
  StringCutObjectSelector<reco::TrackRef, false> tracksCut_;
};


CaloMuonMerger::CaloMuonMerger(const edm::ParameterSet & iConfig) :
    muons_(iConfig.getParameter<edm::InputTag>("muons")),
    muonsCut_(iConfig.existsAs<std::string>("muonsCut") ? iConfig.getParameter<std::string>("muonsCut") : ""),
    mergeCaloMuons_(iConfig.existsAs<bool>("mergeCaloMuons") ? iConfig.getParameter<bool>("mergeCaloMuons") : true),
    caloMuons_(iConfig.getParameter<edm::InputTag>("caloMuons")),
    caloMuonsCut_(iConfig.existsAs<std::string>("caloMuonsCut") ? iConfig.getParameter<std::string>("caloMuonsCut") : ""),
    minCaloCompatibility_(mergeCaloMuons_ ? iConfig.getParameter<double>("minCaloCompatibility") : 0),
    mergeTracks_(iConfig.existsAs<bool>("mergeTracks") ? iConfig.getParameter<bool>("mergeTracks") : false),
    tracks_(mergeTracks_ ? iConfig.getParameter<edm::InputTag>("tracks") : edm::InputTag()),
    tracksCut_(iConfig.existsAs<std::string>("tracksCut") ? iConfig.getParameter<std::string>("tracksCut") : "")
{
    produces<std::vector<reco::Muon> >();
}

void 
CaloMuonMerger::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle<std::vector<reco::Muon> > muons;
    edm::Handle<std::vector<reco::CaloMuon> > caloMuons;
    edm::Handle<std::vector<reco::Track> > tracks;

    iEvent.getByLabel(muons_, muons);
    if(mergeCaloMuons_) iEvent.getByLabel(caloMuons_, caloMuons);
    if(mergeTracks_) iEvent.getByLabel(tracks_, tracks);

    std::auto_ptr<std::vector<reco::Muon> >  out(new std::vector<reco::Muon>());
    out->reserve(muons->size() + (mergeTracks_?tracks->size():0));

    // copy reco::Muons, turning on the CaloCompatibility flag if enabled and possible
    for (std::vector<reco::Muon>::const_iterator it = muons->begin(), ed = muons->end(); it != ed; ++it) {
        if(!muonsCut_(*it)) continue;
        out->push_back(*it);
        reco::Muon & mu = out->back();
        if (mergeCaloMuons_ && mu.track().isNonnull()) {
            if (mu.isCaloCompatibilityValid()) {
                if (mu.caloCompatibility() >= minCaloCompatibility_) {
                    mu.setType(mu.type() | reco::Muon::CaloMuon);
                }
            } else throw cms::Exception("Boh") << "Muon with track and no CaloCompatibility; pt = " << mu.pt() << ", eta = " << mu.eta() << ", type = " << mu.type() << "\n";
        }
    }

    if (mergeCaloMuons_) {
        // copy reco::CaloMuon 
        for (std::vector<reco::CaloMuon>::const_iterator it = caloMuons->begin(), ed = caloMuons->end(); it != ed; ++it) {
            if(!caloMuonsCut_(*it)) continue;
            // make a reco::Muon
            reco::TrackRef track = it->track();
            double energy = sqrt(track->p() * track->p() + 0.011163691);
            math::XYZTLorentzVector p4(track->px(), track->py(), track->pz(), energy);
            out->push_back(reco::Muon(track->charge(), p4, track->vertex()));
            reco::Muon & mu = out->back();
            // fill info 
            mu.setCalEnergy( it->calEnergy() );
            mu.setCaloCompatibility( it->caloCompatibility() );
            mu.setInnerTrack( track );
            mu.setType( reco::Muon::CaloMuon );
        }
    }

    // merge reco::Track avoiding duplication of innerTracks
    if(mergeTracks_){
        for (size_t i = 0; i < tracks->size(); i++) {
            reco::TrackRef track(tracks, i);
            if(!tracksCut_(track)) continue;
            // check if it is a muon or calomuon
            bool isMuon = false;
            for(std::vector<reco::Muon>::const_iterator muon = muons->begin(); muon < muons->end(); muon++){
                if(muon->innerTrack() == track){
                    isMuon = true;
                    break;
                }
            }
            if(isMuon) continue;
            if (mergeCaloMuons_) {
                bool isCaloMuon = false;
                for(std::vector<reco::CaloMuon>::const_iterator muon = caloMuons->begin(); muon < caloMuons->end(); muon++){
                    if(muon->innerTrack() == track){
                        isCaloMuon = true;
                        break;
                    }
                }
                if(isCaloMuon) continue;
            }
            // make a reco::Muon
            double energy = sqrt(track->p() * track->p() + 0.011163691);
            math::XYZTLorentzVector p4(track->px(), track->py(), track->pz(), energy);
            out->push_back(reco::Muon(track->charge(), p4, track->vertex()));
            reco::Muon & mu = out->back();
            // fill info 
            mu.setInnerTrack( track );
        }
    }

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloMuonMerger);
