//
// $Id: CaloMuonMerger.cc,v 1.2 2010/02/11 00:14:28 wmtan Exp $
//

/**
  \class    CaloMuonMerger CaloMuonMerger.h "PhysicsTools/PatAlgos/interface/CaloMuonMerger.h"
  \brief    Merges reco::CaloMuons and reco::Muons in a single reco::Muon collection
            
  \author   Giovanni Petrucciani
  \version  $Id: CaloMuonMerger.cc,v 1.2 2010/02/11 00:14:28 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"


class CaloMuonMerger : public edm::EDProducer {
public:
  explicit CaloMuonMerger(const edm::ParameterSet & iConfig);
  virtual ~CaloMuonMerger() { }

  virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

private:
  edm::InputTag muons_, caloMuons_;
  double minCaloCompatibility_;
};


CaloMuonMerger::CaloMuonMerger(const edm::ParameterSet & iConfig) :
    muons_(iConfig.getParameter<edm::InputTag>("muons")),
    caloMuons_(iConfig.getParameter<edm::InputTag>("caloMuons")),
    minCaloCompatibility_(iConfig.getParameter<double>("minCaloCompatibility"))
{
    produces<std::vector<reco::Muon> >();
}

void 
CaloMuonMerger::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    edm::Handle<std::vector<reco::Muon> > muons;
    edm::Handle<std::vector<reco::CaloMuon> > caloMuons;

    iEvent.getByLabel(muons_, muons);
    iEvent.getByLabel(caloMuons_, caloMuons);

    std::auto_ptr<std::vector<reco::Muon> >  out(new std::vector<reco::Muon>());
    out->reserve(caloMuons->size() + muons->size());

    // copy reco::Muons, turning on the CaloCompatibility flag if possible
    for (std::vector<reco::Muon>::const_iterator it = muons->begin(), ed = muons->end(); it != ed; ++it) {
        out->push_back(*it);
        reco::Muon & mu = out->back();
        if (mu.track().isNonnull()) {
            if (mu.isCaloCompatibilityValid()) {
                if (mu.caloCompatibility() >= minCaloCompatibility_) {
                    mu.setType(mu.type() | reco::Muon::CaloMuon);
                }
            } else throw cms::Exception("Boh") << "Muon with track and no CaloCompatibility; pt = " << mu.pt() << ", eta = " << mu.eta() << ", type = " << mu.type() << "\n";
        }
    }

    // copy reco::CaloMuon 
    for (std::vector<reco::CaloMuon>::const_iterator it = caloMuons->begin(), ed = caloMuons->end(); it != ed; ++it) {
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

    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloMuonMerger);
