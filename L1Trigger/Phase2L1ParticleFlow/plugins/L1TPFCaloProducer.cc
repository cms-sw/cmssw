// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "L1Trigger/Phase2L1ParticleFlow/src/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/CaloClusterer.h"

//--------------------------------------------------------------------------------------------------
class L1TPFCaloProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1TPFCaloProducer(const edm::ParameterSet&);

    private:
        bool ecalOnly_, debug_;
        std::vector<edm::EDGetTokenT<reco::CandidateView>> ecalCands_;
        std::vector<edm::EDGetTokenT<reco::CandidateView>> hcalCands_;

        std::vector<edm::EDGetTokenT<HcalTrigPrimDigiCollection>> hcalDigis_;
        edm::ESHandle<CaloTPGTranscoder> decoder_;
        std::vector<edm::EDGetTokenT<l1t::HGCalTriggerCellBxCollection>> hcalHGCTCs_;
        std::vector<edm::EDGetTokenT<l1t::HGCalTowerBxCollection>> hcalHGCTowers_;
        double hcalHGCTCEtCut_;
        bool hcalHGCTowersHadOnly_;

        l1tpf::corrector emCorrector_;
        l1tpf::corrector hcCorrector_;
        l1tpf::corrector hadCorrector_;

        l1tpf_calo::SingleCaloClusterer ecalClusterer_, hcalClusterer_;
        std::unique_ptr<l1tpf_calo::SimpleCaloLinkerBase> caloLinker_;

        l1tpf::ParametricResolution resol_;

        virtual void produce(edm::Event&, const edm::EventSetup&) override;

        void readHcalDigis_(edm::Event &event, const edm::EventSetup&) ;
        void readHcalHGCTCs_(edm::Event &event, const edm::EventSetup &) ;
        void readHcalHGCTowers_(edm::Event &event, const edm::EventSetup &) ;
        struct SimpleHGCTC {
            float et, eta, phi;
            SimpleHGCTC(float aet, float aeta, float aphi) : et(aet), eta(aeta), phi(aphi) {}
        };

};

//
// constructors and destructor
//
L1TPFCaloProducer::L1TPFCaloProducer(const edm::ParameterSet& iConfig):
    ecalOnly_(iConfig.existsAs<bool>("ecalOnly") ? iConfig.getParameter<bool>("ecalOnly") : false),
    debug_(iConfig.getUntrackedParameter<int>("debug",0)),
    emCorrector_(iConfig.getParameter<std::string>("emCorrector"), -1, debug_),
    hcCorrector_(iConfig.getParameter<std::string>("hcCorrector"), -1, debug_),
    hadCorrector_(iConfig.getParameter<std::string>("hadCorrector"), iConfig.getParameter<double>("hadCorrectorEmfMax"), debug_),
    ecalClusterer_(iConfig.getParameter<edm::ParameterSet>("ecalClusterer")),
    hcalClusterer_(iConfig.getParameter<edm::ParameterSet>("hcalClusterer")),
    caloLinker_(l1tpf_calo::makeCaloLinker(iConfig.getParameter<edm::ParameterSet>("linker"), ecalClusterer_, hcalClusterer_)),
    resol_(iConfig.getParameter<edm::ParameterSet>("resol"))
{
    produces<l1t::PFClusterCollection>("ecalCells");

    produces<l1t::PFClusterCollection>("emCalibrated");
    produces<l1t::PFClusterCollection>("emUncalibrated");

    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("ecalCandidates")) {
        ecalCands_.push_back(consumes<reco::CandidateView>(tag));
    }

    if (ecalOnly_) return;

    produces<l1t::PFClusterCollection>("hcalCells");

    produces<l1t::PFClusterCollection>("hcalUnclustered");
    produces<l1t::PFClusterCollection>("hcalUncalibrated");
    produces<l1t::PFClusterCollection>("hcalCalibrated");

    produces<l1t::PFClusterCollection>("uncalibrated");
    produces<l1t::PFClusterCollection>("calibrated");

    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("hcalCandidates")) {
        hcalCands_.push_back(consumes<reco::CandidateView>(tag));
    }

    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("hcalDigis")) {
        hcalDigis_.push_back(consumes<HcalTrigPrimDigiCollection>(tag));
    }
    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("hcalHGCTCs")) {
        hcalHGCTCs_.push_back(consumes<l1t::HGCalTriggerCellBxCollection>(tag));
    }
    if (!hcalHGCTCs_.empty()) hcalHGCTCEtCut_ = iConfig.getParameter<double>("hcalHGCTCEtMin");
    for (auto & tag : iConfig.getParameter<std::vector<edm::InputTag>>("hcalHGCTowers")) {
        hcalHGCTowers_.push_back(consumes<l1t::HGCalTowerBxCollection>(tag));
    }
    if (!hcalHGCTowers_.empty()) hcalHGCTowersHadOnly_ = iConfig.getParameter<bool>("hcalHGCTowersHadOnly");


}

// ------------ method called to produce the data  ------------
void
L1TPFCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    /// ----------------ECAL INFO-------------------
    edm::Handle<reco::CandidateView> ecals;
    for (const auto & token : ecalCands_) {
        iEvent.getByToken(token, ecals);
        for (const reco::Candidate & it : *ecals) {
            if (debug_) std::cout << "L1TPFCaloProducer: adding ECal input pt " << it.pt() << ", eta " << it.eta() << ", phi " << it.phi() << std::endl;
            ecalClusterer_.add(it);
        }
    }

    /// ----------------HCAL INFO-------------------
    if (!ecalOnly_) {
        edm::Handle<reco::CandidateView> hcals;
        for (const auto & token : hcalCands_) {
            iEvent.getByToken(token, hcals);
            for (const reco::Candidate & it : *hcals) {
                if (debug_) std::cout << "L1TPFCaloProducer: adding HCal cand input pt " << it.pt() << ", eta " << it.eta() << ", phi " << it.phi() << std::endl;
                hcalClusterer_.add(it);
            }
        }
        if (!hcalDigis_.empty()) {
            readHcalDigis_(iEvent, iSetup);
        }
        if (!hcalHGCTCs_.empty()) {
            readHcalHGCTCs_(iEvent, iSetup);
        }
        if (!hcalHGCTowers_.empty()) {
            readHcalHGCTowers_(iEvent, iSetup);
        }
    }

    /// --------------- CLUSTERING ------------------
    ecalClusterer_.run();

    auto ecalCellsH = iEvent.put(ecalClusterer_.fetchCells(),  "ecalCells");

    iEvent.put(ecalClusterer_.fetch(ecalCellsH),  "emUncalibrated");
   
    if (emCorrector_.valid()) { 
        ecalClusterer_.correct( [&](const l1tpf_calo::Cluster &c) -> float { 
                return emCorrector_.correctedPt(0., c.et, std::abs(c.eta));
                } );
    }
    
    std::unique_ptr<l1t::PFClusterCollection> corrEcal = ecalClusterer_.fetch(ecalCellsH);

    if (debug_) {
        for (const l1t::PFCluster & it : *corrEcal) {
            std::cout << "L1TPFCaloProducer: corrected ECal cluster pt " << it.pt() << ", eta " << it.eta() << ", phi " << it.phi() << std::endl;
        }
    }

    auto ecalClustH = iEvent.put(std::move(corrEcal), "emCalibrated");

    if (ecalOnly_) {
        ecalClusterer_.clear();
        return;
    }


    hcalClusterer_.run();

    auto hcalCellsH = iEvent.put(hcalClusterer_.fetchCells(),  "hcalCells");

    // this we put separately for debugging
    iEvent.put(hcalClusterer_.fetchCells(/*unclustered=*/true),  "hcalUnclustered");

    iEvent.put(hcalClusterer_.fetch(hcalCellsH),  "hcalUncalibrated");

    if (hcCorrector_.valid()) { 
        hcalClusterer_.correct( [&](const l1tpf_calo::Cluster &c) -> float { 
                return hcCorrector_.correctedPt(c.et, 0., std::abs(c.eta));
                } );
    }

    auto hcalClustH = iEvent.put(hcalClusterer_.fetch(hcalCellsH),  "hcalCalibrated");
 
    // Calorimeter linking
    caloLinker_->run();

    iEvent.put(caloLinker_->fetch(ecalClustH,hcalClustH),  "uncalibrated");

    if (hadCorrector_.valid()) {
        caloLinker_->correct( [&](const l1tpf_calo::CombinedCluster &c) -> float { 
                if (debug_) std::cout << "L1TPFCaloProducer: raw linked cluster pt " << c.et << ", eta " << c.eta << ", phi " << c.phi << ", emPt " << c.ecal_et << std::endl;
                return hadCorrector_.correctedPt(c.et, c.ecal_et, std::abs(c.eta)); 
                } );
    }

    std::unique_ptr<l1t::PFClusterCollection> clusters = caloLinker_->fetch(ecalClustH,hcalClustH);
    for (l1t::PFCluster & c : *clusters) {
        c.setPtError(resol_(c.pt(), std::abs(c.eta())));
        if (debug_) std::cout << "L1TPFCaloProducer: calibrated linked cluster pt " << c.pt() << ", eta " << c.eta() << ", phi " << c.phi() << ", emPt " << c.emEt() << std::endl;
    }
    iEvent.put(std::move(clusters),  "calibrated");

    ecalClusterer_.clear();
    hcalClusterer_.clear();
    caloLinker_->clear();
}

void
L1TPFCaloProducer::readHcalDigis_(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    iSetup.get<CaloTPGRecord>().get(decoder_);
    edm::Handle<HcalTrigPrimDigiCollection> hcalTPs;
    for (const auto & token : hcalDigis_) {
        iEvent.getByToken(token, hcalTPs);
        for (const auto & itr : *hcalTPs) {
            HcalTrigTowerDetId id = itr.id();
            double et = decoder_->hcaletValue(itr.id(), itr.t0());
            if (et <= 0) continue;
            float towerEta = l1t::CaloTools::towerEta(id.ieta());
            float towerPhi = l1t::CaloTools::towerPhi(id.ieta(), id.iphi());
            if (debug_) std::cout << "L1TPFCaloProducer: adding HCal digi input pt " << et << ", eta " << towerEta << ", phi " << towerPhi << std::endl;
            hcalClusterer_.add(et, towerEta, towerPhi);
        }
    }
}


void
L1TPFCaloProducer::readHcalHGCTCs_(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<l1t::HGCalTriggerCellBxCollection> hgcTCs;

    std::map<std::pair<int,int>,std::vector<SimpleHGCTC>> pseudoTowers;

    for (const auto & token : hcalHGCTCs_) {
        iEvent.getByToken(token, hgcTCs);
        for(auto it = hgcTCs->begin(0), ed = hgcTCs->end(0); it != ed; ++it) {
            HGCalDetId id(it->detId());
            bool em = id.subdetId() == HGCEE;
            if (em) continue;
            if (it->pt() <= hcalHGCTCEtCut_) continue;
            pseudoTowers[make_pair(id.zside(), id.wafer())].emplace_back(it->pt(), it->eta(), it->phi());
        }

        for (const auto & pair : pseudoTowers) {
            double etsum = 0., etaetsum = 0., phietsum = 0.;
            double reta = pair.second.front().eta, rphi = pair.second.front().phi;
            for (const SimpleHGCTC & hit : pair.second) {
                etsum += hit.et;
                etaetsum += (hit.eta - reta) * hit.et;
                phietsum += reco::deltaPhi(hit.phi, rphi) * hit.et;
            }
            etaetsum /= etsum;
            phietsum /= etsum;
            float eta = etaetsum + reta, phi = reco::deltaPhi(phietsum + rphi, 0.);
            if (debug_) std::cout << "L1TPFCaloProducer: adding HGC TC sum input pt " << etsum << ", eta " << eta << ", phi " << phi << std::endl;
            hcalClusterer_.add(etsum, eta, phi);
        }
        pseudoTowers.clear();

    }
}

void
L1TPFCaloProducer::readHcalHGCTowers_(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<l1t::HGCalTowerBxCollection> hgcTowers;

    for (const auto & token : hcalHGCTowers_) {
        iEvent.getByToken(token, hgcTowers);
        for(auto it = hgcTowers->begin(0), ed = hgcTowers->end(0); it != ed; ++it) {
            if (debug_) std::cout << "L1TPFCaloProducer: adding HGC Tower hadEt " << it->etHad() << ", emEt " << it->etEm() << ", pt " << it->pt() << ", eta " << it->eta() << ", phi " << it->phi() << std::endl;
            hcalClusterer_.add(it->etHad(), it->eta(), it->phi());
            if (!hcalHGCTowersHadOnly_) ecalClusterer_.add(it->etEm(), it->eta(), it->phi());
        }
    }
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFCaloProducer);
