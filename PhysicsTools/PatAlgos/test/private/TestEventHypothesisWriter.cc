#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include "DataFormats/PatCandidates/interface/EventHypothesisLooper.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

class TestEventHypothesisWriter : public edm::EDProducer {
    public:
        TestEventHypothesisWriter(const edm::ParameterSet &iConfig) ;
        virtual void produce( edm::Event &iEvent, const edm::EventSetup &iSetup) ;
        void runTests( const pat::EventHypothesis &h) ;
    private:
        edm::EDGetTokenT<edm::View<reco::Candidate> > jetsToken_;
        edm::EDGetTokenT<edm::View<reco::Candidate> > muonsToken_;
};



TestEventHypothesisWriter::TestEventHypothesisWriter(const edm::ParameterSet &iConfig) :
    jetsToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("jets"))),
    muonsToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("muons")))
{
    produces<std::vector<pat::EventHypothesis> >();
    produces<edm::ValueMap<double> >("deltaR");
}

void
TestEventHypothesisWriter::runTests( const pat::EventHypothesis &h) {
    using namespace std;
    using namespace pat::eventhypothesis;
    cout << "Test 1: Print the muon " << h["mu"]->pt() << endl;

    for (size_t i = 0; i < h.count() - 2; ++i) {
        cout << "Test 2." << (i+1) << ": Getting of the other jets: " << h.get("other jet",i)->et() << endl;
    }

    cout << "Test 3: count: " << (h.count() - 2) << " vs " << h.count("other jet") << endl;

    cout << "Test 4: regexp count: " << (h.count() - 1) << " vs " << h.count(".*jet") << endl;

    cout << "Test 5.0: all with muon: " << h.all("mu").size() << endl;
    cout << "Test 5.1: all with muon: " << h.all("mu").front()->pt() << endl;
    cout << "Test 5.2: all with other jets: " << h.all("other jet").size() << endl;
    cout << "Test 5.3: all with regex: " << h.all(".*jet").size() << endl;

    cout << "Test 6.0: get as : " << h.getAs<reco::CaloJet>("nearest jet")->maxEInHadTowers() << endl;

    cout << "Loopers" << endl;
    cout << "Test 7.0: simple looper on all" << endl;
    for (CandLooper jet = h.loop(); jet; ++jet) {
        cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->et() << endl;
    }
    cout << "Test 7.1: simple looper on jets" << endl;
    for (CandLooper jet = h.loop(".*jet"); jet; ++jet) {
        cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->et() << endl;
    }
    cout << "Test 7.2: loopAs on jets" << endl;
    for (Looper<reco::CaloJet> jet = h.loopAs<reco::CaloJet>(".*jet"); jet; ++jet) {
        cout << "\titem " << jet.index() << ", role " << jet.role() << ": " << jet->maxEInHadTowers() << endl;
    }


}

void
TestEventHypothesisWriter::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
    using namespace edm;
    using namespace std;
    using reco::Candidate;
    using reco::CandidatePtr;

    auto hyps = std::make_unique<std::vector<pat::EventHypothesis>>();;
    vector<double>  deltaRs;

    Handle<View<Candidate> > hMu;
    iEvent.getByToken(muonsToken_, hMu);

    Handle<View<Candidate> > hJet;
    iEvent.getByToken(jetsToken_, hJet);

    // fake analysis
    for (size_t imu = 0, nmu = hMu->size(); imu < nmu; ++imu) {
        pat::EventHypothesis h;
        CandidatePtr mu = hMu->ptrAt(imu);
        h.add(mu, "mu");

        int bestj = -1; double drmin = 99.0;
        for (size_t ij = 0, nj = hJet->size(); ij < nj; ++ij) {
            CandidatePtr jet = hJet->ptrAt(ij);
            if (jet->et() < 10) break;
            double dr = deltaR(*jet, *mu);
            if (dr < drmin) {
                bestj = ij; drmin = dr;
            }
        }
        if (bestj == -1) continue;

        h.add(hJet->ptrAt(bestj), "nearest jet");

        for (size_t ij = 0, nj = hJet->size(); ij < nj; ++ij) {
            if (ij == size_t(bestj)) continue;
            CandidatePtr jet = hJet->ptrAt(ij);
            if (jet->et() < 10) break;
            h.add(jet, "other jet");
        }

        // save hypothesis
        deltaRs.push_back(drmin);

        runTests(h);

        hyps->push_back(h);
    }

    std::cout << "Found " << deltaRs.size() << " possible options" << std::endl;

    // work done, save results
    OrphanHandle<vector<pat::EventHypothesis> > handle = iEvent.put(std::move(hyps));
    auto deltaRMap = std::make_unique<ValueMap<double>>();
    //if (deltaRs.size() > 0) {
        ValueMap<double>::Filler filler(*deltaRMap);
        filler.insert(handle, deltaRs.begin(), deltaRs.end());
        filler.fill();
    //}
    iEvent.put(std::move(deltaRMap), "deltaR");
}

DEFINE_FWK_MODULE(TestEventHypothesisWriter);
