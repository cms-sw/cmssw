#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>


class IsolatedTrackCleaner : public edm::stream::EDProducer<> {
    public:
        IsolatedTrackCleaner( edm::ParameterSet const & params ) :
            tracks_(consumes<std::vector<pat::IsolatedTrack>>(params.getParameter<edm::InputTag>("tracks"))),
            cut_(params.getParameter<std::string>("cut"))
        {
            for (const edm::InputTag & tag : params.getParameter<std::vector<edm::InputTag>>("finalLeptons")) {
                leptons_.push_back(consumes<reco::CandidateView>(tag));
            }
            produces<std::vector<pat::IsolatedTrack>>();
        }

        ~IsolatedTrackCleaner() override {}

        void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
            auto out  = std::make_unique<std::vector<pat::IsolatedTrack>>();

            std::vector<reco::CandidatePtr> leptonPfCands;
            edm::Handle<reco::CandidateView> leptons;
            for (const auto & token : leptons_) {
                iEvent.getByToken(token, leptons);
                for (const auto & lep : *leptons) {
                    for (unsigned int i = 0, n = lep.numberOfSourceCandidatePtrs(); i < n; ++i) {
                        auto ptr = lep.sourceCandidatePtr(i);
                        if (ptr.isNonnull()) leptonPfCands.push_back(ptr);
                    }
                }
            }
            std::sort(leptonPfCands.begin(), leptonPfCands.end());

            edm::Handle<std::vector<pat::IsolatedTrack>> tracks;
            iEvent.getByToken(tracks_, tracks);
            for (const auto & track : *tracks) {
                if (!cut_(track)) continue; 
                if (track.packedCandRef().isNonnull()) {
                    reco::CandidatePtr pfCand(edm::refToPtr(track.packedCandRef()));
                    if (std::binary_search(leptonPfCands.begin(), leptonPfCands.end(), pfCand)) {
                        continue;
                    }
                }
                out->push_back(track);
            }

            iEvent.put(std::move(out));
        }

    protected:
        edm::EDGetTokenT<std::vector<pat::IsolatedTrack>> tracks_;
        StringCutObjectSelector<pat::IsolatedTrack> cut_;
        std::vector<edm::EDGetTokenT<reco::CandidateView>> leptons_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(IsolatedTrackCleaner);

