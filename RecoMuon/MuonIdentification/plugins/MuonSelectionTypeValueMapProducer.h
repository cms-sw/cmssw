#ifndef MuonIdentification_MuonSelectionTypeValueMapProducer_h
#define MuonIdentification_MuonSelectionTypeValueMapProducer_h

#include <string>
#include <vector>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonSelectionTypeValueMapProducer : public edm::stream::EDProducer<> {
    public:
        explicit MuonSelectionTypeValueMapProducer(const edm::ParameterSet& iConfig) :
            inputMuonCollection_(iConfig.getParameter<edm::InputTag>("inputMuonCollection")),
            selectionTypeLabel_(iConfig.getParameter<std::string>("selectionType"))
        {
            selectionType_ = muon::selectionTypeFromString(selectionTypeLabel_);
            produces<edm::ValueMap<bool> >().setBranchAlias("muid"+selectionTypeLabel_);
	    muonToken_ = consumes<reco::MuonCollection>(inputMuonCollection_);
        }
        virtual ~MuonSelectionTypeValueMapProducer() {}

    private:
        virtual void produce(edm::Event&, const edm::EventSetup&) override;

        edm::InputTag inputMuonCollection_;
	edm::EDGetTokenT<reco::MuonCollection> muonToken_;

        std::string selectionTypeLabel_;
        muon::SelectionType selectionType_;
};

void
MuonSelectionTypeValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // input muon collection
    edm::Handle<reco::MuonCollection> muonsH;
    iEvent.getByToken(muonToken_, muonsH);

    // reserve some space
    std::vector<bool> values;
    values.reserve(muonsH->size());

    // isGoodMuon
    for(reco::MuonCollection::const_iterator it = muonsH->begin(); it != muonsH->end(); ++it)
        values.push_back(muon::isGoodMuon(*it, selectionType_));

    // create and fill value map
    std::auto_ptr<edm::ValueMap<bool> > out(new edm::ValueMap<bool>());
    edm::ValueMap<bool>::Filler filler(*out);
    filler.insert(muonsH, values.begin(), values.end());
    filler.fill();

    // put value map into event
    iEvent.put(out);
}

#endif
