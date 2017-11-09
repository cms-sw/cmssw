// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

class TriggerObjectTableProducer : public edm::global::EDProducer<> {
    public:
        explicit TriggerObjectTableProducer(const edm::ParameterSet &iConfig) :
            name_(iConfig.getParameter<std::string>("name")),
            src_(consumes<std::vector<pat::TriggerObjectStandAlone>>(iConfig.getParameter<edm::InputTag>("src")))
        {
            std::vector<edm::ParameterSet> selPSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("selections");
            sels_.reserve(selPSets.size());
            std::stringstream idstr, qualitystr;
            idstr << "ID of the object: ";
            for (auto & pset : selPSets) {
                sels_.emplace_back(pset);
                idstr << sels_.back().id << " = " << sels_.back().name;
                if (sels_.size() < selPSets.size()) idstr << ", ";
                if (!sels_.back().qualityBitsDoc.empty()) { 
                    qualitystr << sels_.back().qualityBitsDoc << " for " << sels_.back().name << "; ";
                }
            }
            idDoc_ = idstr.str();
            bitsDoc_ = qualitystr.str();

            produces<nanoaod::FlatTable>();
        }

        ~TriggerObjectTableProducer() override {}

    private:
        void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override ;

        std::string name_;
        edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> src_;
        std::string idDoc_, bitsDoc_;

        struct SelectedObject {
            std::string name;
            int id;
            StringCutObjectSelector<pat::TriggerObjectStandAlone> cut;
            StringCutObjectSelector<pat::TriggerObjectStandAlone> l1cut, l2cut;
            float       l1DR2, l2DR2;
            StringObjectFunction<pat::TriggerObjectStandAlone> qualityBits;
            std::string qualityBitsDoc;

            SelectedObject(const edm::ParameterSet & pset) :
                name(pset.getParameter<std::string>("name")),
                id(pset.getParameter<int>("id")),
                cut(pset.getParameter<std::string>("sel")),
                l1cut(""), l2cut(""),
                l1DR2(-1), l2DR2(-1),
                qualityBits(pset.getParameter<std::string>("qualityBits")),
                qualityBitsDoc(pset.getParameter<std::string>("qualityBitsDoc"))
            {
                if (pset.existsAs<std::string>("l1seed")) {
                    l1cut = StringCutObjectSelector<pat::TriggerObjectStandAlone>(pset.getParameter<std::string>("l1seed"));
                    l1DR2 = std::pow(pset.getParameter<double>("l1deltaR"), 2);
                }
                if (pset.existsAs<std::string>("l2seed")) {
                    l2cut = StringCutObjectSelector<pat::TriggerObjectStandAlone>(pset.getParameter<std::string>("l2seed"));
                    l2DR2 = std::pow(pset.getParameter<double>("l2deltaR"), 2);
                }
            }

            bool match(const pat::TriggerObjectStandAlone & obj) const {
                return cut(obj);
            }
        };

        std::vector<SelectedObject> sels_;
};

// ------------ method called to produce the data  ------------
void
TriggerObjectTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const 
{

    edm::Handle<std::vector<pat::TriggerObjectStandAlone>> src;
    iEvent.getByToken(src_, src);

    std::vector<std::pair<const pat::TriggerObjectStandAlone *, const SelectedObject *>> selected;
    for (const auto & obj : *src) {
        for (const auto & sel : sels_) {
            if (sel.match(obj)) {
                selected.emplace_back(&obj,&sel);
                break;
            }
        }
    }

    unsigned int nobj = selected.size();
    std::vector<float> pt(nobj,0), eta(nobj,0), phi(nobj,0), l1pt(nobj, 0), l2pt(nobj, 0);
    std::vector<int>   id(nobj,0), bits(nobj, 0);
    for (unsigned int i = 0; i < nobj; ++i) {
        const auto & obj = *selected[i].first;
        const auto & sel = *selected[i].second;
        pt[i] = obj.pt(); 
        eta[i] = obj.eta(); 
        phi[i] = obj.phi(); 
        id[i] = sel.id;
        bits[i] = sel.qualityBits(obj);
        if (sel.l1DR2 > 0) {   
            float best = sel.l1DR2;
            for (const auto & seed : *src) {
                float dr2 = deltaR2(seed, obj);
                if (dr2 < best && sel.l1cut(seed)) {
                    l2pt[i] = seed.pt();
                }
            }
        }
        if (sel.l2DR2 > 0) {
            float best = sel.l2DR2;
            for (const auto & seed : *src) {
                float dr2 = deltaR2(seed, obj);
                if (dr2 < best && sel.l2cut(seed)) {
                    l2pt[i] = seed.pt();
                }
            }
        }
    }

    auto tab  = std::make_unique<nanoaod::FlatTable>(nobj, name_, false, false);
    tab->addColumn<int>("id", id, idDoc_, nanoaod::FlatTable::IntColumn);
    tab->addColumn<float>("pt", pt, "pt", nanoaod::FlatTable::FloatColumn, 12);
    tab->addColumn<float>("eta", eta, "eta", nanoaod::FlatTable::FloatColumn, 12);
    tab->addColumn<float>("phi", phi, "phi", nanoaod::FlatTable::FloatColumn, 12);
    tab->addColumn<float>("l1pt", l1pt, "pt of associated L1 seed", nanoaod::FlatTable::FloatColumn, 10);
    tab->addColumn<float>("l2pt", l2pt, "pt of associated 'L2' seed (i.e. HLT before tracking/PF)", nanoaod::FlatTable::FloatColumn, 10);
    tab->addColumn<float>("filterBits", bits, "extra bits of associated information: "+bitsDoc_, nanoaod::FlatTable::FloatColumn, 10);
    iEvent.put(std::move(tab));
}


//define this as a plug-in
DEFINE_FWK_MODULE(TriggerObjectTableProducer);
