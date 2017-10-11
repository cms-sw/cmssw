#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include <vector>
#include <iostream>


class LHETablesProducer : public edm::global::EDProducer<> {
    public:
        LHETablesProducer( edm::ParameterSet const & params ) :
            lheTag_(consumes<LHEEventProduct>(params.getParameter<edm::InputTag>("lheInfo")))
        {
            produces<nanoaod::FlatTable>();
        }

        ~LHETablesProducer() override {}

        void produce(edm::StreamID id, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {
            auto lheTab  = std::make_unique<nanoaod::FlatTable>(1, "LHE", true);

            edm::Handle<LHEEventProduct> lheInfo;
            if (iEvent.getByToken(lheTag_, lheInfo)) {
                fillLHEObjectTable(*lheInfo, *lheTab);
            }

            iEvent.put(std::move(lheTab));
        }

        void fillLHEObjectTable(const LHEEventProduct & lheProd, nanoaod::FlatTable & out) const {
            double lheHT = 0, lheHTIncoming = 0;
            unsigned int lheNj = 0, lheNb = 0, lheNc = 0, lheNuds = 0, lheNglu = 0;
            double lheVpt = 0;

            const auto & hepeup = lheProd.hepeup();
            const auto & pup = hepeup.PUP;
            int lep = -1, lepBar = -1, nu = -1, nuBar = -1;
            for (unsigned int i = 0, n = pup.size(); i  < n; ++i) {
                int status = hepeup.ISTUP[i];
                int idabs = std::abs(hepeup.IDUP[i]);
                if ( (status == 1) && ( ( idabs == 21 ) || (idabs > 0 && idabs < 7) ) ) { //# gluons and quarks
                    // object counters
                    lheNj++;
                    if (idabs==5) lheNb++;
                    else if (idabs==4) lheNc++;
                    else if (idabs <= 3 ) lheNuds++;
                    else if (idabs == 21) lheNglu++;
                    // HT
                    double pt = std::hypot( pup[i][0], pup[i][1] ); // first entry is px, second py
                    lheHT += pt;
                    int mothIdx = std::max(hepeup.MOTHUP[i].first-1,0); //first and last mother as pair; first entry has index 1 in LHE; incoming particles return motherindex 0
                    int mothIdxTwo = std::max(hepeup.MOTHUP[i].second-1,0);
                    int mothStatus  = hepeup.ISTUP[mothIdx];
                    int mothStatusTwo  = hepeup.ISTUP[mothIdxTwo];
                    bool hasIncomingAsMother = mothStatus<0 || mothStatusTwo<0;
                    if (hasIncomingAsMother) lheHTIncoming += pt;
                } else if (idabs == 12 || idabs == 14 || idabs == 16) {
                    (hepeup.IDUP[i] > 0 ? nu : nuBar) = i;
                } else if (idabs == 11 || idabs == 13 || idabs == 15) {
                    (hepeup.IDUP[i] > 0 ? lep : lepBar) = i;
                }
            }
            std::pair<int,int> v(0,0);
            if      (lep != -1 && lepBar != -1) v = std::make_pair(lep,lepBar);
            else if (lep != -1 &&  nuBar != -1) v = std::make_pair(lep, nuBar);
            else if (nu  != -1 && lepBar != -1) v = std::make_pair(nu ,lepBar);
            else if (nu  != -1 &&  nuBar != -1) v = std::make_pair(nu , nuBar);
            if (v.first != -1 && v.second != -1) {
                lheVpt = std::hypot( pup[v.first][0] + pup[v.second][0], pup[v.first][1] + pup[v.second][1] ); 
            }

            out.addColumnValue<uint8_t>("Njets", lheNj, "Number of jets (partons) at LHE step", nanoaod::FlatTable::UInt8Column);
            out.addColumnValue<uint8_t>("Nb",    lheNb, "Number of b partons at LHE step", nanoaod::FlatTable::UInt8Column);
            out.addColumnValue<uint8_t>("Nc",    lheNc, "Number of c partons at LHE step", nanoaod::FlatTable::UInt8Column);
            out.addColumnValue<uint8_t>("Nuds", lheNuds, "Number of u,d,s partons at LHE step", nanoaod::FlatTable::UInt8Column);
            out.addColumnValue<uint8_t>("Nglu", lheNglu, "Number of gluon partons at LHE step", nanoaod::FlatTable::UInt8Column);
            out.addColumnValue<float>("HT", lheHT, "HT, scalar sum of parton pTs at LHE step", nanoaod::FlatTable::FloatColumn);
            out.addColumnValue<float>("HTIncoming", lheHTIncoming, "HT, scalar sum of parton pTs at LHE step, restricted to partons", nanoaod::FlatTable::FloatColumn);
            out.addColumnValue<float>("Vpt", lheVpt, "pT of the W or Z boson at LHE step", nanoaod::FlatTable::FloatColumn);
        }

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("lheInfo", edm::InputTag("externalLHEProducer"))->setComment("tag for the LHE information (LHEEventProduct)");
            descriptions.add("lheInfoTable", desc);
        }

    protected:
        const edm::EDGetTokenT<LHEEventProduct> lheTag_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHETablesProducer);

