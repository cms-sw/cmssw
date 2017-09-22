#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "PhysicsTools/NanoAOD/interface/FlatTable.h"

class NanoAODDQM : public DQMEDAnalyzer {
    public:
        NanoAODDQM(const edm::ParameterSet&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    protected:
        //Book histograms
        void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
        virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override {}
        virtual void endRun(const edm::Run&, const edm::EventSetup&) override {}

    private:
        class Plot {
            public:
                Plot(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    col_(cfg.getParameter<std::string>("column")),
                    plot_(booker.book1D(cfg.getParameter<std::string>("name").c_str(), "", cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max")))
                {
                }
                void fill(const FlatTable & table) {
                    if (col_ == "@size") {
                        plot_->Fill(table.size());
                        return;
                    }
                    int icol = table.columnIndex(col_);
                    if (icol == -1) throw cms::Exception("LogicError", "Missing "+col_);
                    switch(table.columnType(icol)) {
                        case FlatTable::FloatColumn: vfill<float>(table,icol); break;
                        case FlatTable::IntColumn: vfill<int>(table,icol); break;
                        case FlatTable::UInt8Column: vfill<uint8_t>(table,icol); break;
                        case FlatTable::BoolColumn: vfill<uint8_t>(table,icol); break;
                    }
                }
            protected:
                std::string col_;
                FlatTable::ColumnType type_;
                MonitorElement * plot_;

                template<typename T>
                void vfill(const FlatTable & table, int icol) {
                    for (const T & x : table.columnData<T>(icol)) plot_->Fill(x);
                }
        };

        std::map<std::string, std::vector<edm::ParameterSet>> psets_;
        std::map<std::string, std::vector<Plot>> plots_;
};

NanoAODDQM::NanoAODDQM(const edm::ParameterSet & iConfig) 
{
    const edm::ParameterSet & vplots = iConfig.getParameter<edm::ParameterSet>("vplots");
    for (const std::string & name : vplots.getParameterNamesForType<std::vector<edm::ParameterSet>>()) {
        psets_[name] = vplots.getParameter<std::vector<edm::ParameterSet>>(name);
    }
    consumesMany<FlatTable>();
}

void NanoAODDQM::bookHistograms(DQMStore::IBooker & booker, edm::Run const &, edm::EventSetup const &) {
    booker.setCurrentFolder("Physics/NanoAODDQM");

    plots_.clear();
    for (const auto & pair : psets_) {
        booker.setCurrentFolder("Physics/NanoAODDQM/"+pair.first);
        std::vector<Plot> & plots = plots_[pair.first];
        plots.reserve(pair.second.size());
        for (const auto & cfg : pair.second) plots.emplace_back(booker, cfg);
    }
}

void NanoAODDQM::analyze(const edm::Event &iEvent, const edm::EventSetup &) {
    std::vector<edm::Handle<FlatTable>> alltables;
    iEvent.getManyByType(alltables);
    std::map<std::string, std::pair<const FlatTable *, std::vector<const FlatTable *>>> maintables;

    for (const auto & htab : alltables) { 
        if (htab->extension()) continue;
        maintables[htab->name()] = std::make_pair(htab.product(), std::vector<const FlatTable *>());
    }
    for (const auto & htab : alltables) { 
        if (htab->extension()) {
            if (maintables.find(htab->name()) == maintables.end()) throw cms::Exception("LogicError","Missing main table for "+htab->name());
            maintables[htab->name()].second.push_back(htab.product());
        }
    }

    FlatTable merged;
    for (auto & pair : plots_) {
        if (maintables.find(pair.first) == maintables.end()) continue; // may happen for missing collections
        auto & tables = maintables[pair.first];
        const FlatTable * table = tables.first;
        if (!tables.second.empty()) {
            merged = *tables.first;
            for (auto * other : tables.second) {
                merged.addExtension(*other);
            }
            table = & merged;
        }
        for (auto & plot : pair.second) plot.fill(*table);
    }    
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NanoAODDQM);
