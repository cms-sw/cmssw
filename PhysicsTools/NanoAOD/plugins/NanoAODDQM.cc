#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include <boost/ptr_container/ptr_vector.hpp>

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
                Plot(MonitorElement * me) : plot_(me) {}
                virtual ~Plot() {}
                virtual void fill(const FlatTable & table) = 0;
                const std::string & name() const { return plot_->getName(); }
            protected:
                MonitorElement * plot_;
        };
        class Plot1D : public Plot {
            public:
                Plot1D(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    Plot(booker.book1D(cfg.getParameter<std::string>("name").c_str(), "", cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max"))),
                    col_(cfg.getParameter<std::string>("column")), once_(true)
                {
                }
                ~Plot1D() override {}
                void fill(const FlatTable & table) override {
                    if (col_ == "@size") {
                        plot_->Fill(table.size());
                        return;
                    }
                    int icol = table.columnIndex(col_);
                    if (icol == -1) throw cms::Exception("LogicError", "Missing "+col_);
                    if (once_) {
                        plot_->setTitle(table.columnDoc(icol));
                        once_ = false;
                    }
                    switch(table.columnType(icol)) {
                        case FlatTable::FloatColumn: vfill<float>(table,icol); break;
                        case FlatTable::IntColumn: vfill<int>(table,icol); break;
                        case FlatTable::UInt8Column: vfill<uint8_t>(table,icol); break;
                        case FlatTable::BoolColumn: vfill<uint8_t>(table,icol); break;
                    }
                }
            protected:
                std::string col_;
                bool once_;
                template<typename T>
                void vfill(const FlatTable & table, int icol) {
                    for (const T & x : table.columnData<T>(icol)) plot_->Fill(x);
                }
        };
        class Profile1D : public Plot {
            public:
                Profile1D(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    Plot(booker.bookProfile(cfg.getParameter<std::string>("name"), "", 
                            cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max"),
                            0., 0., "")),
                    ycol_(cfg.getParameter<std::string>("ycolumn")), xcol_(cfg.getParameter<std::string>("xcolumn")), once_(true)
                {
                }
                ~Profile1D() override {}
                void fill(const FlatTable & table) override {
                    int icolx = table.columnIndex(xcol_);
                    int icoly = table.columnIndex(ycol_);
                    if (icolx == -1) throw cms::Exception("LogicError", "Missing "+xcol_);
                    if (icoly == -1) throw cms::Exception("LogicError", "Missing "+ycol_);
                    if (once_) {
                        plot_->setTitle(table.columnDoc(icoly)+" vs "+xcol_);
                        once_ = false;
                    }
                    for (unsigned int irow = 0, n = table.size(); irow < n; ++irow) {
                        plot_->Fill(table.getAnyValue(irow,icolx), table.getAnyValue(irow,icoly));
                    }
                }
            protected:
                std::string ycol_, xcol_;
                bool once_;
        };



        static std::unique_ptr<Plot> makePlot(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) {
            const std::string & kind = cfg.getParameter<std::string>("kind");
            if (kind == "none") return nullptr;
            if (kind == "hist1d") return std::make_unique<Plot1D>(booker,cfg);
            if (kind == "prof1d") return std::make_unique<Profile1D>(booker,cfg);
            throw cms::Exception("Configuration", "Unsupported plot kind '"+kind+"'");
        }

        std::map<std::string, std::vector<edm::ParameterSet>> psets_;
        std::map<std::string, boost::ptr_vector<Plot>> plots_;
};

NanoAODDQM::NanoAODDQM(const edm::ParameterSet & iConfig) 
{
    const edm::ParameterSet & vplots = iConfig.getParameter<edm::ParameterSet>("vplots");
    for (const std::string & name : vplots.getParameterNamesForType<edm::ParameterSet>()) {
        const auto & pset = vplots.getParameter<edm::ParameterSet>(name);
        psets_[name] = pset.getParameter<std::vector<edm::ParameterSet>>("plots");
    }
    consumesMany<FlatTable>();
}

void NanoAODDQM::bookHistograms(DQMStore::IBooker & booker, edm::Run const &, edm::EventSetup const &) {
    booker.setCurrentFolder("Physics/NanoAODDQM");

    plots_.clear();
    for (const auto & pair : psets_) {
        booker.setCurrentFolder("Physics/NanoAODDQM/"+pair.first);
        boost::ptr_vector<Plot> & plots = plots_[pair.first];
        plots.reserve(pair.second.size());
        for (const auto & cfg : pair.second) {
            auto plot = makePlot(booker, cfg);
            if (plot) plots.push_back(plot.release());
        }
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
        for (auto & plot : pair.second) {
            if (table->columnIndex(plot.name()) != -1) {
                plot.fill(*table);
            }
        }
    }    
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NanoAODDQM);
