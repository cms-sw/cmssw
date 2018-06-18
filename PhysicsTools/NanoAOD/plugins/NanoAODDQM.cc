#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <regex>
#include <sstream>
#include <numeric>
 
namespace {
    std::string replaceStringsToColumGets(const std::string & expr, const nanoaod::FlatTable & table) {
        std::regex token("\\w+");
        std::sregex_iterator tbegin(expr.begin(), expr.end(), token), tend;
        if (tbegin == tend) return expr;
        std::stringstream out;
        std::sregex_iterator last;
        for (std::sregex_iterator i = tbegin; i != tend; last = i, ++i) {
            std::smatch match = *i;
            out << match.prefix().str();
            if (table.columnIndex(match.str()) != -1) {
                out << "getAnyValue(\"" << match.str() << "\")";
            } else {
                out << match.str();
            }
        }   
        out << last->suffix().str();
        return out.str();
    };
}

class NanoAODDQM : public DQMEDAnalyzer {
    public:
        typedef nanoaod::FlatTable FlatTable;

        NanoAODDQM(const edm::ParameterSet&);
        void analyze(const edm::Event&, const edm::EventSetup&) override;

    protected:
        //Book histograms
        void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
        void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override {}
        void endRun(const edm::Run&, const edm::EventSetup&) override {}

    private:
        class Plot {
            public: 
                Plot(MonitorElement * me) : plot_(me) {}
                virtual ~Plot() {}
                virtual void fill(const FlatTable & table, const std::vector<bool> & rowsel) = 0;
                const std::string & name() const { return plot_->getName(); }
            protected:
                MonitorElement * plot_;
        };
        class Count1D : public Plot {
            public:
                Count1D(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    Plot(booker.book1D(cfg.getParameter<std::string>("name"), cfg.getParameter<std::string>("title"), cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max")))
                {
                }
                ~Count1D() override {}
                void fill(const FlatTable & table, const std::vector<bool> & rowsel) override {
                    plot_->Fill(std::accumulate(rowsel.begin(), rowsel.end(), 0u));
                }
        };

        class Plot1D : public Plot {
            public:
                Plot1D(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    Plot(booker.book1D(cfg.getParameter<std::string>("name"), cfg.getParameter<std::string>("title"), cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max"))),
                    col_(cfg.getParameter<std::string>("column"))
                {
                }
                ~Plot1D() override {}
                void fill(const FlatTable & table, const std::vector<bool> & rowsel) override {
                    int icol = table.columnIndex(col_);
                    if (icol == -1) return; // columns may be missing (e.g. mc-only)
                    switch(table.columnType(icol)) {
                        case FlatTable::FloatColumn: vfill<float>(table,icol,rowsel); break;
                        case FlatTable::IntColumn: vfill<int>(table,icol,rowsel); break;
                        case FlatTable::UInt8Column: vfill<uint8_t>(table,icol,rowsel); break;
                        case FlatTable::BoolColumn: vfill<uint8_t>(table,icol,rowsel); break;
                    }
                }
            protected:
                std::string col_;
                template<typename T>
                void vfill(const FlatTable & table, int icol, const std::vector<bool> & rowsel) {
                    const auto & data = table.columnData<T>(icol);
                    for (unsigned int i = 0, n = data.size(); i < n; ++i) {
                        if (rowsel[i]) plot_->Fill(data[i]);
                    }
                }
        };
        class Profile1D : public Plot {
            public:
                Profile1D(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) :
                    Plot(booker.bookProfile(cfg.getParameter<std::string>("name"), cfg.getParameter<std::string>("title"), 
                            cfg.getParameter<uint32_t>("nbins"), cfg.getParameter<double>("min"), cfg.getParameter<double>("max"),
                            0., 0., "")),
                    ycol_(cfg.getParameter<std::string>("ycolumn")), xcol_(cfg.getParameter<std::string>("xcolumn"))
                {
                }
                ~Profile1D() override {}
                void fill(const FlatTable & table, const std::vector<bool> & rowsel) override {
                    int icolx = table.columnIndex(xcol_);
                    int icoly = table.columnIndex(ycol_);
                    if (icolx == -1) throw cms::Exception("LogicError", "Missing "+xcol_);
                    if (icoly == -1) throw cms::Exception("LogicError", "Missing "+ycol_);
                    for (unsigned int irow = 0, n = table.size(); irow < n; ++irow) {
                        if (rowsel[irow]) plot_->Fill(table.getAnyValue(irow,icolx), table.getAnyValue(irow,icoly));
                    }
                }
            protected:
                std::string ycol_, xcol_;
        };



        static std::unique_ptr<Plot> makePlot(DQMStore::IBooker & booker, const edm::ParameterSet & cfg) {
            const std::string & kind = cfg.getParameter<std::string>("kind");
            if (kind == "none") return nullptr;
            if (kind == "count1d") return std::make_unique<Count1D>(booker,cfg);
            if (kind == "hist1d") return std::make_unique<Plot1D>(booker,cfg);
            if (kind == "prof1d") return std::make_unique<Profile1D>(booker,cfg);
            throw cms::Exception("Configuration", "Unsupported plot kind '"+kind+"'");
        }

        struct SelGroupConfig {
            typedef StringCutObjectSelector<FlatTable::RowView> Selector;
            std::string name;
            std::string cutstr;
            std::unique_ptr<StringCutObjectSelector<FlatTable::RowView>> cutptr;
            std::vector<std::unique_ptr<Plot>> plots;
            SelGroupConfig() : name(), cutstr(), cutptr(), plots() {}
            SelGroupConfig(const std::string & nam, const std::string & cut) : name(nam), cutstr(cut), cutptr(), plots() {}
            bool nullCut() const { return cutstr.empty(); }
            void fillSel(const FlatTable & table, std::vector<bool> & out) {
                out.resize(table.size());
                if (nullCut()) { 
                    std::fill(out.begin(), out.end(), true);
                } else {
                    if (!cutptr) {
                        cutptr.reset(new Selector(replaceStringsToColumGets(cutstr, table)));
                    }
                    for (unsigned int i = 0, n = table.size(); i < n; ++i) {
                        out[i] = (*cutptr)(table.row(i));
                    }
                }
            }
        };
        struct GroupConfig {
            std::vector<edm::ParameterSet> plotPSets;
            std::vector<SelGroupConfig> selGroups;
        };
        std::map<std::string, GroupConfig> groups_;
 };

NanoAODDQM::NanoAODDQM(const edm::ParameterSet & iConfig) 
{
    const edm::ParameterSet & vplots = iConfig.getParameter<edm::ParameterSet>("vplots");
    for (const std::string & name : vplots.getParameterNamesForType<edm::ParameterSet>()) {
        auto & group = groups_[name];
        const auto & pset = vplots.getParameter<edm::ParameterSet>(name);
        group.plotPSets = pset.getParameter<std::vector<edm::ParameterSet>>("plots");
        group.selGroups.emplace_back(); // no selection (all entries)
        const auto & cuts = pset.getParameter<edm::ParameterSet>("sels");
        for (const std::string & cname : cuts.getParameterNamesForType<std::string>()) {
            group.selGroups.emplace_back(cname, cuts.getParameter<std::string>(cname));
        }
    }
    consumesMany<FlatTable>();
}

void NanoAODDQM::bookHistograms(DQMStore::IBooker & booker, edm::Run const &, edm::EventSetup const &) {
    booker.setCurrentFolder("Physics/NanoAODDQM");

    for (auto & pair : groups_) {
        booker.setCurrentFolder("Physics/NanoAODDQM/"+pair.first);
        for (auto & sels : pair.second.selGroups) {
            std::string dir("Physics/NanoAODDQM/"+pair.first);
            if (!sels.nullCut()) dir += "/" + sels.name;
            booker.setCurrentFolder(dir);
            auto & plots = sels.plots;
            plots.clear();
            plots.reserve(pair.second.plotPSets.size());
            for (const auto & cfg : pair.second.plotPSets) {
                auto plot = makePlot(booker, cfg);
                if (plot) plots.push_back(std::move(plot));
            }
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
    for (auto & pair : groups_) {
        const std::string & name = pair.first;
        if (maintables.find(name) == maintables.end()) continue; // may happen for missing collections
        auto & tables = maintables[name];
        const FlatTable * table = tables.first;
        if (!tables.second.empty()) {
            merged = *tables.first;
            for (auto * other : tables.second) {
                merged.addExtension(*other);
            }
            table = & merged;
        }
        std::vector<bool> selbits;
        for (auto & sel : pair.second.selGroups) {
            sel.fillSel(*table, selbits);
        
            for (auto & plot : sel.plots) {
                plot->fill(*table, selbits);
            }
        }
    }    
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NanoAODDQM);
