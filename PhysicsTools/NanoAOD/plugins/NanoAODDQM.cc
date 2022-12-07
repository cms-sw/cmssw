#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

#include <memory>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <type_traits>

namespace {
  std::string replaceStringsToColumGets(const std::string &expr, const nanoaod::FlatTable &table) {
    std::regex token("\\w+");
    std::sregex_iterator tbegin(expr.begin(), expr.end(), token), tend;
    if (tbegin == tend)
      return expr;
    std::stringstream out;
    std::sregex_iterator last;
    for (std::sregex_iterator i = tbegin; i != tend; last = i, ++i) {
      const std::smatch &match = *i;
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
}  // namespace

class NanoAODDQM : public DQMEDAnalyzer {
public:
  typedef nanoaod::FlatTable FlatTable;

  NanoAODDQM(const edm::ParameterSet &);
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  //Book histograms
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  class Plot {
  public:
    Plot(MonitorElement *me) : plot_(me) {}
    virtual ~Plot() {}
    virtual void fill(const FlatTable &table, const std::vector<bool> &rowsel) = 0;
    const std::string &name() const { return plot_->getName(); }

  protected:
    MonitorElement *plot_;
  };
  class Count1D : public Plot {
  public:
    Count1D(DQMStore::IBooker &booker, const edm::ParameterSet &cfg)
        : Plot(booker.book1D(cfg.getParameter<std::string>("name"),
                             cfg.getParameter<std::string>("title"),
                             cfg.getParameter<uint32_t>("nbins"),
                             cfg.getParameter<double>("min"),
                             cfg.getParameter<double>("max"))) {}
    ~Count1D() override {}
    void fill(const FlatTable &table, const std::vector<bool> &rowsel) override {
      plot_->Fill(std::accumulate(rowsel.begin(), rowsel.end(), 0u));
    }
  };

  class Plot1D : public Plot {
  public:
    Plot1D(DQMStore::IBooker &booker, const edm::ParameterSet &cfg)
        : Plot(booker.book1D(cfg.getParameter<std::string>("name"),
                             cfg.getParameter<std::string>("title"),
                             cfg.getParameter<uint32_t>("nbins"),
                             cfg.getParameter<double>("min"),
                             cfg.getParameter<double>("max"))),
          col_(cfg.getParameter<std::string>("column")),
          bitset_(cfg.getParameter<bool>("bitset")) {}
    ~Plot1D() override {}
    void fill(const FlatTable &table, const std::vector<bool> &rowsel) override {
      int icol = table.columnIndex(col_);
      if (icol == -1)
        return;  // columns may be missing (e.g. mc-only)
      switch (table.columnType(icol)) {
        case FlatTable::ColumnType::Int8:
          vfill<int8_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::UInt8:
          vfill<uint8_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::Int16:
          vfill<int16_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::UInt16:
          vfill<uint16_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::Int32:
          vfill<int32_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::UInt32:
          vfill<uint32_t>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::Bool:
          vfill<bool>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::Float:
          vfill<float>(table, icol, rowsel);
          break;
        case FlatTable::ColumnType::Double:
          vfill<double>(table, icol, rowsel);
          break;
        default:
          throw cms::Exception("LogicError", "Unsupported type");
      }
    }

  protected:
    std::string col_;
    bool bitset_;
    template <typename T>
    void vfill(const FlatTable &table, int icol, const std::vector<bool> &rowsel) {
      const auto &data = table.columnData<T>(icol);
      for (unsigned int i = 0, n = data.size(); i < n; ++i) {
        if (rowsel[i]) {
          const T val = data[i];
          if constexpr (std::is_integral<T>::value) {
            if (bitset_) {
              for (unsigned int b = 0; b < std::numeric_limits<T>::digits; b++) {
                if ((val >> b) & 0b1)
                  plot_->Fill(b);
              }
            } else {
              plot_->Fill(val);
            }
          } else {
            plot_->Fill(val);
          }
        }
      }
    }
  };

  class Profile1D : public Plot {
  public:
    Profile1D(DQMStore::IBooker &booker, const edm::ParameterSet &cfg)
        : Plot(booker.bookProfile(cfg.getParameter<std::string>("name"),
                                  cfg.getParameter<std::string>("title"),
                                  cfg.getParameter<uint32_t>("nbins"),
                                  cfg.getParameter<double>("min"),
                                  cfg.getParameter<double>("max"),
                                  0.,
                                  0.,
                                  "")),
          ycol_(cfg.getParameter<std::string>("ycolumn")),
          xcol_(cfg.getParameter<std::string>("xcolumn")) {}
    ~Profile1D() override {}
    void fill(const FlatTable &table, const std::vector<bool> &rowsel) override {
      int icolx = table.columnIndex(xcol_);
      int icoly = table.columnIndex(ycol_);
      if (icolx == -1)
        throw cms::Exception("LogicError", "Missing " + xcol_);
      if (icoly == -1)
        throw cms::Exception("LogicError", "Missing " + ycol_);
      for (unsigned int irow = 0, n = table.size(); irow < n; ++irow) {
        if (rowsel[irow])
          plot_->Fill(table.getAnyValue(irow, icolx), table.getAnyValue(irow, icoly));
      }
    }

  protected:
    std::string ycol_, xcol_;
  };

  static std::unique_ptr<Plot> makePlot(DQMStore::IBooker &booker, const edm::ParameterSet &cfg) {
    const std::string &kind = cfg.getParameter<std::string>("kind");
    if (kind == "none")
      return nullptr;
    if (kind == "count1d")
      return std::make_unique<Count1D>(booker, cfg);
    if (kind == "hist1d")
      return std::make_unique<Plot1D>(booker, cfg);
    if (kind == "prof1d")
      return std::make_unique<Profile1D>(booker, cfg);
    throw cms::Exception("Configuration", "Unsupported plot kind '" + kind + "'");
  }

  struct SelGroupConfig {
    typedef StringCutObjectSelector<FlatTable::RowView> Selector;
    std::string name;
    std::string cutstr;
    std::unique_ptr<StringCutObjectSelector<FlatTable::RowView>> cutptr;
    std::vector<std::unique_ptr<Plot>> plots;
    SelGroupConfig() : name(), cutstr(), cutptr(), plots() {}
    SelGroupConfig(const std::string &nam, const std::string &cut) : name(nam), cutstr(cut), cutptr(), plots() {}
    bool nullCut() const { return cutstr.empty(); }
    void fillSel(const FlatTable &table, std::vector<bool> &out) {
      out.resize(table.size());
      if (nullCut()) {
        std::fill(out.begin(), out.end(), true);
      } else {
        if (!cutptr) {
          cutptr = std::make_unique<Selector>(replaceStringsToColumGets(cutstr, table));
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
  edm::GetterOfProducts<FlatTable> getterOfProducts_;
};

NanoAODDQM::NanoAODDQM(const edm::ParameterSet &iConfig) : getterOfProducts_(edm::ProcessMatch("*"), this) {
  const edm::ParameterSet &vplots = iConfig.getParameter<edm::ParameterSet>("vplots");
  for (const std::string &name : vplots.getParameterNamesForType<edm::ParameterSet>()) {
    auto &group = groups_[name];
    const auto &pset = vplots.getParameter<edm::ParameterSet>(name);
    group.plotPSets = pset.getParameter<std::vector<edm::ParameterSet>>("plots");
    group.selGroups.emplace_back();  // no selection (all entries)
    const auto &cuts = pset.getParameter<edm::ParameterSet>("sels");
    for (const std::string &cname : cuts.getParameterNamesForType<std::string>()) {
      group.selGroups.emplace_back(cname, cuts.getParameter<std::string>(cname));
    }
  }
  callWhenNewProductsRegistered(getterOfProducts_);
}

void NanoAODDQM::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription sels;
  sels.setComment("a paramerter set to define the selections to be made from the table row");
  sels.addNode(edm::ParameterWildcard<std::string>("*", edm::RequireZeroOrMore, true));

  edm::ParameterDescription<std::string> name("name", true, edm::Comment(""));
  edm::ParameterDescription<std::string> title("title", true, edm::Comment("title of the plot"));
  edm::ParameterDescription<uint32_t> nbins("nbins", true, edm::Comment("number of bins of the plot"));
  edm::ParameterDescription<double> min("min", true, edm::Comment("starting value of the x axis"));
  edm::ParameterDescription<double> max("max", true, edm::Comment("ending value of the x axis"));
  edm::ParameterDescription<bool> bitset("bitset", false, true, edm::Comment("plot individual bits of values"));
  edm::ParameterDescription<std::string> column(
      "column", true, edm::Comment("name of the raw to fill the content of the plot"));
  edm::ParameterDescription<std::string> xcolumn(
      "xcolumn", true, edm::Comment("name of the raw to fill the x content of the plot"));
  edm::ParameterDescription<std::string> ycolumn(
      "ycolumn", true, edm::Comment("name of the raw to fill the y content of the plot"));

  edm::ParameterSetDescription plot;
  plot.setComment("a parameter set that defines a DQM histogram");
  plot.ifValue(
      edm::ParameterDescription<std::string>("kind", "none", true, edm::Comment("the type of histogram")),
      "none" >> (name) or  //it should really be edm::EmptyGroupDescription(), but name is used in python by modifiers
          "count1d" >> (name and title and nbins and min and max) or
          "hist1d" >> (name and title and nbins and min and max and column and bitset) or
          "prof1d" >> (name and title and nbins and min and max and xcolumn and ycolumn));

  edm::ParameterSetDescription vplot;
  vplot.setComment(
      "a parameter set to define all the plots to be made from a table row selected from the name of the PSet");
  vplot.add<edm::ParameterSetDescription>("sels", sels);
  vplot.addVPSet("plots", plot);

  edm::ParameterSetDescription vplots;
  vplots.setComment("a parameter set to define all the set of plots to be made from the tables");
  vplots.addNode(edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, true, vplot));
  desc.add<edm::ParameterSetDescription>("vplots", vplots);

  descriptions.addWithDefaultLabel(desc);
}

void NanoAODDQM::bookHistograms(DQMStore::IBooker &booker, edm::Run const &, edm::EventSetup const &) {
  booker.setCurrentFolder("Physics/NanoAODDQM");

  for (auto &pair : groups_) {
    booker.setCurrentFolder("Physics/NanoAODDQM/" + pair.first);
    for (auto &sels : pair.second.selGroups) {
      std::string dir("Physics/NanoAODDQM/" + pair.first);
      if (!sels.nullCut())
        dir += "/" + sels.name;
      booker.setCurrentFolder(dir);
      auto &plots = sels.plots;
      plots.clear();
      plots.reserve(pair.second.plotPSets.size());
      for (const auto &cfg : pair.second.plotPSets) {
        auto plot = makePlot(booker, cfg);
        if (plot)
          plots.push_back(std::move(plot));
      }
    }
  }
}

void NanoAODDQM::analyze(const edm::Event &iEvent, const edm::EventSetup &) {
  std::vector<edm::Handle<FlatTable>> alltables;
  getterOfProducts_.fillHandles(iEvent, alltables);
  std::map<std::string, std::pair<const FlatTable *, std::vector<const FlatTable *>>> maintables;

  for (const auto &htab : alltables) {
    if (htab->extension())
      continue;
    maintables[htab->name()] = std::make_pair(htab.product(), std::vector<const FlatTable *>());
  }
  for (const auto &htab : alltables) {
    if (htab->extension()) {
      if (maintables.find(htab->name()) == maintables.end())
        throw cms::Exception("LogicError", "Missing main table for " + htab->name());
      maintables[htab->name()].second.push_back(htab.product());
    }
  }

  FlatTable merged;
  for (auto &pair : groups_) {
    const std::string &name = pair.first;
    if (maintables.find(name) == maintables.end())
      continue;  // may happen for missing collections
    auto &tables = maintables[name];
    const FlatTable *table = tables.first;
    if (!tables.second.empty()) {
      merged = *tables.first;
      for (auto *other : tables.second) {
        merged.addExtension(*other);
      }
      table = &merged;
    }
    std::vector<bool> selbits;
    for (auto &sel : pair.second.selGroups) {
      sel.fillSel(*table, selbits);

      for (auto &plot : sel.plots) {
        plot->fill(*table, selbits);
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(NanoAODDQM);
