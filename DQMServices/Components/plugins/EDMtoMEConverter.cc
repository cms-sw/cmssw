/** \class EDMtoMEConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Histograms/interface/DQMToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// data format
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

// helper files
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <tuple>

#include "TString.h"
#include "TList.h"

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"

class EDMtoMEConverter : public edm::one::EDProducer<edm::one::WatchRuns,
                                                     edm::one::WatchLuminosityBlocks,
                                                     edm::one::SharedResources,
                                                     edm::EndLuminosityBlockProducer,
                                                     edm::EndRunProducer> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit EDMtoMEConverter(const edm::ParameterSet &);
  ~EDMtoMEConverter() override = default;

  void beginRun(const edm::Run &, const edm::EventSetup &) final{};
  void endRun(const edm::Run &, const edm::EventSetup &) final{};
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) final{};
  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) final{};
  void produce(edm::Event &, edm::EventSetup const &) final{};

  void endLuminosityBlockProduce(edm::LuminosityBlock &, edm::EventSetup const &) override;
  void endRunProduce(edm::Run &run, edm::EventSetup const &setup) override;

  template <class T>
  void getData(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, T &iGetFrom);

  using TagList = std::vector<uint32_t>;

private:
  std::string name;
  int verbosity;
  int frequency;

  bool convertOnEndLumi;
  bool convertOnEndRun;
  MonitorElementData::Scope reScope;

  template <typename T>
  class Tokens {
  public:
    using type = T;
    using Product = MEtoEDM<T>;

    Tokens() = default;

    void set(const edm::InputTag &runInputTag, const edm::InputTag &lumiInputTag, edm::ConsumesCollector &iC);

    void getData(const edm::Run &iRun, edm::Handle<Product> &handle) const;
    void getData(const edm::LuminosityBlock &iLumi, edm::Handle<Product> &handle) const;

  private:
    edm::EDGetTokenT<Product> runToken;
    edm::EDGetTokenT<Product> lumiToken;
  };

  std::tuple<Tokens<TH1F>,
             Tokens<TH1S>,
             Tokens<TH1D>,
             Tokens<TH1I>,
             Tokens<TH2F>,
             Tokens<TH2S>,
             Tokens<TH2D>,
             Tokens<TH2I>,
             Tokens<TH3F>,
             Tokens<TProfile>,
             Tokens<TProfile2D>,
             Tokens<double>,
             Tokens<int>,
             Tokens<long long>,
             Tokens<TString> >
      tokens_;

  edm::EDPutTokenT<DQMToken> dqmLumiToken_;
  edm::EDPutTokenT<DQMToken> dqmRunToken_;
};  // end class declaration

using namespace lat;
using dqm::legacy::DQMStore;
using dqm::legacy::MonitorElement;

template <typename T>
void EDMtoMEConverter::Tokens<T>::set(const edm::InputTag &runInputTag,
                                      const edm::InputTag &lumiInputTag,
                                      edm::ConsumesCollector &iC) {
  runToken = iC.mayConsume<MEtoEDM<T>, edm::InRun>(runInputTag);
  lumiToken = iC.mayConsume<MEtoEDM<T>, edm::InLumi>(lumiInputTag);
}

template <typename T>
void EDMtoMEConverter::Tokens<T>::getData(const edm::Run &iRun, edm::Handle<Product> &handle) const {
  iRun.getByToken(runToken, handle);
}

template <typename T>
void EDMtoMEConverter::Tokens<T>::getData(const edm::LuminosityBlock &iLumi, edm::Handle<Product> &handle) const {
  iLumi.getByToken(lumiToken, handle);
}

namespace {
  // general
  template <size_t I, size_t N>
  struct ForEachHelper {
    template <typename Tuple, typename Func>
    static void call(Tuple &&tpl, Func &&func) {
      func(std::get<I - 1>(tpl));
      ForEachHelper<I + 1, N>::call(std::forward<Tuple>(tpl), std::forward<Func>(func));
    }
  };
  // break recursion
  template <size_t N>
  struct ForEachHelper<N, N> {
    template <typename Tuple, typename Func>
    static void call(Tuple &&tpl, Func &&func) {
      func(std::get<N - 1>(tpl));
    }
  };

  // helper function to provide nice interface
  template <typename Tuple, typename Func>
  void for_each(Tuple &&tpl, Func &&func) {
    constexpr auto size = std::tuple_size<typename std::decay<Tuple>::type>::value;
    ForEachHelper<1, size>::call(std::forward<Tuple>(tpl), std::forward<Func>(func));
  }

  template <typename T>
  struct HistoTraits;
  template <>
  struct HistoTraits<TH1F> {
    static TH1F *get(MonitorElement *me) { return me->getTH1F(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book1D(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH1S> {
    static TH1S *get(MonitorElement *me) { return me->getTH1S(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book1S(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH1D> {
    static TH1D *get(MonitorElement *me) { return me->getTH1D(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book1DD(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH1I> {
    static TH1I *get(MonitorElement *me) { return me->getTH1I(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book1I(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH2F> {
    static TH2F *get(MonitorElement *me) { return me->getTH2F(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book2D(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH2S> {
    static TH2S *get(MonitorElement *me) { return me->getTH2S(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book2S(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH2D> {
    static TH2D *get(MonitorElement *me) { return me->getTH2D(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book2DD(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH2I> {
    static TH2I *get(MonitorElement *me) { return me->getTH2I(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book2I(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TH3F> {
    static TH3F *get(MonitorElement *me) { return me->getTH3F(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.book3D(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TProfile> {
    static TProfile *get(MonitorElement *me) { return me->getTProfile(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.bookProfile(std::forward<Args>(args)...);
    }
  };
  template <>
  struct HistoTraits<TProfile2D> {
    static TProfile2D *get(MonitorElement *me) { return me->getTProfile2D(); }
    template <typename... Args>
    static MonitorElement *book(DQMStore::IBooker &iBooker, Args &&...args) {
      return iBooker.bookProfile2D(std::forward<Args>(args)...);
    }
  };

  // Default to histograms and similar, others are specialized
  template <typename T>
  struct AddMonitorElement {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static MonitorElement *call(DQMStore::IBooker &iBooker,
                                DQMStore::IGetter &iGetter,
                                MEtoEDMObject_object *metoedmobject,
                                const std::string &dir,
                                const std::string &name,
                                const RunOrLumi &runOrLumi) {
      MonitorElement *me = iGetter.get(dir + "/" + metoedmobject->GetName());

      if (me) {
        auto histo = HistoTraits<T>::get(me);
        assert(histo);
        TList list;
        list.Add(metoedmobject);
        if (histo->Merge(&list) == -1)
          edm::LogError("EDMtoMEConverter") << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                                            << metoedmobject->GetName() << "'" << std::endl;
        return me;
      } else {
        iBooker.setCurrentFolder(dir);
        return HistoTraits<T>::book(iBooker, metoedmobject->GetName(), metoedmobject);
      }
    }
  };

  template <>
  struct AddMonitorElement<double> {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static MonitorElement *call(DQMStore::IBooker &iBooker,
                                DQMStore::IGetter &iGetter,
                                MEtoEDMObject_object *metoedmobject,
                                const std::string &dir,
                                const std::string &name,
                                const RunOrLumi &runOrLumi) {
      iBooker.setCurrentFolder(dir);
      MonitorElement *me = iBooker.bookFloat(name);
      me->Fill(*metoedmobject);
      return me;
    }
  };

  // long long and int share some commonalities, which are captured here (we can have only one default template definition)
  template <typename T>
  struct AddMonitorElementForIntegers {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static MonitorElement *call(DQMStore::IBooker &iBooker,
                                DQMStore::IGetter &iGetter,
                                MEtoEDMObject_object *metoedmobject,
                                const std::string &dir,
                                const std::string &name,
                                const RunOrLumi &runOrLumi) {
      iBooker.setCurrentFolder(dir);
      iGetter.setCurrentFolder(dir);
      T ival = getProcessedEvents(iGetter, dir, name, runOrLumi);
      MonitorElement *me = iBooker.bookInt(name);
      me->Fill(*metoedmobject + ival);
      return me;
    }

    static T getProcessedEvents(DQMStore::IGetter &iGetter,
                                const std::string &dir,
                                const std::string &name,
                                const edm::Run &) {
      if (name.find("processedEvents") != std::string::npos) {
        if (const MonitorElement *me = iGetter.get(dir + "/" + name)) {
          return me->getIntValue();
        }
      }
      return 0;
    }

    static T getProcessedEvents(DQMStore::IGetter &iGetter,
                                const std::string &dir,
                                const std::string &name,
                                const edm::LuminosityBlock &) {
      return 0;
    }
  };
  template <>
  struct AddMonitorElement<long long> {
    template <typename... Args>
    static MonitorElement *call(Args &&...args) {
      return AddMonitorElementForIntegers<long long>::call(std::forward<Args>(args)...);
    }
  };
  template <>
  struct AddMonitorElement<int> {
    template <typename... Args>
    static MonitorElement *call(Args &&...args) {
      return AddMonitorElementForIntegers<int>::call(std::forward<Args>(args)...);
    }
  };

  template <>
  struct AddMonitorElement<TString> {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static MonitorElement *call(DQMStore::IBooker &iBooker,
                                DQMStore::IGetter &iGetter,
                                MEtoEDMObject_object *metoedmobject,
                                const std::string &dir,
                                const std::string &name,
                                const RunOrLumi &runOrLumi) {
      iBooker.setCurrentFolder(dir);
      std::string scont = metoedmobject->Data();
      return iBooker.bookString(name, scont);
    }
  };

  // TODO: might need re-scoping to JOB here.
  void adjustScope(DQMStore::IBooker &ibooker, const edm::Run &, MonitorElementData::Scope reScope) {
    if (reScope == MonitorElementData::Scope::JOB) {
      ibooker.setScope(MonitorElementData::Scope::JOB);
    } else {
      ibooker.setScope(MonitorElementData::Scope::RUN);
    }
  }
  void adjustScope(DQMStore::IBooker &ibooker, const edm::LuminosityBlock &, MonitorElementData::Scope reScope) {
    // will be LUMI for no reScoping, else the expected scope.
    ibooker.setScope(reScope);
  }

}  // namespace

EDMtoMEConverter::EDMtoMEConverter(const edm::ParameterSet &iPSet) : verbosity(0), frequency(0) {
  const edm::InputTag &runInputTag = iPSet.getParameter<edm::InputTag>("runInputTag");
  const edm::InputTag &lumiInputTag = iPSet.getParameter<edm::InputTag>("lumiInputTag");
  edm::ConsumesCollector iC = consumesCollector();

  for_each(tokens_, [&](auto &tok) { tok.set(runInputTag, lumiInputTag, iC); });

  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_EDMtoMEConverter";

  // get information from parameter set
  name = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");

  convertOnEndLumi = iPSet.getUntrackedParameter<bool>("convertOnEndLumi", true);
  convertOnEndRun = iPSet.getUntrackedParameter<bool>("convertOnEndRun", true);

  auto scopeDecode = std::map<std::string, MonitorElementData::Scope>{{"", MonitorElementData::Scope::LUMI},
                                                                      {"LUMI", MonitorElementData::Scope::LUMI},
                                                                      {"RUN", MonitorElementData::Scope::RUN},
                                                                      {"JOB", MonitorElementData::Scope::JOB}};
  reScope = scopeDecode[iPSet.getUntrackedParameter<std::string>("reScope", "")];

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) << "\n===============================\n"
                               << "Initialized as EDAnalyzer with parameter values:\n"
                               << "    Name          = " << name << "\n"
                               << "    Verbosity     = " << verbosity << "\n"
                               << "    Frequency     = " << frequency << "\n"
                               << "===============================\n";
  }

  assert(sizeof(int64_t) == sizeof(long long));
  usesResource("DQMStore");

  dqmLumiToken_ = produces<DQMToken, edm::Transition::EndLuminosityBlock>("endLumi");
  dqmRunToken_ = produces<DQMToken, edm::Transition::EndRun>("endRun");
}  // end constructor

void EDMtoMEConverter::endRunProduce(edm::Run &iRun, edm::EventSetup const &iSetup) {
  if (convertOnEndRun) {
    DQMStore *store = edm::Service<DQMStore>().operator->();
    store->meBookerGetter([&](DQMStore::IBooker &b, DQMStore::IGetter &g) { getData(b, g, iRun); });
  }

  iRun.put(dqmRunToken_, std::make_unique<DQMToken>());
}

void EDMtoMEConverter::endLuminosityBlockProduce(edm::LuminosityBlock &iLumi, edm::EventSetup const &iSetup) {
  if (convertOnEndLumi) {
    DQMStore *store = edm::Service<DQMStore>().operator->();
    store->meBookerGetter([&](DQMStore::IBooker &b, DQMStore::IGetter &g) { getData(b, g, iLumi); });
  }

  iLumi.put(dqmLumiToken_, std::make_unique<DQMToken>());
}

template <class T>
void EDMtoMEConverter::getData(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, T &iGetFrom) {
  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_getData";

  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) << "\nRestoring MonitorElements.";

  for_each(tokens_, [&](const auto &tok) {
    using Tokens_T = typename std::decay<decltype(tok)>::type;
    using METype = typename Tokens_T::type;
    using MEtoEDM_T = typename Tokens_T::Product;
    edm::Handle<MEtoEDM_T> metoedm;
    tok.getData(iGetFrom, metoedm);
    if (!metoedm.isValid()) {
      //edm::LogWarning(MsgLoggerCat)
      //  << "MEtoEDM<TH1F> doesn't exist in run";
      return;
    }

    std::vector<typename MEtoEDM_T::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject();

    for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
      // get full path of monitor element
      const std::string &pathname = metoedmobject[i].name;
      if (verbosity > 0)
        std::cout << pathname << std::endl;

      std::string dir;

      // deconstruct path from fullpath
      StringList fulldir = StringOps::split(pathname, "/");
      std::string name = *(fulldir.end() - 1);

      for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
        dir += fulldir[j];
        if (j != fulldir.size() - 2)
          dir += "/";
      }

      // define new monitor element
      adjustScope(iBooker, iGetFrom, reScope);
      AddMonitorElement<METype>::call(iBooker, iGetter, &metoedmobject[i].object, dir, name, iGetFrom);

    }  // end loop thorugh metoedmobject
  });
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EDMtoMEConverter);
