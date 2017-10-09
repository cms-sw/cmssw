/** \file EDMtoMEConverter.cc
 *
 *  See header file for description of class
 *
 *  \author M. Strang SUNY-Buffalo
 */

#include <cassert>

#include "DQMServices/Components/plugins/EDMtoMEConverter.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

using namespace lat;

template<typename T>
void EDMtoMEConverter::Tokens<T>::set(const edm::InputTag& runInputTag, const edm::InputTag& lumiInputTag, edm::ConsumesCollector& iC) {
  runToken = iC.mayConsume<MEtoEDM<T>, edm::InRun>(runInputTag);
  lumiToken = iC.mayConsume<MEtoEDM<T>, edm::InLumi>(lumiInputTag);
}

template<typename T>
void EDMtoMEConverter::Tokens<T>::getData(const edm::Run& iRun, edm::Handle<Product>& handle) const {
  iRun.getByToken(runToken, handle);
}

template<typename T>
void EDMtoMEConverter::Tokens<T>::getData(const edm::LuminosityBlock& iLumi, edm::Handle<Product>& handle) const {
  iLumi.getByToken(lumiToken, handle);
}

namespace {
  // general
  template <size_t I, size_t N>
  struct ForEachHelper {
    template <typename Tuple, typename Func>
    static
    void call(Tuple&& tpl, Func&& func) {
      func(std::get<I-1>(tpl));
      ForEachHelper<I+1, N>::call(std::forward<Tuple>(tpl), std::forward<Func>(func));
    }
  };
  // break recursion
  template <size_t N>
  struct ForEachHelper<N, N> {
    template <typename Tuple, typename Func>
    static
    void call(Tuple&& tpl, Func&& func) {
      func(std::get<N-1>(tpl));
    }
  };

  // helper function to provide nice interface
  template <typename Tuple, typename Func>
  void for_each(Tuple&& tpl, Func&& func) {
    constexpr auto size = std::tuple_size<typename std::decay<Tuple>::type>::value;
    ForEachHelper<1, size>::call(std::forward<Tuple>(tpl), std::forward<Func>(func));
  }


  template <typename T> struct HistoTraits;
  template <> struct HistoTraits<TH1F> {
    static TH1F *get(MonitorElement *me) { return me->getTH1F(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book1D(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH1S> {
    static TH1S *get(MonitorElement *me) { return me->getTH1S(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book1S(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH1D> {
    static TH1D *get(MonitorElement *me) { return me->getTH1D(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book1DD(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH2F> {
    static TH2F *get(MonitorElement *me) { return me->getTH2F(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book2D(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH2S> {
    static TH2S *get(MonitorElement *me) { return me->getTH2S(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book2S(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH2D> {
    static TH2D *get(MonitorElement *me) { return me->getTH2D(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book2DD(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TH3F> {
    static TH3F *get(MonitorElement *me) { return me->getTH3F(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->book3D(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TProfile>{
    static TProfile *get(MonitorElement *me) { return me->getTProfile(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->bookProfile(std::forward<Args>(args)...);
    }
  };
  template <> struct HistoTraits<TProfile2D> {
    static TProfile2D *get(MonitorElement *me) { return me->getTProfile2D(); }
    template <typename ...Args> static MonitorElement *book(DQMStore *dbe, Args&&... args) {
      return dbe->bookProfile2D(std::forward<Args>(args)...);
    }
  };
  
  // Default to histograms and similar, others are specialized
  template <typename T>
  struct AddMonitorElement {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static
    MonitorElement *call(DQMStore *dbe, MEtoEDMObject_object *metoedmobject, const std::string& dir, const std::string& name, const RunOrLumi& runOrLumi) {
      MonitorElement *me = dbe->get(dir+"/"+metoedmobject->GetName());
      if(me) {
        auto histo = HistoTraits<T>::get(me);
        if(histo && me->getTH1()->CanExtendAllAxes()) {
          TList list;
          list.Add(metoedmobject);
          if (histo->Merge(&list) == -1)
            std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                      << metoedmobject->GetName() << "'" <<  std::endl;
          return me;
        }
      }

      dbe->setCurrentFolder(dir);
      return HistoTraits<T>::book(dbe, metoedmobject->GetName(), metoedmobject);
    }
  };

  template <>
  struct AddMonitorElement<double> {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static
    MonitorElement *call(DQMStore *dbe, MEtoEDMObject_object *metoedmobject, const std::string& dir, const std::string& name, const RunOrLumi& runOrLumi) {
      dbe->setCurrentFolder(dir);
      MonitorElement *me = dbe->bookFloat(name);
      me->Fill(*metoedmobject);
      return me;
    }
  };

  // long long and int share some commonalities, which are captured here (we can have only one default template definition)
  template <typename T>
  struct AddMonitorElementForIntegers {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static
    MonitorElement *call(DQMStore *dbe, MEtoEDMObject_object *metoedmobject, const std::string& dir, const std::string& name, const RunOrLumi& runOrLumi) {
      dbe->setCurrentFolder(dir);
      T ival = getProcessedEvents(dbe, dir, name, runOrLumi);
      MonitorElement *me = dbe->bookInt(name);
      me->Fill(*metoedmobject + ival);
      return me;
    }

    static
    T getProcessedEvents(const DQMStore *dbe, const std::string& dir, const std::string& name, const edm::Run&) {
      if(name.find("processedEvents") != std::string::npos) {
        if(const MonitorElement *me = dbe->get(dir+"/"+name)) {
          return me->getIntValue();
        }
      }
      return 0;
    }

    static
    T getProcessedEvents(const DQMStore *dbe, const std::string& dir, const std::string& name, const edm::LuminosityBlock&) {
      return 0;
    }
  };
  template <>
  struct AddMonitorElement<long long> {
    template <typename ...Args>
    static
    MonitorElement *call(Args&&... args) {
      return AddMonitorElementForIntegers<long long>::call(std::forward<Args>(args)...);
    }
  };
  template <>
  struct AddMonitorElement<int> {
    template <typename ...Args>
    static
    MonitorElement *call(Args&&... args) {
      return AddMonitorElementForIntegers<int>::call(std::forward<Args>(args)...);
    }
  };


  template <>
  struct AddMonitorElement<TString> {
    template <typename MEtoEDMObject_object, typename RunOrLumi>
    static
    MonitorElement *call(DQMStore *dbe, MEtoEDMObject_object *metoedmobject, const std::string& dir, const std::string& name, const RunOrLumi& runOrLumi) {
      dbe->setCurrentFolder(dir);
      std::string scont = metoedmobject->Data();
      return dbe->bookString(name, scont);
    }
  };

  void maybeSetLumiFlag(MonitorElement *me, const edm::Run&) {
  }
  void maybeSetLumiFlag(MonitorElement *me, const edm::LuminosityBlock&) {
    me->setLumiFlag();
  }

}

EDMtoMEConverter::EDMtoMEConverter(const edm::ParameterSet & iPSet) :
  verbosity(0), frequency(0)
{
  const edm::InputTag& runInputTag = iPSet.getParameter<edm::InputTag>("runInputTag");
  const edm::InputTag& lumiInputTag = iPSet.getParameter<edm::InputTag>("lumiInputTag");
  edm::ConsumesCollector iC = consumesCollector();

  for_each(tokens_, [&](auto& tok) {
      tok.set(runInputTag, lumiInputTag, iC);
    });

  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_EDMtoMEConverter";

  // get information from parameter set
  name = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");

  convertOnEndLumi = iPSet.getUntrackedParameter<bool>("convertOnEndLumi",true);
  convertOnEndRun = iPSet.getUntrackedParameter<bool>("convertOnEndRun",true);

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat)
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << name << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }

  iCountf = 0;
  iCount.clear();

  assert(sizeof(int64_t) == sizeof(long long));

} // end constructor

EDMtoMEConverter::~EDMtoMEConverter() {}

void EDMtoMEConverter::beginJob()
{
}

void EDMtoMEConverter::endJob()
{
  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat)
      << "Terminating having processed " << iCount.size() << " runs across "
      << iCountf << " files.";
  return;
}

void EDMtoMEConverter::respondToOpenInputFile(const edm::FileBlock& iFb)
{
  ++iCountf;
  return;
}

void EDMtoMEConverter::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_beginRun";

  int nrun = iRun.run();

  // keep track of number of unique runs processed
  ++iCount[nrun];

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << iCount.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || iCount.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
        << "Processing run " << nrun << " (" << iCount.size() << " runs total)";
    }
  }
}

void EDMtoMEConverter::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if (convertOnEndRun) {
    getData(iRun);
  }
}

void EDMtoMEConverter::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup)
{
}

void EDMtoMEConverter::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup)
{
  if (convertOnEndLumi) {
    getData(iLumi);
  }
}

void EDMtoMEConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

template <class T>
void
EDMtoMEConverter::getData(T& iGetFrom)
{
  constexpr char MsgLoggerCat[] = "EDMtoMEConverter_getData";

  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat) << "\nRestoring MonitorElements.";

  for_each(tokens_, [&](const auto& tok) {
      using Tokens_T = typename std::decay<decltype(tok)>::type;
      using METype = typename Tokens_T::type;
      using MEtoEDM_T = typename Tokens_T::Product;
      edm::Handle<MEtoEDM_T> metoedm;
      tok.getData(iGetFrom, metoedm);
      if(!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH1F> doesn't exist in run";
        return;
      }

      std::vector<typename MEtoEDM_T::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
        // get full path of monitor element
        const std::string& pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        std::string name = *(fulldir.end() - 1);

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        MonitorElement *me = AddMonitorElement<METype>::call(dbe, &metoedmobject[i].object, dir, name, iGetFrom);
        maybeSetLumiFlag(me, iGetFrom);

        // attach taglist
        for(const auto& tag: metoedmobject[i].tags) {
          dbe->tag(me->getFullname(), tag);
        }
      } // end loop thorugh metoedmobject
    });

  // verify tags stored properly
  if (verbosity > 0) {
    std::vector<std::string> stags;
    dbe->getAllTags(stags);
    for (unsigned int i = 0; i < stags.size(); ++i) {
      std::cout << "Tags: " << stags[i] << std::endl;
    }
  }
}

