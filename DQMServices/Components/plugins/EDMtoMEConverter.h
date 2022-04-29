#ifndef EDMtoMEConverter_h
#define EDMtoMEConverter_h

/** \class EDMtoMEConverter
 *  
 *  Class to take dqm monitor elements and convert into a
 *  ROOT dataformat stored in Run tree of edm file
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// data format
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

// helper files
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

  explicit EDMtoMEConverter(const edm::ParameterSet&);
  ~EDMtoMEConverter() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) final{};
  void endRun(const edm::Run&, const edm::EventSetup&) final{};
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) final{};
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) final{};
  void produce(edm::Event&, edm::EventSetup const&) final{};

  void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override;
  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) override;

  template <class T>
  void getData(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, T& iGetFrom);

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

    void set(const edm::InputTag& runInputTag, const edm::InputTag& lumiInputTag, edm::ConsumesCollector& iC);

    void getData(const edm::Run& iRun, edm::Handle<Product>& handle) const;
    void getData(const edm::LuminosityBlock& iLumi, edm::Handle<Product>& handle) const;

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

#endif
