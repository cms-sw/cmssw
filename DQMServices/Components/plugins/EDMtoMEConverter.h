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
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// data format
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"

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

class EDMtoMEConverter : public edm::EDAnalyzer
{

 public:

  explicit EDMtoMEConverter(const edm::ParameterSet&);
  ~EDMtoMEConverter() override;
  void beginJob() override;
  void endJob() override;  
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void respondToOpenInputFile(const edm::FileBlock&) override;

  template <class T>
  void getData(T& iGetFrom);

  typedef std::vector<uint32_t> TagList;

 private:
  
  std::string name;
  int verbosity;
  int frequency;

  bool convertOnEndLumi;
  bool convertOnEndRun;

  DQMStore *dbe;

  // private statistics information
  unsigned int iCountf;
  std::map<int,int> iCount;

  template <typename T>
  class Tokens {
  public:
    using type = T;
    using Product = MEtoEDM<T>;

    Tokens() {}

    void set(const edm::InputTag& runInputTag, const edm::InputTag& lumiInputTag, edm::ConsumesCollector& iC);

    void getData(const edm::Run& iRun, edm::Handle<Product>& handle) const;
    void getData(const edm::LuminosityBlock& iLumi, edm::Handle<Product>& handle) const;
    
  private:
    edm::EDGetTokenT<Product> runToken;
    edm::EDGetTokenT<Product> lumiToken;
  };

  std::tuple<
    Tokens<TH1F>,
    Tokens<TH1S>,
    Tokens<TH1D>,
    Tokens<TH2F>,
    Tokens<TH2S>,
    Tokens<TH2D>,
    Tokens<TH3F>,
    Tokens<TProfile>,
    Tokens<TProfile2D>,
    Tokens<double>,
    Tokens<int>,
    Tokens<long long>,
    Tokens<TString>
    > tokens_;
}; // end class declaration

#endif
