#ifndef MEtoEDMConverter_h
#define MEtoEDMConverter_h

/** \class MEtoEDMConverter
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// helper files
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <cassert>
#include <cstdint>

#include "TString.h"
#include "TH1F.h"
#include "TH1S.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TH2S.h"
#include "TH2D.h"
#include "TH2I.h"
#include "TH3F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjString.h"

namespace meedm {
  struct Void {};
}  // namespace meedm

//Using RunCache and LuminosityBlockCache tells the framework the module is able to
// allow multiple concurrent Runs and LuminosityBlocks.

class MEtoEDMConverter : public edm::one::EDProducer<edm::RunCache<meedm::Void>,
                                                     edm::LuminosityBlockCache<meedm::Void>,
                                                     edm::EndLuminosityBlockProducer,
                                                     edm::EndRunProducer,
                                                     edm::one::SharedResources> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit MEtoEDMConverter(const edm::ParameterSet&);
  ~MEtoEDMConverter() override;
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  std::shared_ptr<meedm::Void> globalBeginRun(edm::Run const&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, const edm::EventSetup&) override;
  void endRunProduce(edm::Run&, const edm::EventSetup&) override;
  void endLuminosityBlockProduce(edm::LuminosityBlock&, const edm::EventSetup&) override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override{};
  std::shared_ptr<meedm::Void> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                          edm::EventSetup const&) const override;

  template <class T>
  void putData(DQMStore::IGetter& g, T& iPutTo, bool iLumiOnly, uint32_t run, uint32_t lumi);

  using TagList = std::vector<uint32_t>;

private:
  std::string fName;
  int verbosity;
  int frequency;
  std::string path;

  // private statistics information
  std::map<int, int> iCount;

};  // end class declaration

#endif
