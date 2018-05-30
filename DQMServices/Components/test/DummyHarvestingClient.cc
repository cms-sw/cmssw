// This is a simple skeleton of a typical harvesting client. Its only
// purpose is to produce easy to interpret results in order to quickly
// x-check them and validate all the possible scenarios in which we
// can book/reset/collate histograms.
//
// LS-based histograms' harvesting
//
// It uses few helper classes, CumulatorBase and its derived
// TH1FCumulator and TH2FCumulator, to perform proper tasks on the
// LS-based histograms. In particular we create one instance of these
// classes, matching the type of the object we have to handle, for
// each LS-based histograms we would like to 'harvest'; each instance
// will extract the associated-by-name LS-based histograms at every
// endLuminosity transition and internally stores the histogram's
// entries versus the current LS number in a local map. This job is
// internally performed in the cumulate method. At the endRun the
// method finalizeCumulate is called for every instance: here we book
// a new histogram (remember, one for each instance!) that will store
// the number of entries vs LS number of all the processed LS,
// basically translating the internal map filled by the 'cumulate()'
// calls into a TH1F.
//
// Run-based histograms' harvesting
//
// So far nothing is done on top of the run-based histograms: we
// simply want to be sure that run-based histograms are correctly
// saved in the output files and properly merged in case we run this
// client with many sources in input.
//
// Configuration
//
// The client is completely configurable in terms of histograms to
// harvest using python PSets. In particular the 'cumulateLumis' PSet
// will controll which LS-based histogram we have to 'cumulate()' and
// 'finalizeCumulate()'

// system include files
#include <memory>
#include <utility>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

# define DEBUG 0
//
// class declaration
//
namespace {
class CumulatorBase {
 public:
  virtual ~CumulatorBase()  = default;
  virtual void cumulate(int lumi_section) = 0;
  virtual void finalizeCumulate() = 0;
};

class TH1FCumulator : public CumulatorBase {
 public:
  TH1FCumulator(const edm::ParameterSet& iPSet,
                DQMStore& iStore,
                std::string folder,
                bool iSetLumiFlag)
      :store_(&iStore),
       folder_(std::move(std::move(folder))) {
    std::string extension;
    if (iSetLumiFlag) {
      extension = "_lumi";
    }
    name_ = iPSet.getUntrackedParameter<std::string>("name") + extension;
  }

  ~TH1FCumulator() override = default;

  void cumulate(int ls) override {
    MonitorElement *tmp = nullptr;
    if (!(tmp = store_->get(folder_ + name_)))
      throw cms::Exception("MissingHistogram") << name_ << std::endl;

    entries_per_LS_[ls] = tmp->getTH1F()->GetEntries();
    if (DEBUG)
      std::cout << "Getting from " << folder_ << name_
                << " entries: " << entries_per_LS_[ls]
                << " in LS: " << ls << std::endl;
  }

  void finalizeCumulate() override {
    if (DEBUG)
      std::cout << "TH1FCumulator::finalizaCumulate()" << std::endl;

    auto it = entries_per_LS_.begin();
    auto ite = entries_per_LS_.end();
    std::string extension("_cumulative");
    store_->setCurrentFolder(folder_);
    MonitorElement *tmp = store_->book1D(name_ + extension,
                                         name_ + extension,
                                         (--(entries_per_LS_.end()))->first,
                                         0,
                                         (--(entries_per_LS_.end()))->first);

    int lastAccessed = 0;
    for (; it != ite; it++) {
      while (lastAccessed < (*it).first) {
        tmp->Fill(lastAccessed, -1.);
        lastAccessed++;
      }
      if (DEBUG)
        std::cout << "TH1FCumulator::finalizaCumulate() fill "
                  << (*it).first << " " << (*it).second<< std::endl;
      tmp->Fill((*it).first, (*it).second);
      lastAccessed = (*it).first + 1;
    }
  }

 private:
  DQMStore* store_;
  std::string folder_;
  std::string name_;
  std::map<int, int> entries_per_LS_;
};

class TH2FCumulator : public CumulatorBase {
 public:
  TH2FCumulator(const edm::ParameterSet& iPSet,
                DQMStore& iStore,
                std::string folder,
                bool iSetLumiFlag)
      :store_(&iStore),
       folder_(std::move(std::move(folder))) {
    std::string extension;
    if (iSetLumiFlag) {
      extension = "_lumi";
    }
    name_ = iPSet.getUntrackedParameter<std::string>("name")+extension;
  }

  ~TH2FCumulator() override = default;

  void cumulate(int ls) override {
    MonitorElement *tmp = nullptr;
    if (!(tmp = store_->get(folder_ + name_)))
      throw cms::Exception("MissingHistogram") << name_ << std::endl;

    entries_per_LS_[ls] = tmp->getTH2F()->GetEntries();
  };

  void finalizeCumulate() override {
    auto it = entries_per_LS_.begin();
    auto ite = entries_per_LS_.end();
    std::string extension("_cumulative");
    store_->setCurrentFolder(folder_);
    MonitorElement *tmp = store_->book1D(name_ + extension,
                                         name_ + extension,
                                         (--(entries_per_LS_.end()))->first,
                                         0,
                                         (--(entries_per_LS_.end()))->first);

    int lastAccessed = 0;
    for (; it != ite; it++) {
      while (lastAccessed < (*it).first) {
        tmp->Fill(lastAccessed, -1.);
        lastAccessed++;
      }
      tmp->Fill((*it).first, (*it).second);
      lastAccessed = (*it).first + 1;
    }
  };

 private:
  DQMStore* store_;
  std::string folder_;
  std::string name_;
  std::map<int, int> entries_per_LS_;
};
}

class DummyHarvestingClient : public edm::EDAnalyzer {
 public:
  using PSets = std::vector<edm::ParameterSet>;
  explicit DummyHarvestingClient(const edm::ParameterSet&);
  ~DummyHarvestingClient() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&,
                                  edm::EventSetup const&) override;

  void bookHistograms();
  // ----------member data ---------------------------
  std::vector<boost::shared_ptr<CumulatorBase> > m_lumiCumulators;
  bool m_cumulateRuns;
  bool m_cumulateLumis;
  bool book_at_constructor_;
  bool book_at_beginJob_;
  bool book_at_beginRun_;
  PSets elements_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
DummyHarvestingClient::DummyHarvestingClient(const edm::ParameterSet& iConfig)
    : m_cumulateRuns(iConfig.getUntrackedParameter<bool>("cumulateRuns")),
      m_cumulateLumis(iConfig.getUntrackedParameter<bool>("cumulateLumis")),
      book_at_constructor_(iConfig.getUntrackedParameter<bool>("book_at_constructor", false)),
      book_at_beginJob_(iConfig.getUntrackedParameter<bool>("book_at_beginJob", false)),
      book_at_beginRun_(iConfig.getUntrackedParameter<bool>("book_at_beginRun", false)),
      elements_(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("elements")) {
  edm::Service<DQMStore> dstore;
  // TODO(rovere): assert on multiple book conditions
  if (book_at_constructor_)
    bookHistograms();

  std::string folder = iConfig.getUntrackedParameter<std::string>("folder", "/TestFolder/");
  if (m_cumulateLumis) {
    m_lumiCumulators.reserve(elements_.size());
    auto it = elements_.begin();
    auto ite = elements_.end();
    for (; it != ite; ++it) {
      switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
        case 1:
          m_lumiCumulators.push_back(boost::shared_ptr<CumulatorBase>(
              new TH1FCumulator(*it, *dstore, folder, true)));
          break;
        case 2:
          m_lumiCumulators.push_back(boost::shared_ptr<CumulatorBase>(
              new TH2FCumulator(*it, *dstore, folder, true)));
          break;
      }
    }
  }
}

void DummyHarvestingClient::bookHistograms() {
}


DummyHarvestingClient::~DummyHarvestingClient() = default;


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyHarvestingClient::analyze(edm::Event const& iEvent,
                               edm::EventSetup const& iSetup) {
  using namespace edm;
}

// ------------ method called once each job just before starting event loop  ------------
void
DummyHarvestingClient::beginJob() {
  if (book_at_beginJob_)
    bookHistograms();
}

// ------------ method called once each job just after ending the event loop  ------------
void
DummyHarvestingClient::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
DummyHarvestingClient::beginRun(edm::Run const&, edm::EventSetup const&) {
  if (book_at_beginRun_)
    bookHistograms();
}

// ------------ method called when ending the processing of a run  ------------
void
DummyHarvestingClient::endRun(edm::Run const&, edm::EventSetup const&) {
  auto it = m_lumiCumulators.begin();
  auto ite = m_lumiCumulators.end();
  for (; it != ite; ++it)
    (*it)->finalizeCumulate();
}

void
DummyHarvestingClient::endLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                          edm::EventSetup const&) {
  auto it = m_lumiCumulators.begin();
  auto ite = m_lumiCumulators.end();
  for (; it != ite; ++it)
    (*it)->cumulate(iLumi.id().luminosityBlock());
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyHarvestingClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DummyHarvestingClient);
