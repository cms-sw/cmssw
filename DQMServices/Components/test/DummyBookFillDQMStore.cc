// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class declaration
//
namespace {
class FillerBase {
 public:
  virtual void fill() = 0;
  virtual void reset() = 0;
};

class TH1FFiller : public FillerBase {
 public:
  TH1FFiller(const edm::ParameterSet& iPSet,
             DQMStore& iStore,
             bool iSetLumiFlag)
      : m_valueToFill(iPSet.getUntrackedParameter<double>("value")),
        valuesToFill_(iPSet.getUntrackedParameter<std::vector<double>>("values")),
        m_min(iPSet.getUntrackedParameter<double>("lowX")),
        m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
    std::string extension;
    if (iSetLumiFlag) {
      extension = "_lumi";
    }
    m_element = iStore.book1D(
        iPSet.getUntrackedParameter<std::string>("name") + extension,
        iPSet.getUntrackedParameter<std::string>("title") + extension,
        m_steps,
        m_min,
        iPSet.getUntrackedParameter<double>("highX"));

    if (iSetLumiFlag) {
      m_element->setLumiFlag();
    }
    m_hist = m_element->getTH1F();

    if (valuesToFill_.size() >0)
      assert (valuesToFill_.size() == m_steps);
  }

  virtual ~TH1FFiller() {;}

  void reset() {
    m_element->Reset();
  }

  void fill() {
    if (valuesToFill_.size() > 0) {
      for (size_t i = 0; i < valuesToFill_.size(); ++i)
        for (size_t j = 0; j < valuesToFill_[i]; ++j)
          m_hist->Fill(i);
    } else {
      m_hist->Fill(m_valueToFill);
    }
  }

 private:
  TH1F* m_hist;
  double m_valueToFill;
  std::vector<double> valuesToFill_;
  double m_min;
  unsigned int m_steps;
  MonitorElement* m_element;
};

class TH2FFiller : public FillerBase {
 public:
  TH2FFiller(const edm::ParameterSet& iPSet,
             DQMStore& iStore,
             bool iSetLumiFlag)
      : m_valueToFill(iPSet.getUntrackedParameter<double>("value")),
        valuesToFill_(iPSet.getUntrackedParameter<std::vector<double>>("values")),
        m_min(iPSet.getUntrackedParameter<double>("lowX")),
        m_steps(iPSet.getUntrackedParameter<int>("nchX")) {
    std::string extension;
    if (iSetLumiFlag) {
      extension = "_lumi";
    }
    m_element = iStore.book2D(
        iPSet.getUntrackedParameter<std::string>("name") + extension,
        iPSet.getUntrackedParameter<std::string>("title") + extension,
        m_steps,
        m_min,
        iPSet.getUntrackedParameter<double>("highX"),
        iPSet.getUntrackedParameter<int>("nchY"),
        iPSet.getUntrackedParameter<double>("lowY"),
        iPSet.getUntrackedParameter<double>("highY"));

    if (iSetLumiFlag) {
      m_element->setLumiFlag();
    }
    m_hist = m_element->getTH2F();

    if (valuesToFill_.size() >0)
      assert (valuesToFill_.size() == m_steps);
  }

  virtual ~TH2FFiller() {;}

  void reset() {
    m_element->Reset();
  }

  void fill() {
    if (valuesToFill_.size() > 0) {
      for (size_t i = 0; i < valuesToFill_.size(); ++i)
        for (size_t j = 0; j < valuesToFill_[i]; ++j)
          m_hist->Fill(i, i);
    } else {
      m_hist->Fill(m_valueToFill, m_valueToFill);
    }
  }

 private:
  TH2F* m_hist;
  double m_valueToFill;
  std::vector<double> valuesToFill_;
  double m_min;
  unsigned int m_steps;
  MonitorElement* m_element;
};
}

class DummyBookFillDQMStore :  public edm::EDAnalyzer {
 public:
  typedef std::vector<edm::ParameterSet> PSets;
  explicit DummyBookFillDQMStore(const edm::ParameterSet&);
  ~DummyBookFillDQMStore();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
                                    edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&,
                                  edm::EventSetup const&);

  void bookHistograms();
  void fillerDispose();

  // ----------member data ---------------------------
  std::vector<boost::shared_ptr<FillerBase> > m_runFillers;
  std::vector<boost::shared_ptr<FillerBase> > m_lumiFillers;
  std::string folder_;
  bool m_fillRuns;
  bool m_fillLumis;
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
DummyBookFillDQMStore::DummyBookFillDQMStore(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getUntrackedParameter<std::string>("folder", "TestFolder/")),
      m_fillRuns(iConfig.getUntrackedParameter<bool>("fillRuns")),
      m_fillLumis(iConfig.getUntrackedParameter<bool>("fillLumis")),
      book_at_constructor_(iConfig.getUntrackedParameter<bool>("book_at_constructor", false)),
      book_at_beginJob_(iConfig.getUntrackedParameter<bool>("book_at_beginJob", false)),
      book_at_beginRun_(iConfig.getUntrackedParameter<bool>("book_at_beginRun", false)),
      elements_(iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >("elements")) {
  // TODO(rovere): assert on multiple book conditions
  if (book_at_constructor_)
    bookHistograms();
}

void DummyBookFillDQMStore::fillerDispose(void) {
  std::cout << "fillerDispose" << std::endl;
  m_runFillers.erase(m_runFillers.begin(), m_runFillers.end());
  m_lumiFillers.erase(m_lumiFillers.begin(), m_lumiFillers.end());
}


void DummyBookFillDQMStore::bookHistograms() {
  fillerDispose();

  std::cout << "Booking" << std::endl;

  edm::Service<DQMStore> dstore;

  (*dstore).setCurrentFolder(folder_);

  if (m_fillRuns) {
    m_runFillers.reserve(elements_.size());
    PSets::const_iterator it = elements_.begin();
    PSets::const_iterator ite = elements_.end();
    for (; it != ite; ++it) {
      switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
        case 1:
          m_runFillers.push_back(boost::shared_ptr<FillerBase>(
              new TH1FFiller(*it, *dstore, false)));
          break;
        case 2:
          m_runFillers.push_back(boost::shared_ptr<FillerBase>(
              new TH2FFiller(*it, *dstore, false)));
          break;
      }
    }
  }

  if (m_fillLumis) {
    m_lumiFillers.reserve(elements_.size());
    for (PSets::const_iterator it = elements_.begin(), itEnd = elements_.end();
         it != itEnd; ++it) {
      switch (it->getUntrackedParameter<unsigned int>("type", 1)) {
        case 1:
          m_lumiFillers.push_back(boost::shared_ptr<FillerBase>(
              new TH1FFiller(*it, *dstore, true)));
          break;
        case 2:
          m_lumiFillers.push_back(boost::shared_ptr<FillerBase>(
              new TH2FFiller(*it, *dstore, true)));
          break;
      }
    }
  }
}

DummyBookFillDQMStore::~DummyBookFillDQMStore() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyBookFillDQMStore::analyze(edm::Event const& iEvent,
                               edm::EventSetup const& iSetup) {
//   using namespace edm;
  /* This is an event example
  //Read 'ExampleData' from the Event
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);

  //Use the ExampleData to create an ExampleData2 which
  // is put into the Event
  std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
  iEvent.put(pOut);
  */

  /* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
  */
}

// ------------ method called once each job just before starting event loop  ------------
void DummyBookFillDQMStore::beginJob() {
  if (book_at_beginJob_)
    bookHistograms();
}

// ------------ method called once each job just after ending the event loop  ------------
void DummyBookFillDQMStore::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void DummyBookFillDQMStore::beginRun(edm::Run const&, edm::EventSetup const&) {
  if (book_at_beginRun_)
    bookHistograms();
}

// ------------ method called when ending the processing of a run  ------------
void DummyBookFillDQMStore::endRun(edm::Run const&, edm::EventSetup const&) {
  std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_runFillers.begin();
  std::vector<boost::shared_ptr<FillerBase> >::iterator ite = m_runFillers.end();
  for (; it != ite; ++it)
    (*it)->fill();
}

// ------------ method called when starting to processes a luminosity block  ------------
void
DummyBookFillDQMStore::beginLuminosityBlock(edm::LuminosityBlock const&,
                                            edm::EventSetup const&) {
  std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin();
  std::vector<boost::shared_ptr<FillerBase> >::iterator ite = m_lumiFillers.end();
  for (; it != ite; ++it)
    (*it)->reset();
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
DummyBookFillDQMStore::endLuminosityBlock(edm::LuminosityBlock const&,
                                          edm::EventSetup const&) {
  std::vector<boost::shared_ptr<FillerBase> >::iterator it = m_lumiFillers.begin();
  std::vector<boost::shared_ptr<FillerBase> >::iterator ite = m_lumiFillers.end();
  for (; it != ite; ++it)
    (*it)->fill();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyBookFillDQMStore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(DummyBookFillDQMStore);
