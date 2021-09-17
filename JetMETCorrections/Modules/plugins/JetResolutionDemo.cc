#include <memory>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <DataFormats/PatCandidates/interface/Jet.h>

#include <TH2.h>

//
// class declaration
//

class JetResolutionDemo : public edm::EDAnalyzer {
public:
  explicit JetResolutionDemo(const edm::ParameterSet&);
  ~JetResolutionDemo() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<std::vector<pat::Jet>> m_jets_token;
  edm::EDGetTokenT<double> m_rho_token;

  bool m_debug = false;
  bool m_use_conddb = false;

  std::string m_payload;
  std::string m_resolutions_file;
  std::string m_scale_factors_file;

  edm::Service<TFileService> fs;

  TH2* m_res_vs_eta = nullptr;
};
//
//----------- Class Implementation ------------------------------------------
//
//---------------------------------------------------------------------------
JetResolutionDemo::JetResolutionDemo(const edm::ParameterSet& iConfig) {
  m_jets_token = consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"));
  m_rho_token = consumes<double>(iConfig.getParameter<edm::InputTag>("rho"));
  m_debug = iConfig.getUntrackedParameter<bool>("debug", false);
  m_use_conddb = iConfig.getUntrackedParameter<bool>("useCondDB", false);

  if (m_use_conddb)
    m_payload = iConfig.getParameter<std::string>("payload");
  else {
    m_resolutions_file = iConfig.getParameter<edm::FileInPath>("resolutionsFile").fullPath();
    m_scale_factors_file = iConfig.getParameter<edm::FileInPath>("scaleFactorsFile").fullPath();
  }
}
//---------------------------------------------------------------------------
JetResolutionDemo::~JetResolutionDemo() {}

//---------------------------------------------------------------------------
void JetResolutionDemo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::Jet>> jets;
  iEvent.getByToken(m_jets_token, jets);

  edm::Handle<double> rho;
  iEvent.getByToken(m_rho_token, rho);

  // Access jet resolution and scale factor from the condition database
  // or from text files
  JME::JetResolution resolution;
  JME::JetResolutionScaleFactor res_sf;

  // Two differents way to create a class instance
  if (m_use_conddb) {
    // First way, using the get() static method
    resolution = JME::JetResolution::get(iSetup, m_payload);
    res_sf = JME::JetResolutionScaleFactor::get(iSetup, m_payload);
  } else {
    // Second way, using the constructor
    resolution = JME::JetResolution(m_resolutions_file);
    res_sf = JME::JetResolutionScaleFactor(m_scale_factors_file);
  }

  if (m_debug) {
    // Dump resolution
    resolution.dump();
  }

  if (!m_res_vs_eta) {
    // Advanced usage. Create the histogram by retriving the eta binning directly from the JetResolutionObject. This suppose you need that the parametrization only depends on eta.

    // Get the list of bins of this object
    const std::vector<JME::Binning>& bins = resolution.getResolutionObject()->getDefinition().getBins();

    // Check that the first bin is eta
    if ((bins.size()) && (bins[0] == JME::Binning::JetEta)) {
      const std::vector<JME::JetResolutionObject::Record> records = resolution.getResolutionObject()->getRecords();
      // Get all records from the object. Each record correspond to a different binning and different parameters

      std::vector<float> etas;
      for (const auto& record : records) {
        if (etas.empty()) {
          etas.push_back(record.getBinsRange()[0].min);
          etas.push_back(record.getBinsRange()[0].max);
        } else {
          etas.push_back(record.getBinsRange()[0].max);
        }
      }

      std::vector<float> res;
      for (std::size_t i = 0; i < 40; i++) {
        res.push_back(i * 0.005);
      }

      m_res_vs_eta = fs->make<TH2F>("res_vs_eta", "res_vs_eta", etas.size() - 1, &etas[0], res.size() - 1, &res[0]);
    }
  }

  for (const auto& jet : *jets) {
    if (m_debug) {
      std::cout << "New jet; pt=" << jet.pt() << "  eta=" << jet.eta() << "  phi=" << jet.phi()
                << "  e=" << jet.energy() << "  rho=" << *rho << std::endl;
    }

    // Get resolution for this jet
    // The method to use is getResolution(const JME::JetParameters& parameters) from a JetResolution object

    // You *must* now in advance which variables are needed for getting the resolution. For the moment, only pt and eta are needed, but this will
    // probably change in the futur when PU dependency is added. Please keep an eye on the twiki page
    //     https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution
    // to stay up-to-date. All currently supported parameters (ie, 'set' functions) are available here:
    //     https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution#List_of_supported_parameters

    // Three way to create a JetParameters object

    // First, using 'set' functions
    JME::JetParameters parameters_1;
    parameters_1.setJetPt(jet.pt());
    parameters_1.setJetEta(jet.eta());
    parameters_1.setRho(*rho);

    // You can also chain calls

    JME::JetParameters parameters_2;
    parameters_2.setJetPt(jet.pt()).setJetEta(jet.eta()).setRho(*rho);

    // Second, using the set() function
    JME::JetParameters parameters_3;
    parameters_3.set(JME::Binning::JetPt, jet.pt());
    parameters_3.set({JME::Binning::JetEta, jet.eta()});
    parameters_3.set({JME::Binning::Rho, *rho});

    // Or

    JME::JetParameters parameters_4;
    parameters_4.set(JME::Binning::JetPt, jet.pt()).set(JME::Binning::JetEta, jet.eta()).set(JME::Binning::Rho, *rho);

    // Third, using a initializer_list
    JME::JetParameters parameters_5 = {
        {JME::Binning::JetPt, jet.pt()}, {JME::Binning::JetEta, jet.eta()}, {JME::Binning::Rho, *rho}};

    // Now, get the resolution

    float r = resolution.getResolution(parameters_1);
    if (m_debug) {
      std::cout << "Resolution with parameters_1: " << r << std::endl;
      std::cout << "Resolution with parameters_2: " << resolution.getResolution(parameters_2) << std::endl;
      std::cout << "Resolution with parameters_3: " << resolution.getResolution(parameters_3) << std::endl;
      std::cout << "Resolution with parameters_4: " << resolution.getResolution(parameters_4) << std::endl;
      std::cout << "Resolution with parameters_5: " << resolution.getResolution(parameters_5) << std::endl;

      // You can also use a shortcut to get the resolution
      float r2 = resolution.getResolution(
          {{JME::Binning::JetPt, jet.pt()}, {JME::Binning::JetEta, jet.eta()}, {JME::Binning::Rho, *rho}});
      std::cout << "Resolution using shortcut   : " << r2 << std::endl;
    }

    m_res_vs_eta->Fill(jet.eta(), r);

    // We do the same thing to access the scale factors
    float sf = res_sf.getScaleFactor({{JME::Binning::JetPt, jet.pt()}, {JME::Binning::JetEta, jet.eta()}});

    // Access up and down variation of the scale factor
    float sf_up =
        res_sf.getScaleFactor({{JME::Binning::JetPt, jet.pt()}, {JME::Binning::JetEta, jet.eta()}}, Variation::UP);
    float sf_down =
        res_sf.getScaleFactor({{JME::Binning::JetPt, jet.pt()}, {JME::Binning::JetEta, jet.eta()}}, Variation::DOWN);

    if (m_debug) {
      std::cout << "Scale factors (Nominal / Up / Down) : " << sf << " / " << sf_up << " / " << sf_down << std::endl;
    }
  }
}
//---------------------------------------------------------------------------
void JetResolutionDemo::endJob() {}
//---------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(JetResolutionDemo);
