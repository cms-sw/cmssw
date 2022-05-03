#include <iostream>
#include <fstream>

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "TH1.h"

#include <fastjet/JetDefinition.hh>
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/LeptonAnalyserHepMC.h"
#include "GeneratorInterface/Pythia8Interface/test/analyserhepmc/JetInputHepMC.h"

struct ParticlePtGreater {
  double operator()(const HepMC::GenParticle* v1, const HepMC::GenParticle* v2) {
    return v1->momentum().perp() > v2->momentum().perp();
  }
};

class ZJetsAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  //
  explicit ZJetsAnalyzer(const edm::ParameterSet&);
  virtual ~ZJetsAnalyzer() = default;  // no need to delete ROOT stuff
                                       // as it'll be deleted upon closing TFile

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override {}
  void endRun(const edm::Run&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<GenEventInfoProduct> tokenGenEvent_;
  const edm::EDGetTokenT<edm::HepMCProduct> tokenHepMC_;
  const edm::EDGetTokenT<GenRunInfoProduct> tokenGenRun_;

  LeptonAnalyserHepMC LA;
  JetInputHepMC JetInput;
  fastjet::Strategy strategy;
  fastjet::RecombinationScheme recombScheme;
  fastjet::JetDefinition* jetDef;

  int icategories[6];

  TH1D* fHist2muMass;
};

ZJetsAnalyzer::ZJetsAnalyzer(const edm::ParameterSet& pset)
    : tokenGenEvent_(consumes<GenEventInfoProduct>(
          edm::InputTag(pset.getUntrackedParameter("moduleLabel", std::string("generator")), ""))),
      tokenHepMC_(consumes<edm::HepMCProduct>(
          edm::InputTag(pset.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      tokenGenRun_(consumes<GenRunInfoProduct, edm::InRun>(
          edm::InputTag(pset.getUntrackedParameter("moduleLabel", std::string("generator")), ""))),
      fHist2muMass(0) {
  usesResource(TFileService::kSharedResource);
  // actually, pset is NOT in use - we keep it here just for illustratory putposes
}

void ZJetsAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  fHist2muMass = fs->make<TH1D>("Hist2muMass", "2-mu inv. mass", 100, 60., 120.);

  double Rparam = 0.5;
  strategy = fastjet::Best;
  recombScheme = fastjet::E_scheme;
  jetDef = new fastjet::JetDefinition(fastjet::antikt_algorithm, Rparam, recombScheme, strategy);

  for (int ind = 0; ind < 6; ind++) {
    icategories[ind] = 0;
  }

  return;
}

void ZJetsAnalyzer::endRun(const edm::Run& r, const edm::EventSetup&) {
  std::ofstream testi("testi.dat");
  double val, errval;

  const edm::Handle<GenRunInfoProduct>& genRunInfoProduct = r.getHandle(tokenGenRun_);

  val = (double)genRunInfoProduct->crossSection();
  std::cout << std::endl;
  std::cout << "cross section = " << val << std::endl;
  std::cout << std::endl;

  errval = 0.;
  if (icategories[0] > 0)
    errval = val / sqrt((double)(icategories[0]));
  testi << "pythia8_test1  1   " << val << " " << errval << " " << std::endl;

  std::cout << std::endl;
  std::cout << " Events with at least 1 isolated lepton  :                     "
            << ((double)icategories[1]) / ((double)icategories[0]) << std::endl;
  std::cout << " Events with at least 2 isolated leptons :                     "
            << ((double)icategories[2]) / ((double)icategories[0]) << std::endl;
  std::cout << " Events with at least 2 isolated leptons and at least 1 jet  : "
            << ((double)icategories[3]) / ((double)icategories[0]) << std::endl;
  std::cout << " Events with at least 2 isolated leptons and at least 2 jets : "
            << ((double)icategories[4]) / ((double)icategories[0]) << std::endl;
  std::cout << std::endl;

  val = ((double)icategories[4]) / ((double)icategories[0]);
  errval = 0.;
  if (icategories[4] > 0)
    errval = val / sqrt((double)icategories[4]);
  testi << "pythia8_test1  2   " << val << " " << errval << " " << std::endl;
}

void ZJetsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  icategories[0]++;

  // here's an example of accessing GenEventInfoProduct
  const edm::Handle<GenEventInfoProduct>& GenInfoHandle = e.getHandle(tokenGenEvent_);

  double qScale = GenInfoHandle->qScale();
  double pthat = (GenInfoHandle->hasBinningValues() ? (GenInfoHandle->binningValues())[0] : 0.0);
  std::cout << " qScale = " << qScale << " pthat = " << pthat << std::endl;
  //
  // this (commented out) code below just exemplifies how to access certain info
  //
  //double evt_weight1 = GenInfoHandle->weights()[0]; // this is "stanrd Py6 evt weight;
  // corresponds to PYINT1/VINT(97)
  //double evt_weight2 = GenInfoHandle->weights()[1]; // in case you run in CSA mode or otherwise
  // use PYEVWT routine, this will be weight
  // as returned by PYEVWT, i.e. PYINT1/VINT(99)
  //std::cout << " evt_weight1 = " << evt_weight1 << std::endl;
  //std::cout << " evt_weight2 = " << evt_weight2 << std::endl;
  //double weight = GenInfoHandle->weight();
  //std::cout << " as returned by the weight() method, integrated event weight = " << weight << std::endl;

  // here's an example of accessing particles in the event record (HepMCProduct)
  //
  const edm::Handle<edm::HepMCProduct>& EvtHandle = e.getHandle(tokenHepMC_);

  const HepMC::GenEvent* Evt = EvtHandle->GetEvent();

  int nisolep = LA.nIsolatedLeptons(Evt);

  //std::cout << "Number of leptons = " << nisolep << std::endl;
  if (nisolep > 0)
    icategories[1]++;
  if (nisolep > 1)
    icategories[2]++;

  JetInputHepMC::ParticleVector jetInput = JetInput(Evt);
  std::sort(jetInput.begin(), jetInput.end(), ParticlePtGreater());

  // Fastjet input
  std::vector<fastjet::PseudoJet> jfInput;
  jfInput.reserve(jetInput.size());
  for (JetInputHepMC::ParticleVector::const_iterator iter = jetInput.begin(); iter != jetInput.end(); ++iter) {
    jfInput.push_back(fastjet::PseudoJet(
        (*iter)->momentum().px(), (*iter)->momentum().py(), (*iter)->momentum().pz(), (*iter)->momentum().e()));
    jfInput.back().set_user_index(iter - jetInput.begin());
  }

  // Run Fastjet algorithm
  std::vector<fastjet::PseudoJet> inclusiveJets, sortedJets, cleanedJets;
  fastjet::ClusterSequence clustSeq(jfInput, *jetDef);

  // Extract inclusive jets sorted by pT (note minimum pT in GeV)
  inclusiveJets = clustSeq.inclusive_jets(20.0);
  sortedJets = sorted_by_pt(inclusiveJets);

  cleanedJets = LA.removeLeptonsFromJets(sortedJets, Evt);

  if (nisolep > 1) {
    if (cleanedJets.size() > 0)
      icategories[3]++;
    if (cleanedJets.size() > 1)
      icategories[4]++;
  }

  return;
}

typedef ZJetsAnalyzer ZJetsTest;
DEFINE_FWK_MODULE(ZJetsTest);
