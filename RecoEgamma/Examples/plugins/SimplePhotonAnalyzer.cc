/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
 ** \author Nancy Marinelli, U. of Notre Dame, US
 **
 **/

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TFile.h"
#include "TH1.h"
#include "TProfile.h"

#include <memory>
#include <string>

class SimplePhotonAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit SimplePhotonAnalyzer(const edm::ParameterSet&);
  ~SimplePhotonAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  float etaTransformation(float a, float b);

  std::string mcProducer_;
  std::string mcCollection_;
  std::string photonCollectionProducer_;
  std::string photonCollection_;
  std::string valueMapPFCandPhoton_;
  edm::InputTag pfEgammaCandidates_;
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;

  edm::ESHandle<CaloTopology> theCaloTopo_;

  std::string vertexProducer_;
  float sample_;

  DQMStore* dbe_;

  MonitorElement* h1_scEta_;
  MonitorElement* h1_deltaEtaSC_;
  MonitorElement* h1_pho_E_;
  MonitorElement* h1_pho_Et_;
  MonitorElement* h1_pho_Eta_;
  MonitorElement* h1_pho_Phi_;
  MonitorElement* h1_pho_R9Barrel_;
  MonitorElement* h1_pho_R9Endcap_;
  MonitorElement* h1_pho_sigmaIetaIetaBarrel_;
  MonitorElement* h1_pho_sigmaIetaIetaEndcap_;
  MonitorElement* h1_pho_hOverEBarrel_;
  MonitorElement* h1_pho_hOverEEndcap_;
  MonitorElement* h1_pho_ecalIsoBarrel_;
  MonitorElement* h1_pho_ecalIsoEndcap_;
  MonitorElement* h1_pho_hcalIsoBarrel_;
  MonitorElement* h1_pho_hcalIsoEndcap_;
  MonitorElement* h1_pho_trkIsoBarrel_;
  MonitorElement* h1_pho_trkIsoEndcap_;

  MonitorElement* h1_recEoverTrueEBarrel_;
  MonitorElement* h1_recEoverTrueEEndcap_;
  MonitorElement* h1_deltaEta_;
  MonitorElement* h1_deltaPhi_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePhotonAnalyzer);

//========================================================================
SimplePhotonAnalyzer::SimplePhotonAnalyzer(const edm::ParameterSet& ps)
//========================================================================
{
  photonCollectionProducer_ = ps.getParameter<std::string>("phoProducer");
  photonCollection_ = ps.getParameter<std::string>("photonCollection");

  barrelEcalHits_ = ps.getParameter<edm::InputTag>("barrelEcalHits");
  endcapEcalHits_ = ps.getParameter<edm::InputTag>("endcapEcalHits");

  pfEgammaCandidates_ = ps.getParameter<edm::InputTag>("pfEgammaCandidates");
  valueMapPFCandPhoton_ = ps.getParameter<std::string>("valueMapPhotons");

  mcProducer_ = ps.getParameter<std::string>("mcProducer");
  //mcCollection_ = ps.getParameter<std::string>("mcCollection");
  vertexProducer_ = ps.getParameter<std::string>("primaryVertexProducer");
  sample_ = ps.getParameter<int>("sample");
}

//========================================================================
SimplePhotonAnalyzer::~SimplePhotonAnalyzer()
//========================================================================
{}

//========================================================================
void SimplePhotonAnalyzer::beginJob() {
  //========================================================================

  dbe_ = nullptr;
  dbe_ = edm::Service<DQMStore>().operator->();

  float hiE = 0;
  float loE = 0;
  float hiEt = 0;
  float loEt = 0;
  float dPhi = 0;
  float loRes = 0;
  float hiRes = 0;
  if (sample_ == 1) {
    loE = 0.;
    hiE = 30.;
    loEt = 0.;
    hiEt = 30.;
    dPhi = 0.2;
    loRes = 0.;
    hiRes = 1.2;
  } else if (sample_ == 2) {
    loE = 0.;
    hiE = 200.;
    loEt = 0.;
    hiEt = 50.;
    dPhi = 0.05;
    loRes = 0.7;
    hiRes = 1.2;
  } else if (sample_ == 3) {
    loE = 0.;
    hiE = 500.;
    loEt = 0.;
    hiEt = 500.;
    dPhi = 0.05;
    loRes = 0.7;
    hiRes = 1.2;
  } else if (sample_ == 4) {
    loE = 0.;
    hiE = 6000.;
    loEt = 0.;
    hiEt = 1200.;
    dPhi = 0.05;
    loRes = 0.7;
    hiRes = 1.2;
  }

  h1_deltaEta_ = dbe_->book1D("deltaEta", " Reco photon Eta minus Generated photon Eta  ", 100, -0.2, 0.2);
  h1_deltaPhi_ = dbe_->book1D("deltaPhi", "Reco photon Phi minus Generated photon Phi ", 100, -dPhi, dPhi);
  h1_pho_Eta_ = dbe_->book1D("phoEta", "Photon  Eta ", 40, -3., 3.);
  h1_pho_Phi_ = dbe_->book1D("phoPhi", "Photon  Phi ", 40, -3.14, 3.14);
  h1_pho_E_ = dbe_->book1D("phoE", "Photon Energy ", 100, loE, hiE);
  h1_pho_Et_ = dbe_->book1D("phoEt", "Photon Et ", 100, loEt, hiEt);

  h1_scEta_ = dbe_->book1D("scEta", " SC Eta ", 40, -3., 3.);
  h1_deltaEtaSC_ = dbe_->book1D("deltaEtaSC", " SC Eta minus Generated photon Eta  ", 100, -0.02, 0.02);

  //
  h1_recEoverTrueEBarrel_ = dbe_->book1D(
      "recEoverTrueEBarrel", " Reco photon Energy over Generated photon Energy: Barrel ", 100, loRes, hiRes);
  h1_recEoverTrueEEndcap_ = dbe_->book1D(
      "recEoverTrueEEndcap", " Reco photon Energy over Generated photon Energy: Endcap ", 100, loRes, hiRes);

  //

  h1_pho_R9Barrel_ = dbe_->book1D("phoR9Barrel", "Photon  3x3 energy / SuperCluster energy : Barrel ", 100, 0., 1.2);
  h1_pho_R9Endcap_ = dbe_->book1D("phoR9Endcap", "Photon  3x3 energy / SuperCluster energy : Endcap ", 100, 0., 1.2);
  h1_pho_sigmaIetaIetaBarrel_ = dbe_->book1D("sigmaIetaIetaBarrel", "sigmaIetaIeta: Barrel", 100, 0., 0.05);
  h1_pho_sigmaIetaIetaEndcap_ = dbe_->book1D("sigmaIetaIetaEndcap", "sigmaIetaIeta: Endcap", 100, 0., 0.1);
  h1_pho_hOverEBarrel_ = dbe_->book1D("hOverEBarrel", "H/E: Barrel", 100, 0., 0.1);
  h1_pho_hOverEEndcap_ = dbe_->book1D("hOverEEndcap", "H/E: Endcap", 100, 0., 0.1);
  h1_pho_ecalIsoBarrel_ = dbe_->book1D("ecalIsolBarrel", "isolation et sum in Ecal: Barrel", 100, 0., 100.);
  h1_pho_ecalIsoEndcap_ = dbe_->book1D("ecalIsolEndcap", "isolation et sum in Ecal: Endcap", 100, 0., 100.);
  h1_pho_hcalIsoBarrel_ = dbe_->book1D("hcalIsolBarrel", "isolation et sum in Hcal: Barrel", 100, 0., 100.);
  h1_pho_hcalIsoEndcap_ = dbe_->book1D("hcalIsolEndcap", "isolation et sum in Hcal: Endcap", 100, 0., 100.);
  h1_pho_trkIsoBarrel_ = dbe_->book1D("trkIsolBarrel", "isolation pt sum in the tracker: Barrel", 100, 0., 100.);
  h1_pho_trkIsoEndcap_ = dbe_->book1D("trkIsolEndcap", "isolation pt sum in the tracker: Endcap", 100, 0., 100.);
}

//========================================================================
void SimplePhotonAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  //========================================================================

  using namespace edm;  // needed for all fwk related classes
  edm::LogInfo("PhotonAnalyzer") << "Analyzing event number: " << evt.id() << "\n";

  // get the  calo topology  from the event setup:
  edm::ESHandle<CaloTopology> pTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopo_);

  // Get the  corrected  photon collection (set in the configuration) which also contains infos about conversions

  Handle<reco::PhotonCollection> photonHandle;
  evt.getByLabel(photonCollectionProducer_, photonCollection_, photonHandle);
  const reco::PhotonCollection photonCollection = *(photonHandle.product());

  Handle<HepMCProduct> hepProd;
  evt.getByLabel(mcProducer_, hepProd);
  const HepMC::GenEvent* myGenEvent = hepProd->GetEvent();

  // Get the  PF refined cluster  collection
  Handle<reco::PFCandidateCollection> pfCandidateHandle;
  evt.getByLabel(pfEgammaCandidates_, pfCandidateHandle);
  if (!pfCandidateHandle.isValid()) {
    edm::LogError("SimplePhotonAnalyzer") << "Error! Can't get the product " << pfEgammaCandidates_.label();
  }

  edm::Handle<edm::ValueMap<reco::PhotonRef> > pfCandToPhotonMapHandle;
  edm::ValueMap<reco::PhotonRef> pfCandToPhotonMap;
  evt.getByLabel("gedPhotons", valueMapPFCandPhoton_, pfCandToPhotonMapHandle);
  if (!pfCandToPhotonMapHandle.isValid()) {
    edm::LogInfo("SimplePhotonAnalyzer") << "Error! Can't get the product: valueMapPhotons " << std::endl;
  }
  pfCandToPhotonMap = *(pfCandToPhotonMapHandle.product());

  std::cout << " SimplePhotonAnalyzer  valueMap size" << pfCandToPhotonMap.size() << std::endl;
  unsigned nObj = pfCandidateHandle->size();
  for (unsigned int lCand = 0; lCand < nObj; lCand++) {
    reco::PFCandidateRef pfCandRef(reco::PFCandidateRef(pfCandidateHandle, lCand));
    if (pfCandRef->particleId() != reco::PFCandidate::gamma)
      continue;
    reco::PhotonRef myPho = (pfCandToPhotonMap)[pfCandRef];
    if (myPho.isNonnull())
      std::cout << " PF SC " << pfCandRef->superClusterRef()->energy() << " Photon SC "
                << myPho->superCluster()->energy() << std::endl;
  }

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if (!((*p)->pdg_id() == 22 && (*p)->status() == 1))
      continue;

    // single primary photons or photons from Higgs or RS Graviton
    HepMC::GenParticle* mother = nullptr;
    if ((*p)->production_vertex()) {
      if ((*p)->production_vertex()->particles_begin(HepMC::parents) !=
          (*p)->production_vertex()->particles_end(HepMC::parents))
        mother = *((*p)->production_vertex()->particles_begin(HepMC::parents));
    }
    if (((mother == nullptr) || ((mother != nullptr) && (mother->pdg_id() == 25)) ||
         ((mother != nullptr) && (mother->pdg_id() == 22)))) {
      float minDelta = 10000.;
      std::vector<reco::Photon> localPhotons;
      int index = 0;
      int iMatch = -1;

      float phiPho = (*p)->momentum().phi();
      float etaPho = (*p)->momentum().eta();
      etaPho = etaTransformation(etaPho, (*p)->production_vertex()->position().z() / 10.);

      bool matched = false;
      // loop  Photon candidates
      for (reco::PhotonCollection::const_iterator iPho = photonCollection.begin(); iPho != photonCollection.end();
           iPho++) {
        reco::Photon localPho = reco::Photon(*iPho);
        localPhotons.push_back(localPho);

        /// Match reconstructed photon candidates with the nearest generated photonPho;
        float phiClu = localPho.phi();
        float etaClu = localPho.eta();
        float deltaPhi = phiClu - phiPho;
        float deltaEta = etaClu - etaPho;

        if (deltaPhi > pi)
          deltaPhi -= twopi;
        if (deltaPhi < -pi)
          deltaPhi += twopi;
        deltaPhi = std::pow(deltaPhi, 2);
        deltaEta = std::pow(deltaEta, 2);
        float delta = sqrt(deltaPhi + deltaEta);
        if (delta < 0.1 && delta < minDelta) {
          minDelta = delta;
          iMatch = index;
        }
        index++;
      }  // End loop over photons

      if (iMatch > -1)
        matched = true;

      /// Plot kinematic disctributions for matched photons
      if (matched) {
        reco::Photon matchingPho = localPhotons[iMatch];

        bool phoIsInBarrel = false;
        if (fabs(matchingPho.superCluster()->position().eta()) < 1.479) {
          phoIsInBarrel = true;
        }
        edm::Handle<EcalRecHitCollection> ecalRecHitHandle;

        h1_scEta_->Fill(matchingPho.superCluster()->position().eta());
        float trueEta = (*p)->momentum().eta();
        trueEta = etaTransformation(trueEta, (*p)->production_vertex()->position().z() / 10.);
        h1_deltaEtaSC_->Fill(localPhotons[iMatch].superCluster()->eta() - trueEta);

        float photonE = matchingPho.energy();
        float photonEt = matchingPho.et();
        float photonEta = matchingPho.eta();
        float photonPhi = matchingPho.phi();

        float r9 = matchingPho.r9();
        float sigmaIetaIeta = matchingPho.sigmaIetaIeta();
        float hOverE = matchingPho.hadronicOverEm();
        float ecalIso = matchingPho.ecalRecHitSumEtConeDR04();
        float hcalIso = matchingPho.hcalTowerSumEtConeDR04();
        float trkIso = matchingPho.trkSumPtSolidConeDR04();

        h1_pho_E_->Fill(photonE);
        h1_pho_Et_->Fill(photonEt);
        h1_pho_Eta_->Fill(photonEta);
        h1_pho_Phi_->Fill(photonPhi);

        h1_deltaEta_->Fill(photonEta - (*p)->momentum().eta());
        h1_deltaPhi_->Fill(photonPhi - (*p)->momentum().phi());

        if (phoIsInBarrel) {
          h1_recEoverTrueEBarrel_->Fill(photonE / (*p)->momentum().e());
          h1_pho_R9Barrel_->Fill(r9);
          h1_pho_sigmaIetaIetaBarrel_->Fill(sigmaIetaIeta);
          h1_pho_hOverEBarrel_->Fill(hOverE);
          h1_pho_ecalIsoBarrel_->Fill(ecalIso);
          h1_pho_hcalIsoBarrel_->Fill(hcalIso);
          h1_pho_trkIsoBarrel_->Fill(trkIso);

        } else {
          h1_recEoverTrueEEndcap_->Fill(photonE / (*p)->momentum().e());
          h1_pho_R9Endcap_->Fill(r9);
          h1_pho_sigmaIetaIetaEndcap_->Fill(sigmaIetaIeta);
          h1_pho_hOverEEndcap_->Fill(hOverE);
          h1_pho_ecalIsoEndcap_->Fill(ecalIso);
          h1_pho_hcalIsoEndcap_->Fill(hcalIso);
          h1_pho_trkIsoEndcap_->Fill(trkIso);
        }

      }  //  reco photon matching MC truth

    }  // End loop over MC particles
  }
}

float SimplePhotonAnalyzer::etaTransformation(float EtaParticle, float Zvertex) {
  //---Definitions
  const float PI = 3.1415927;
  //UNUSED const float TWOPI = 2.0*PI;

  //---Definitions for ECAL
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  //---ETA correction

  float Theta = 0.0;
  float ZEcal = R_ECAL * sinh(EtaParticle) + Zvertex;

  if (ZEcal != 0.0)
    Theta = atan(R_ECAL / ZEcal);
  if (Theta < 0.0)
    Theta = Theta + PI;
  float ETA = -log(tan(0.5 * Theta));

  if (fabs(ETA) > etaBarrelEndcap) {
    float Zend = Z_Endcap;
    if (EtaParticle < 0.0)
      Zend = -Zend;
    float Zlen = Zend - Zvertex;
    float RR = Zlen / sinh(EtaParticle);
    Theta = atan(RR / Zend);
    if (Theta < 0.0)
      Theta = Theta + PI;
    ETA = -log(tan(0.5 * Theta));
  }
  //---Return the result
  return ETA;
  //---end
}

//========================================================================
void SimplePhotonAnalyzer::endJob() {
  //========================================================================
}
