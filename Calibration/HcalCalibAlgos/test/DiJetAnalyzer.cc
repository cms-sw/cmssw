//
// DiJetAnalyzer.cc
//
// description: Create an ntuple necessary for dijet balance calibration for the HCAL
//

#include "DiJetAnalyzer.h"

DiJetAnalyzer::DiJetAnalyzer(const edm::ParameterSet& iConfig) {
  // set parameters
  pfJetCollName_ = iConfig.getParameter<std::string>("pfJetCollName");
  pfJetCorrName_ = iConfig.getParameter<std::string>("pfJetCorrName");
  hbheRecHitName_ = iConfig.getParameter<std::string>("hbheRecHitName");
  hfRecHitName_ = iConfig.getParameter<std::string>("hfRecHitName");
  hoRecHitName_ = iConfig.getParameter<std::string>("hoRecHitName");
  pvCollName_ = iConfig.getParameter<std::string>("pvCollName");
  rootHistFilename_ = iConfig.getParameter<std::string>("rootHistFilename");
  maxDeltaEta_ = iConfig.getParameter<double>("maxDeltaEta");
  minTagJetEta_ = iConfig.getParameter<double>("minTagJetEta");
  maxTagJetEta_ = iConfig.getParameter<double>("maxTagJetEta");
  minSumJetEt_ = iConfig.getParameter<double>("minSumJetEt");
  minJetEt_ = iConfig.getParameter<double>("minJetEt");
  maxThirdJetEt_ = iConfig.getParameter<double>("maxThirdJetEt");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  tok_PFJet_ = consumes<reco::PFJetCollection>(pfJetCollName_);
  tok_HBHE_ = consumes<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>>(hbheRecHitName_);
  tok_HF_ = consumes<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>>(hfRecHitName_);
  tok_HO_ = consumes<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>>(hoRecHitName_);
  tok_Vertex_ = consumes<reco::VertexCollection>(pvCollName_);

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
}

DiJetAnalyzer::~DiJetAnalyzer() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void DiJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  pf_Run_ = iEvent.id().run();
  pf_Lumi_ = iEvent.id().luminosityBlock();
  pf_Event_ = iEvent.id().event();

  // Get PFJets
  edm::Handle<reco::PFJetCollection> pfjets;
  iEvent.getByToken(tok_PFJet_, pfjets);
  if (!pfjets.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
        << " could not find PFJetCollection named " << pfJetCollName_ << ".\n";
    return;
  }

  // Get RecHits in HB and HE
  edm::Handle<edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>> hbhereco;
  iEvent.getByToken(tok_HBHE_, hbhereco);
  if (!hbhereco.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
        << " could not find HBHERecHit named " << hbheRecHitName_ << ".\n";
    return;
  }

  // Get RecHits in HF
  edm::Handle<edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>> hfreco;
  iEvent.getByToken(tok_HF_, hfreco);
  if (!hfreco.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HFRecHit named " << hfRecHitName_ << ".\n";
    return;
  }

  // Get RecHits in HO
  edm::Handle<edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>> horeco;
  iEvent.getByToken(tok_HO_, horeco);
  if (!horeco.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HORecHit named " << hoRecHitName_ << ".\n";
    return;
  }

  // Get geometry
  const CaloGeometry* geo = &evSetup.getData(tok_geom_);
  const HcalGeometry* HBGeom = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, 1));
  const HcalGeometry* HEGeom = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, 2));
  const CaloSubdetectorGeometry* HOGeom = geo->getSubdetectorGeometry(DetId::Hcal, 3);
  const CaloSubdetectorGeometry* HFGeom = geo->getSubdetectorGeometry(DetId::Hcal, 4);

  int HBHE_n = 0;
  for (edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>::const_iterator ith = hbhereco->begin();
       ith != hbhereco->end();
       ++ith) {
    HBHE_n++;
  }
  int HF_n = 0;
  for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator ith = hfreco->begin();
       ith != hfreco->end();
       ++ith) {
    HF_n++;
  }
  int HO_n = 0;
  for (edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>::const_iterator ith = horeco->begin();
       ith != horeco->end();
       ++ith) {
    HO_n++;
  }

  // Get primary vertices
  edm::Handle<std::vector<reco::Vertex>> pv;
  iEvent.getByToken(tok_Vertex_, pv);
  if (!pv.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find Vertex named " << pvCollName_ << ".\n";
    return;
  }
  pf_NPV_ = 0;
  for (std::vector<reco::Vertex>::const_iterator it = pv->begin(); it != pv->end(); ++it) {
    if (!it->isFake() && it->ndof() > 4)
      ++pf_NPV_;
  }

  // Get jet corrections
  const JetCorrector* correctorPF = JetCorrector::getJetCorrector(pfJetCorrName_, evSetup);

  //////////////////////////////
  // Event Selection
  //////////////////////////////

  // determine which cut results in failure
  int passSelPF = 0;

  // sort jets by corrected et
  std::set<JetCorretPair, JetCorretPairComp> pfjetcorretpairset;
  for (reco::PFJetCollection::const_iterator it = pfjets->begin(); it != pfjets->end(); ++it) {
    const reco::PFJet* jet = &(*it);
    double jec = correctorPF->correction(*it, iEvent, evSetup);
    pfjetcorretpairset.insert(JetCorretPair(jet, jec));
  }

  JetCorretPair pf_tag, pf_probe;
  pf_thirdjet_px_ = pf_thirdjet_py_ = 0.0;
  pf_realthirdjet_px_ = pf_realthirdjet_py_ = 0.0;
  pf_realthirdjet_px_ = 1;
  int cntr = 0;
  for (std::set<JetCorretPair, JetCorretPairComp>::const_iterator it = pfjetcorretpairset.begin();
       it != pfjetcorretpairset.end();
       ++it) {
    JetCorretPair jet = (*it);
    ++cntr;
    if (cntr == 1)
      pf_tag = jet;
    else if (cntr == 2)
      pf_probe = jet;
    else {
      pf_thirdjet_px_ += jet.scale() * jet.jet()->px();
      pf_thirdjet_py_ += jet.scale() * jet.jet()->py();
      if (cntr == 3) {
        pf_realthirdjet_px_ = jet.jet()->px();
        pf_realthirdjet_py_ = jet.jet()->py();
        pf_realthirdjet_scale_ = jet.scale();
      }
    }
  }

  if (pf_tag.jet() && pf_probe.jet()) {
    // require that the first two jets are above some minimum,
    // and the rest are below some maximum
    if ((pf_tag.jet()->et() + pf_probe.jet()->et()) < minSumJetEt_)
      passSelPF |= 0x1;
    if (pf_tag.jet()->et() < minJetEt_ || pf_probe.jet()->et() < minJetEt_)
      passSelPF |= 0x2;
    if (sqrt(pf_thirdjet_px_ * pf_thirdjet_px_ + pf_thirdjet_py_ * pf_thirdjet_py_) > maxThirdJetEt_)
      passSelPF |= 0x4;

    // force the tag jet to have the smaller |eta|
    if (std::fabs(pf_tag.jet()->eta()) > std::fabs(pf_probe.jet()->eta())) {
      JetCorretPair temp = pf_tag;
      pf_tag = pf_probe;
      pf_probe = temp;
    }

    // eta cuts
    double dAbsEta = std::fabs(std::fabs(pf_tag.jet()->eta()) - std::fabs(pf_probe.jet()->eta()));
    if (dAbsEta > maxDeltaEta_)
      passSelPF |= 0x8;
    if (fabs(pf_tag.jet()->eta()) < minTagJetEta_)
      passSelPF |= 0x10;
    if (fabs(pf_tag.jet()->eta()) > maxTagJetEta_)
      passSelPF |= 0x10;
  } else {
    passSelPF = 0x40;
  }

  h_PassSelPF_->Fill(passSelPF);
  if (passSelPF)
    return;
  // dump
  if (debug_) {
    std::cout << "Run: " << iEvent.id().run() << "; Event: " << iEvent.id().event() << std::endl;
    for (reco::PFJetCollection::const_iterator it = pfjets->begin(); it != pfjets->end(); ++it) {
      const reco::PFJet* jet = &(*it);
      std::cout << "istag=" << (jet == pf_tag.jet()) << "; isprobe=" << (jet == pf_probe.jet()) << "; et=" << jet->et()
                << "; eta=" << jet->eta() << std::endl;
    }
  }

  // Reset particle variables
  tpfjet_unkown_E_ = tpfjet_unkown_px_ = tpfjet_unkown_py_ = tpfjet_unkown_pz_ = tpfjet_unkown_EcalE_ = 0.0;
  tpfjet_electron_E_ = tpfjet_electron_px_ = tpfjet_electron_py_ = tpfjet_electron_pz_ = tpfjet_electron_EcalE_ = 0.0;
  tpfjet_muon_E_ = tpfjet_muon_px_ = tpfjet_muon_py_ = tpfjet_muon_pz_ = tpfjet_muon_EcalE_ = 0.0;
  tpfjet_photon_E_ = tpfjet_photon_px_ = tpfjet_photon_py_ = tpfjet_photon_pz_ = tpfjet_photon_EcalE_ = 0.0;
  tpfjet_unkown_n_ = tpfjet_electron_n_ = tpfjet_muon_n_ = tpfjet_photon_n_ = 0;
  tpfjet_had_n_ = 0;
  tpfjet_cluster_n_ = 0;
  ppfjet_unkown_E_ = ppfjet_unkown_px_ = ppfjet_unkown_py_ = ppfjet_unkown_pz_ = ppfjet_unkown_EcalE_ = 0.0;
  ppfjet_electron_E_ = ppfjet_electron_px_ = ppfjet_electron_py_ = ppfjet_electron_pz_ = ppfjet_electron_EcalE_ = 0.0;
  ppfjet_muon_E_ = ppfjet_muon_px_ = ppfjet_muon_py_ = ppfjet_muon_pz_ = ppfjet_muon_EcalE_ = 0.0;
  ppfjet_photon_E_ = ppfjet_photon_px_ = ppfjet_photon_py_ = ppfjet_photon_pz_ = ppfjet_photon_EcalE_ = 0.0;
  ppfjet_unkown_n_ = ppfjet_electron_n_ = ppfjet_muon_n_ = ppfjet_photon_n_ = 0;
  ppfjet_had_n_ = 0;
  ppfjet_cluster_n_ = 0;

  tpfjet_had_E_.clear();
  tpfjet_had_px_.clear();
  tpfjet_had_py_.clear();
  tpfjet_had_pz_.clear();
  tpfjet_had_EcalE_.clear();
  tpfjet_had_rawHcalE_.clear();
  tpfjet_had_emf_.clear();
  tpfjet_had_id_.clear();
  tpfjet_had_candtrackind_.clear();
  tpfjet_had_ntwrs_.clear();
  tpfjet_twr_ieta_.clear();
  tpfjet_twr_iphi_.clear();
  tpfjet_twr_depth_.clear();
  tpfjet_twr_subdet_.clear();
  tpfjet_twr_candtrackind_.clear();
  tpfjet_twr_hadind_.clear();
  tpfjet_twr_elmttype_.clear();
  tpfjet_twr_hade_.clear();
  tpfjet_twr_frac_.clear();
  tpfjet_twr_dR_.clear();
  tpfjet_twr_clusterind_.clear();
  tpfjet_cluster_eta_.clear();
  tpfjet_cluster_phi_.clear();
  tpfjet_cluster_dR_.clear();
  tpfjet_candtrack_px_.clear();
  tpfjet_candtrack_py_.clear();
  tpfjet_candtrack_pz_.clear();
  tpfjet_candtrack_EcalE_.clear();
  ppfjet_had_E_.clear();
  ppfjet_had_px_.clear();
  ppfjet_had_py_.clear();
  ppfjet_had_pz_.clear();
  ppfjet_had_EcalE_.clear();
  ppfjet_had_rawHcalE_.clear();
  ppfjet_had_emf_.clear();
  ppfjet_had_id_.clear();
  ppfjet_had_candtrackind_.clear();
  ppfjet_had_ntwrs_.clear();
  ppfjet_twr_ieta_.clear();
  ppfjet_twr_iphi_.clear();
  ppfjet_twr_depth_.clear();
  ppfjet_twr_subdet_.clear();
  ppfjet_twr_candtrackind_.clear();
  ppfjet_twr_hadind_.clear();
  ppfjet_twr_elmttype_.clear();
  ppfjet_twr_hade_.clear();
  ppfjet_twr_frac_.clear();
  ppfjet_twr_dR_.clear();
  ppfjet_twr_clusterind_.clear();
  ppfjet_cluster_eta_.clear();
  ppfjet_cluster_phi_.clear();
  ppfjet_cluster_dR_.clear();
  ppfjet_candtrack_px_.clear();
  ppfjet_candtrack_py_.clear();
  ppfjet_candtrack_pz_.clear();
  ppfjet_candtrack_EcalE_.clear();

  std::map<int, std::pair<int, std::set<float>>> tpfjet_rechits;
  std::map<int, std::pair<int, std::set<float>>> ppfjet_rechits;
  std::map<float, int> tpfjet_clusters;
  std::map<float, int> ppfjet_clusters;

  // fill tag jet variables
  tpfjet_pt_ = pf_tag.jet()->pt();
  tpfjet_p_ = pf_tag.jet()->p();
  tpfjet_E_ = pf_tag.jet()->energy();
  tpfjet_eta_ = pf_tag.jet()->eta();
  tpfjet_phi_ = pf_tag.jet()->phi();
  tpfjet_scale_ = pf_tag.scale();
  tpfjet_area_ = pf_tag.jet()->jetArea();
  tpfjet_ntwrs_ = 0;
  tpfjet_ncandtracks_ = 0;

  tpfjet_jetID_ = 0;  // Not a loose, medium, or tight jet
  if (fabs(pf_tag.jet()->eta()) < 2.4) {
    if (pf_tag.jet()->chargedHadronEnergyFraction() > 0 && pf_tag.jet()->chargedMultiplicity() > 0 &&
        pf_tag.jet()->chargedEmEnergyFraction() < 0.99 &&
        (pf_tag.jet()->chargedMultiplicity() + pf_tag.jet()->neutralMultiplicity()) > 1) {
      if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.9 && pf_tag.jet()->neutralEmEnergyFraction() < 0.9) {
        tpfjet_jetID_ = 3;  // Tight jet
      } else if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.95 && pf_tag.jet()->neutralEmEnergyFraction() < 0.95) {
        tpfjet_jetID_ = 2;  // Medium jet
      } else if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.99 && pf_tag.jet()->neutralEmEnergyFraction() < 0.99) {
        tpfjet_jetID_ = 1;  // Loose jet
      }
    }
  } else if ((pf_tag.jet()->chargedMultiplicity() + pf_tag.jet()->neutralMultiplicity()) > 1) {
    if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.9 && pf_tag.jet()->neutralEmEnergyFraction() < 0.9) {
      tpfjet_jetID_ = 3;  // Tight jet
    } else if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.95 && pf_tag.jet()->neutralEmEnergyFraction() < 0.95) {
      tpfjet_jetID_ = 2;  // Medium jet
    } else if (pf_tag.jet()->neutralHadronEnergyFraction() < 0.99 && pf_tag.jet()->neutralEmEnergyFraction() < 0.99) {
      tpfjet_jetID_ = 1;  // Loose jet
    }
  }

  /////////////////////////////////////////////
  // Get PF constituents and fill HCAL towers
  /////////////////////////////////////////////

  // Get tag PFCandidates
  std::vector<reco::PFCandidatePtr> tagconst = pf_tag.jet()->getPFConstituents();
  for (std::vector<reco::PFCandidatePtr>::const_iterator it = tagconst.begin(); it != tagconst.end(); ++it) {
    bool hasTrack = false;
    // Test PFCandidate type
    reco::PFCandidate::ParticleType candidateType = (*it)->particleId();
    switch (candidateType) {
      case reco::PFCandidate::X:
        tpfjet_unkown_E_ += (*it)->energy();
        tpfjet_unkown_px_ += (*it)->px();
        tpfjet_unkown_py_ += (*it)->py();
        tpfjet_unkown_pz_ += (*it)->pz();
        tpfjet_unkown_EcalE_ += (*it)->ecalEnergy();
        tpfjet_unkown_n_++;
        continue;
      case reco::PFCandidate::h: {
        tpfjet_had_E_.push_back((*it)->energy());
        tpfjet_had_px_.push_back((*it)->px());
        tpfjet_had_py_.push_back((*it)->py());
        tpfjet_had_pz_.push_back((*it)->pz());
        tpfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        tpfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        tpfjet_had_id_.push_back(0);
        tpfjet_had_ntwrs_.push_back(0);
        tpfjet_had_n_++;

        reco::TrackRef trackRef = (*it)->trackRef();
        if (trackRef.isNonnull()) {
          reco::Track track = *trackRef;
          tpfjet_candtrack_px_.push_back(track.px());
          tpfjet_candtrack_py_.push_back(track.py());
          tpfjet_candtrack_pz_.push_back(track.pz());
          tpfjet_candtrack_EcalE_.push_back((*it)->ecalEnergy());
          tpfjet_had_candtrackind_.push_back(tpfjet_ncandtracks_);
          hasTrack = true;
          tpfjet_ncandtracks_++;
        } else {
          tpfjet_had_candtrackind_.push_back(-2);
        }
      } break;
      case reco::PFCandidate::e:
        tpfjet_electron_E_ += (*it)->energy();
        tpfjet_electron_px_ += (*it)->px();
        tpfjet_electron_py_ += (*it)->py();
        tpfjet_electron_pz_ += (*it)->pz();
        tpfjet_electron_EcalE_ += (*it)->ecalEnergy();
        tpfjet_electron_n_++;
        continue;
      case reco::PFCandidate::mu:
        tpfjet_muon_E_ += (*it)->energy();
        tpfjet_muon_px_ += (*it)->px();
        tpfjet_muon_py_ += (*it)->py();
        tpfjet_muon_pz_ += (*it)->pz();
        tpfjet_muon_EcalE_ += (*it)->ecalEnergy();
        tpfjet_muon_n_++;
        continue;
      case reco::PFCandidate::gamma:
        tpfjet_photon_E_ += (*it)->energy();
        tpfjet_photon_px_ += (*it)->px();
        tpfjet_photon_py_ += (*it)->py();
        tpfjet_photon_pz_ += (*it)->pz();
        tpfjet_photon_EcalE_ += (*it)->ecalEnergy();
        tpfjet_photon_n_++;
        continue;
      case reco::PFCandidate::h0: {
        tpfjet_had_E_.push_back((*it)->energy());
        tpfjet_had_px_.push_back((*it)->px());
        tpfjet_had_py_.push_back((*it)->py());
        tpfjet_had_pz_.push_back((*it)->pz());
        tpfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        tpfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        tpfjet_had_id_.push_back(1);
        tpfjet_had_candtrackind_.push_back(-1);
        tpfjet_had_ntwrs_.push_back(0);
        tpfjet_had_n_++;
        break;
      }
      case reco::PFCandidate::h_HF: {
        tpfjet_had_E_.push_back((*it)->energy());
        tpfjet_had_px_.push_back((*it)->px());
        tpfjet_had_py_.push_back((*it)->py());
        tpfjet_had_pz_.push_back((*it)->pz());
        tpfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        tpfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        tpfjet_had_id_.push_back(2);
        tpfjet_had_candtrackind_.push_back(-1);
        tpfjet_had_ntwrs_.push_back(0);
        tpfjet_had_n_++;
        break;
      }
      case reco::PFCandidate::egamma_HF: {
        tpfjet_had_E_.push_back((*it)->energy());
        tpfjet_had_px_.push_back((*it)->px());
        tpfjet_had_py_.push_back((*it)->py());
        tpfjet_had_pz_.push_back((*it)->pz());
        tpfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        tpfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        tpfjet_had_id_.push_back(3);
        tpfjet_had_candtrackind_.push_back(-1);
        tpfjet_had_ntwrs_.push_back(0);
        tpfjet_had_n_++;
        break;
      }
    }

    std::map<int, int> twrietas;
    float HFHAD_E = 0;
    float HFEM_E = 0;
    int maxElement = (*it)->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = (*it)->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == (*it)->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL) {  // Element is HB or HE
            // Get cluster and hits
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            reco::PFCluster cluster = *clusterref;
            double cluster_dR = deltaR(tpfjet_eta_, tpfjet_phi_, cluster.eta(), cluster.phi());
            if (tpfjet_clusters.count(cluster_dR) == 0) {
              tpfjet_clusters[cluster_dR] = tpfjet_cluster_n_;
              tpfjet_cluster_eta_.push_back(cluster.eta());
              tpfjet_cluster_phi_.push_back(cluster.phi());
              tpfjet_cluster_dR_.push_back(cluster_dR);
              tpfjet_cluster_n_++;
            }
            int cluster_ind = tpfjet_clusters[cluster_dR];

            std::vector<std::pair<DetId, float>> hitsAndFracs = cluster.hitsAndFractions();

            // Run over hits and match
            int nHits = hitsAndFracs.size();
            for (int iHit = 0; iHit < nHits; iHit++) {
              int etaPhiPF = getEtaPhi(hitsAndFracs[iHit].first);

              for (edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>::const_iterator ith =
                       hbhereco->begin();
                   ith != hbhereco->end();
                   ++ith) {
                int etaPhiRecHit = getEtaPhi((*ith).id());
                if (etaPhiPF == etaPhiRecHit) {
                  tpfjet_had_ntwrs_.at(tpfjet_had_n_ - 1)++;
                  if (tpfjet_rechits.count((*ith).id()) == 0) {
                    tpfjet_twr_ieta_.push_back((*ith).id().ieta());
                    tpfjet_twr_iphi_.push_back((*ith).id().iphi());
                    tpfjet_twr_depth_.push_back((*ith).id().depth());
                    tpfjet_twr_subdet_.push_back((*ith).id().subdet());
                    if (hitsAndFracs[iHit].second > 0.05 && (*ith).energy() > 0.0)
                      twrietas[(*ith).id().ieta()]++;
                    tpfjet_twr_hade_.push_back((*ith).energy());
                    tpfjet_twr_frac_.push_back(hitsAndFracs[iHit].second);
                    tpfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                    tpfjet_twr_hadind_.push_back(tpfjet_had_n_ - 1);
                    tpfjet_twr_elmttype_.push_back(0);
                    tpfjet_twr_clusterind_.push_back(cluster_ind);
                    if (hasTrack) {
                      tpfjet_twr_candtrackind_.push_back(tpfjet_ncandtracks_ - 1);
                    } else {
                      tpfjet_twr_candtrackind_.push_back(-1);
                    }
                    switch ((*ith).id().subdet()) {
                      case HcalSubdetector::HcalBarrel: {
                        CaloCellGeometry::CornersVec cv = HBGeom->getCorners((*ith).id());
                        float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                        float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                        if (cv[0].phi() < cv[2].phi())
                          avgphi = (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) +
                                    static_cast<double>(cv[2].phi())) /
                                   2.0;
                        tpfjet_twr_dR_.push_back(deltaR(tpfjet_eta_, tpfjet_phi_, avgeta, avgphi));
                        break;
                      }
                      case HcalSubdetector::HcalEndcap: {
                        CaloCellGeometry::CornersVec cv = HEGeom->getCorners((*ith).id());
                        float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                        float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                        if (cv[0].phi() < cv[2].phi())
                          avgphi = (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) +
                                    static_cast<double>(cv[2].phi())) /
                                   2.0;
                        tpfjet_twr_dR_.push_back(deltaR(tpfjet_eta_, tpfjet_phi_, avgeta, avgphi));
                        break;
                      }
                      default:
                        tpfjet_twr_dR_.push_back(-1);
                        break;
                    }
                    tpfjet_rechits[(*ith).id()].first = tpfjet_ntwrs_;
                    ++tpfjet_ntwrs_;
                  } else if (tpfjet_rechits[(*ith).id()].second.count(hitsAndFracs[iHit].second) == 0) {
                    tpfjet_twr_frac_.at(tpfjet_rechits[(*ith).id()].first) += hitsAndFracs[iHit].second;
                    if (cluster_dR <
                        tpfjet_cluster_dR_.at(tpfjet_twr_clusterind_.at(tpfjet_rechits[(*ith).id()].first))) {
                      tpfjet_twr_clusterind_.at(tpfjet_rechits[(*ith).id()].first) = cluster_ind;
                    }
                    tpfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                  }
                }                                                           // Test if ieta,iphi matches
              }                                                             // Loop over rechits
            }                                                               // Loop over hits
          }                                                                 // Test if element is from HCAL
          else if (elements[iEle].type() == reco::PFBlockElement::HFHAD) {  // Element is HF
            for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator ith =
                     hfreco->begin();
                 ith != hfreco->end();
                 ++ith) {
              if ((*ith).id().depth() == 1)
                continue;  // Remove long fibers
              auto thisCell = HFGeom->getGeometry((*ith).id().rawId());
              const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();

              bool passMatch = false;
              if ((*it)->eta() < cv[0].eta() && (*it)->eta() > cv[2].eta()) {
                if ((*it)->phi() < cv[0].phi() && (*it)->phi() > cv[2].phi())
                  passMatch = true;
                else if (cv[0].phi() < cv[2].phi()) {
                  if ((*it)->phi() < cv[0].phi())
                    passMatch = true;
                  else if ((*it)->phi() > cv[2].phi())
                    passMatch = true;
                }
              }

              if (passMatch) {
                tpfjet_had_ntwrs_.at(tpfjet_had_n_ - 1)++;
                tpfjet_twr_ieta_.push_back((*ith).id().ieta());
                tpfjet_twr_iphi_.push_back((*ith).id().iphi());
                tpfjet_twr_depth_.push_back((*ith).id().depth());
                tpfjet_twr_subdet_.push_back((*ith).id().subdet());
                tpfjet_twr_hade_.push_back((*ith).energy());
                tpfjet_twr_frac_.push_back(1.0);
                tpfjet_twr_hadind_.push_back(tpfjet_had_n_ - 1);
                tpfjet_twr_elmttype_.push_back(1);
                tpfjet_twr_clusterind_.push_back(-1);
                tpfjet_twr_candtrackind_.push_back(-1);
                float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                if (cv[0].phi() < cv[2].phi())
                  avgphi =
                      (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                tpfjet_twr_dR_.push_back(deltaR(tpfjet_eta_, tpfjet_phi_, avgeta, avgphi));
                ++tpfjet_ntwrs_;
                HFHAD_E += (*ith).energy();
              }
            }
          } else if (elements[iEle].type() == reco::PFBlockElement::HFEM) {  // Element is HF
            for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator ith =
                     hfreco->begin();
                 ith != hfreco->end();
                 ++ith) {
              if ((*ith).id().depth() == 2)
                continue;  // Remove short fibers
              auto thisCell = HFGeom->getGeometry((*ith).id().rawId());
              const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();

              bool passMatch = false;
              if ((*it)->eta() < cv[0].eta() && (*it)->eta() > cv[2].eta()) {
                if ((*it)->phi() < cv[0].phi() && (*it)->phi() > cv[2].phi())
                  passMatch = true;
                else if (cv[0].phi() < cv[2].phi()) {
                  if ((*it)->phi() < cv[0].phi())
                    passMatch = true;
                  else if ((*it)->phi() > cv[2].phi())
                    passMatch = true;
                }
              }

              if (passMatch) {
                tpfjet_had_ntwrs_.at(tpfjet_had_n_ - 1)++;
                tpfjet_twr_ieta_.push_back((*ith).id().ieta());
                tpfjet_twr_iphi_.push_back((*ith).id().iphi());
                tpfjet_twr_depth_.push_back((*ith).id().depth());
                tpfjet_twr_subdet_.push_back((*ith).id().subdet());
                tpfjet_twr_hade_.push_back((*ith).energy());
                tpfjet_twr_frac_.push_back(1.0);
                tpfjet_twr_hadind_.push_back(tpfjet_had_n_ - 1);
                tpfjet_twr_elmttype_.push_back(2);
                tpfjet_twr_clusterind_.push_back(-1);
                tpfjet_twr_candtrackind_.push_back(-1);
                float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                if (cv[0].phi() < cv[2].phi())
                  avgphi =
                      (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                tpfjet_twr_dR_.push_back(deltaR(tpfjet_eta_, tpfjet_phi_, avgeta, avgphi));
                ++tpfjet_ntwrs_;
                HFEM_E += (*ith).energy();
              }
            }
          } else if (elements[iEle].type() == reco::PFBlockElement::HO) {  // Element is HO
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            reco::PFCluster cluster = *clusterref;
            double cluster_dR = deltaR(tpfjet_eta_, tpfjet_phi_, cluster.eta(), cluster.phi());
            if (tpfjet_clusters.count(cluster_dR) == 0) {
              tpfjet_clusters[cluster_dR] = tpfjet_cluster_n_;
              tpfjet_cluster_eta_.push_back(cluster.eta());
              tpfjet_cluster_phi_.push_back(cluster.phi());
              tpfjet_cluster_dR_.push_back(cluster_dR);
              tpfjet_cluster_n_++;
            }
            int cluster_ind = tpfjet_clusters[cluster_dR];

            std::vector<std::pair<DetId, float>> hitsAndFracs = cluster.hitsAndFractions();
            int nHits = hitsAndFracs.size();
            for (int iHit = 0; iHit < nHits; iHit++) {
              int etaPhiPF = getEtaPhi(hitsAndFracs[iHit].first);

              for (edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>::const_iterator ith =
                       horeco->begin();
                   ith != horeco->end();
                   ++ith) {
                int etaPhiRecHit = getEtaPhi((*ith).id());
                if (etaPhiPF == etaPhiRecHit) {
                  tpfjet_had_ntwrs_.at(tpfjet_had_n_ - 1)++;
                  if (tpfjet_rechits.count((*ith).id()) == 0) {
                    tpfjet_twr_ieta_.push_back((*ith).id().ieta());
                    tpfjet_twr_iphi_.push_back((*ith).id().iphi());
                    tpfjet_twr_depth_.push_back((*ith).id().depth());
                    tpfjet_twr_subdet_.push_back((*ith).id().subdet());
                    if (hitsAndFracs[iHit].second > 0.05 && (*ith).energy() > 0.0)
                      twrietas[(*ith).id().ieta()]++;
                    tpfjet_twr_hade_.push_back((*ith).energy());
                    tpfjet_twr_frac_.push_back(hitsAndFracs[iHit].second);
                    tpfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                    tpfjet_twr_hadind_.push_back(tpfjet_had_n_ - 1);
                    tpfjet_twr_elmttype_.push_back(3);
                    tpfjet_twr_clusterind_.push_back(cluster_ind);
                    if (hasTrack) {
                      tpfjet_twr_candtrackind_.push_back(tpfjet_ncandtracks_ - 1);
                    } else {
                      tpfjet_twr_candtrackind_.push_back(-1);
                    }
                    auto thisCell = HOGeom->getGeometry((*ith).id().rawId());
                    const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();
                    float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                    float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                    if (cv[0].phi() < cv[2].phi())
                      avgphi =
                          (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) /
                          2.0;
                    tpfjet_twr_dR_.push_back(deltaR(tpfjet_eta_, tpfjet_phi_, avgeta, avgphi));
                    tpfjet_rechits[(*ith).id()].first = tpfjet_ntwrs_;
                    ++tpfjet_ntwrs_;
                  } else if (tpfjet_rechits[(*ith).id()].second.count(hitsAndFracs[iHit].second) == 0) {
                    tpfjet_twr_frac_.at(tpfjet_rechits[(*ith).id()].first) += hitsAndFracs[iHit].second;
                    if (cluster_dR <
                        tpfjet_cluster_dR_.at(tpfjet_twr_clusterind_.at(tpfjet_rechits[(*ith).id()].first))) {
                      tpfjet_twr_clusterind_.at(tpfjet_rechits[(*ith).id()].first) = cluster_ind;
                    }
                    tpfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                  }
                }  // Test if ieta,iphi match
              }    // Loop over rechits
            }      // Loop over hits
          }        // Test if element is HO
        }          // Test for right element index
      }            // Loop over elements
    }              // Loop over elements in blocks

    switch (candidateType) {
      case reco::PFCandidate::h_HF:
        tpfjet_had_emf_.push_back(HFEM_E / (HFEM_E + HFHAD_E));
        break;
      case reco::PFCandidate::egamma_HF:
        tpfjet_had_emf_.push_back(-1);
        break;
      default:
        tpfjet_had_emf_.push_back(-1);
        break;
    }
  }  // Loop over PF constitutents

  int tag_had_EcalE = 0;
  int tag_had_rawHcalE = 0;
  for (int i = 0; i < tpfjet_had_n_; i++) {
    tag_had_EcalE += tpfjet_had_EcalE_[i];
    tag_had_rawHcalE += tpfjet_had_rawHcalE_[i];
  }
  tpfjet_EMfrac_ = 1.0 - tag_had_rawHcalE / (tag_had_rawHcalE + tag_had_EcalE + tpfjet_unkown_E_ + tpfjet_electron_E_ +
                                             tpfjet_muon_E_ + tpfjet_photon_E_);
  tpfjet_hadEcalEfrac_ = tag_had_EcalE / (tag_had_rawHcalE + tag_had_EcalE + tpfjet_unkown_E_ + tpfjet_electron_E_ +
                                          tpfjet_muon_E_ + tpfjet_photon_E_);

  // fill probe jet variables
  ppfjet_pt_ = pf_probe.jet()->pt();
  ppfjet_p_ = pf_probe.jet()->p();
  ppfjet_E_ = pf_probe.jet()->energy();
  ppfjet_eta_ = pf_probe.jet()->eta();
  ppfjet_phi_ = pf_probe.jet()->phi();
  ppfjet_scale_ = pf_probe.scale();
  ppfjet_area_ = pf_probe.jet()->jetArea();
  ppfjet_ntwrs_ = 0;
  ppfjet_ncandtracks_ = 0;

  ppfjet_jetID_ = 0;  // Not a loose, medium, or tight jet
  if (fabs(pf_probe.jet()->eta()) < 2.4) {
    if (pf_probe.jet()->chargedHadronEnergyFraction() > 0 && pf_probe.jet()->chargedMultiplicity() > 0 &&
        pf_probe.jet()->chargedEmEnergyFraction() < 0.99 &&
        (pf_probe.jet()->chargedMultiplicity() + pf_probe.jet()->neutralMultiplicity()) > 1) {
      if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.9 && pf_probe.jet()->neutralEmEnergyFraction() < 0.9) {
        ppfjet_jetID_ = 3;  // Tight jet
      } else if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.95 &&
                 pf_probe.jet()->neutralEmEnergyFraction() < 0.95) {
        ppfjet_jetID_ = 2;  // Medium jet
      } else if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.99 &&
                 pf_probe.jet()->neutralEmEnergyFraction() < 0.99) {
        ppfjet_jetID_ = 1;  // Loose jet
      }
    }
  } else if ((pf_probe.jet()->chargedMultiplicity() + pf_probe.jet()->neutralMultiplicity()) > 1) {
    if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.9 && pf_probe.jet()->neutralEmEnergyFraction() < 0.9) {
      ppfjet_jetID_ = 3;  // Tight jet
    } else if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.95 &&
               pf_probe.jet()->neutralEmEnergyFraction() < 0.95) {
      ppfjet_jetID_ = 2;  // Medium jet
    } else if (pf_probe.jet()->neutralHadronEnergyFraction() < 0.99 &&
               pf_probe.jet()->neutralEmEnergyFraction() < 0.99) {
      ppfjet_jetID_ = 1;  // Loose jet
    }
  }

  // Get PF constituents and fill HCAL towers
  std::vector<reco::PFCandidatePtr> probeconst = pf_probe.jet()->getPFConstituents();
  for (std::vector<reco::PFCandidatePtr>::const_iterator it = probeconst.begin(); it != probeconst.end(); ++it) {
    bool hasTrack = false;
    reco::PFCandidate::ParticleType candidateType = (*it)->particleId();
    switch (candidateType) {
      case reco::PFCandidate::X:
        ppfjet_unkown_E_ += (*it)->energy();
        ppfjet_unkown_px_ += (*it)->px();
        ppfjet_unkown_py_ += (*it)->py();
        ppfjet_unkown_pz_ += (*it)->pz();
        ppfjet_unkown_EcalE_ += (*it)->ecalEnergy();
        ppfjet_unkown_n_++;
        continue;
      case reco::PFCandidate::h: {
        ppfjet_had_E_.push_back((*it)->energy());
        ppfjet_had_px_.push_back((*it)->px());
        ppfjet_had_py_.push_back((*it)->py());
        ppfjet_had_pz_.push_back((*it)->pz());
        ppfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        ppfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        ppfjet_had_id_.push_back(0);
        ppfjet_had_ntwrs_.push_back(0);
        ppfjet_had_n_++;

        reco::TrackRef trackRef = (*it)->trackRef();
        if (trackRef.isNonnull()) {
          reco::Track track = *trackRef;
          ppfjet_candtrack_px_.push_back(track.px());
          ppfjet_candtrack_py_.push_back(track.py());
          ppfjet_candtrack_pz_.push_back(track.pz());
          ppfjet_candtrack_EcalE_.push_back((*it)->ecalEnergy());
          ppfjet_had_candtrackind_.push_back(ppfjet_ncandtracks_);
          hasTrack = true;
          ppfjet_ncandtracks_++;
        } else {
          ppfjet_had_candtrackind_.push_back(-2);
        }
      } break;
      case reco::PFCandidate::e:
        ppfjet_electron_E_ += (*it)->energy();
        ppfjet_electron_px_ += (*it)->px();
        ppfjet_electron_py_ += (*it)->py();
        ppfjet_electron_pz_ += (*it)->pz();
        ppfjet_electron_EcalE_ += (*it)->ecalEnergy();
        ppfjet_electron_n_++;
        continue;
      case reco::PFCandidate::mu:
        ppfjet_muon_E_ += (*it)->energy();
        ppfjet_muon_px_ += (*it)->px();
        ppfjet_muon_py_ += (*it)->py();
        ppfjet_muon_pz_ += (*it)->pz();
        ppfjet_muon_EcalE_ += (*it)->ecalEnergy();
        ppfjet_muon_n_++;
        continue;
      case reco::PFCandidate::gamma:
        ppfjet_photon_E_ += (*it)->energy();
        ppfjet_photon_px_ += (*it)->px();
        ppfjet_photon_py_ += (*it)->py();
        ppfjet_photon_pz_ += (*it)->pz();
        ppfjet_photon_EcalE_ += (*it)->ecalEnergy();
        ppfjet_photon_n_++;
        continue;
      case reco::PFCandidate::h0: {
        ppfjet_had_E_.push_back((*it)->energy());
        ppfjet_had_px_.push_back((*it)->px());
        ppfjet_had_py_.push_back((*it)->py());
        ppfjet_had_pz_.push_back((*it)->pz());
        ppfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        ppfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        ppfjet_had_id_.push_back(1);
        ppfjet_had_candtrackind_.push_back(-1);
        ppfjet_had_ntwrs_.push_back(0);
        ppfjet_had_n_++;
        break;
      }
      case reco::PFCandidate::h_HF: {
        ppfjet_had_E_.push_back((*it)->energy());
        ppfjet_had_px_.push_back((*it)->px());
        ppfjet_had_py_.push_back((*it)->py());
        ppfjet_had_pz_.push_back((*it)->pz());
        ppfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        ppfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        ppfjet_had_id_.push_back(2);
        ppfjet_had_candtrackind_.push_back(-1);
        ppfjet_had_ntwrs_.push_back(0);
        ppfjet_had_n_++;
        break;
      }
      case reco::PFCandidate::egamma_HF: {
        ppfjet_had_E_.push_back((*it)->energy());
        ppfjet_had_px_.push_back((*it)->px());
        ppfjet_had_py_.push_back((*it)->py());
        ppfjet_had_pz_.push_back((*it)->pz());
        ppfjet_had_EcalE_.push_back((*it)->ecalEnergy());
        ppfjet_had_rawHcalE_.push_back((*it)->rawHcalEnergy());
        ppfjet_had_id_.push_back(3);
        ppfjet_had_candtrackind_.push_back(-1);
        ppfjet_had_ntwrs_.push_back(0);
        ppfjet_had_n_++;
        break;
      }
    }

    float HFHAD_E = 0;
    float HFEM_E = 0;
    int maxElement = (*it)->elementsInBlocks().size();
    for (int e = 0; e < maxElement; ++e) {
      // Get elements from block
      reco::PFBlockRef blockRef = (*it)->elementsInBlocks()[e].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for (unsigned iEle = 0; iEle < elements.size(); iEle++) {
        if (elements[iEle].index() == (*it)->elementsInBlocks()[e].second) {
          if (elements[iEle].type() == reco::PFBlockElement::HCAL) {  // Element is HB or HE
            // Get cluster and hits
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            reco::PFCluster cluster = *clusterref;
            double cluster_dR = deltaR(ppfjet_eta_, ppfjet_phi_, cluster.eta(), cluster.phi());
            if (ppfjet_clusters.count(cluster_dR) == 0) {
              ppfjet_clusters[cluster_dR] = ppfjet_cluster_n_;
              ppfjet_cluster_eta_.push_back(cluster.eta());
              ppfjet_cluster_phi_.push_back(cluster.phi());
              ppfjet_cluster_dR_.push_back(cluster_dR);
              ppfjet_cluster_n_++;
            }
            int cluster_ind = ppfjet_clusters[cluster_dR];
            std::vector<std::pair<DetId, float>> hitsAndFracs = cluster.hitsAndFractions();

            // Run over hits and match
            int nHits = hitsAndFracs.size();
            for (int iHit = 0; iHit < nHits; iHit++) {
              int etaPhiPF = getEtaPhi(hitsAndFracs[iHit].first);

              for (edm::SortedCollection<HBHERecHit, edm::StrictWeakOrdering<HBHERecHit>>::const_iterator ith =
                       hbhereco->begin();
                   ith != hbhereco->end();
                   ++ith) {
                int etaPhiRecHit = getEtaPhi((*ith).id());
                if (etaPhiPF == etaPhiRecHit) {
                  ppfjet_had_ntwrs_.at(ppfjet_had_n_ - 1)++;
                  if (ppfjet_rechits.count((*ith).id()) == 0) {
                    ppfjet_twr_ieta_.push_back((*ith).id().ieta());
                    ppfjet_twr_iphi_.push_back((*ith).id().iphi());
                    ppfjet_twr_depth_.push_back((*ith).id().depth());
                    ppfjet_twr_subdet_.push_back((*ith).id().subdet());
                    ppfjet_twr_hade_.push_back((*ith).energy());
                    ppfjet_twr_frac_.push_back(hitsAndFracs[iHit].second);
                    ppfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                    ppfjet_twr_hadind_.push_back(ppfjet_had_n_ - 1);
                    ppfjet_twr_elmttype_.push_back(0);
                    ppfjet_twr_clusterind_.push_back(cluster_ind);
                    if (hasTrack) {
                      ppfjet_twr_candtrackind_.push_back(ppfjet_ncandtracks_ - 1);
                    } else {
                      ppfjet_twr_candtrackind_.push_back(-1);
                    }
                    switch ((*ith).id().subdet()) {
                      case HcalSubdetector::HcalBarrel: {
                        CaloCellGeometry::CornersVec cv = HBGeom->getCorners((*ith).id());
                        float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                        float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                        if (cv[0].phi() < cv[2].phi())
                          avgphi = (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) +
                                    static_cast<double>(cv[2].phi())) /
                                   2.0;
                        ppfjet_twr_dR_.push_back(deltaR(ppfjet_eta_, ppfjet_phi_, avgeta, avgphi));
                        break;
                      }
                      case HcalSubdetector::HcalEndcap: {
                        CaloCellGeometry::CornersVec cv = HEGeom->getCorners((*ith).id());
                        float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                        float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                        if (cv[0].phi() < cv[2].phi())
                          avgphi = (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) +
                                    static_cast<double>(cv[2].phi())) /
                                   2.0;
                        ppfjet_twr_dR_.push_back(deltaR(ppfjet_eta_, ppfjet_phi_, avgeta, avgphi));
                        break;
                      }
                      default:
                        ppfjet_twr_dR_.push_back(-1);
                        break;
                    }
                    ppfjet_rechits[(*ith).id()].first = ppfjet_ntwrs_;
                    ++ppfjet_ntwrs_;
                  } else if (ppfjet_rechits[(*ith).id()].second.count(hitsAndFracs[iHit].second) == 0) {
                    ppfjet_twr_frac_.at(ppfjet_rechits[(*ith).id()].first) += hitsAndFracs[iHit].second;
                    if (cluster_dR <
                        ppfjet_cluster_dR_.at(ppfjet_twr_clusterind_.at(ppfjet_rechits[(*ith).id()].first))) {
                      ppfjet_twr_clusterind_.at(ppfjet_rechits[(*ith).id()].first) = cluster_ind;
                    }
                    ppfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                  }
                }                                                           // Test if ieta,iphi matches
              }                                                             // Loop over rechits
            }                                                               // Loop over hits
          }                                                                 // Test if element is from HCAL
          else if (elements[iEle].type() == reco::PFBlockElement::HFHAD) {  // Element is HF
            for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator ith =
                     hfreco->begin();
                 ith != hfreco->end();
                 ++ith) {
              if ((*ith).id().depth() == 1)
                continue;  // Remove long fibers
              auto thisCell = HFGeom->getGeometry((*ith).id().rawId());
              const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();

              bool passMatch = false;
              if ((*it)->eta() < cv[0].eta() && (*it)->eta() > cv[2].eta()) {
                if ((*it)->phi() < cv[0].phi() && (*it)->phi() > cv[2].phi())
                  passMatch = true;
                else if (cv[0].phi() < cv[2].phi()) {
                  if ((*it)->phi() < cv[0].phi())
                    passMatch = true;
                  else if ((*it)->phi() > cv[2].phi())
                    passMatch = true;
                }
              }

              if (passMatch) {
                ppfjet_had_ntwrs_.at(ppfjet_had_n_ - 1)++;
                ppfjet_twr_ieta_.push_back((*ith).id().ieta());
                ppfjet_twr_iphi_.push_back((*ith).id().iphi());
                ppfjet_twr_depth_.push_back((*ith).id().depth());
                ppfjet_twr_subdet_.push_back((*ith).id().subdet());
                ppfjet_twr_hade_.push_back((*ith).energy());
                ppfjet_twr_frac_.push_back(1.0);
                ppfjet_twr_hadind_.push_back(ppfjet_had_n_ - 1);
                ppfjet_twr_elmttype_.push_back(1);
                ppfjet_twr_clusterind_.push_back(-1);
                ppfjet_twr_candtrackind_.push_back(-1);
                float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                if (cv[0].phi() < cv[2].phi())
                  avgphi =
                      (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                ppfjet_twr_dR_.push_back(deltaR(ppfjet_eta_, ppfjet_phi_, avgeta, avgphi));
                ++ppfjet_ntwrs_;
                HFHAD_E += (*ith).energy();
              }
            }
          } else if (elements[iEle].type() == reco::PFBlockElement::HFEM) {  // Element is HF
            for (edm::SortedCollection<HFRecHit, edm::StrictWeakOrdering<HFRecHit>>::const_iterator ith =
                     hfreco->begin();
                 ith != hfreco->end();
                 ++ith) {
              if ((*ith).id().depth() == 2)
                continue;  // Remove short fibers
              auto thisCell = HFGeom->getGeometry((*ith).id().rawId());
              const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();

              bool passMatch = false;
              if ((*it)->eta() < cv[0].eta() && (*it)->eta() > cv[2].eta()) {
                if ((*it)->phi() < cv[0].phi() && (*it)->phi() > cv[2].phi())
                  passMatch = true;
                else if (cv[0].phi() < cv[2].phi()) {
                  if ((*it)->phi() < cv[0].phi())
                    passMatch = true;
                  else if ((*it)->phi() > cv[2].phi())
                    passMatch = true;
                }
              }

              if (passMatch) {
                ppfjet_had_ntwrs_.at(ppfjet_had_n_ - 1)++;
                ppfjet_twr_ieta_.push_back((*ith).id().ieta());
                ppfjet_twr_iphi_.push_back((*ith).id().iphi());
                ppfjet_twr_depth_.push_back((*ith).id().depth());
                ppfjet_twr_subdet_.push_back((*ith).id().subdet());
                ppfjet_twr_hade_.push_back((*ith).energy());
                ppfjet_twr_frac_.push_back(1.0);
                ppfjet_twr_hadind_.push_back(ppfjet_had_n_ - 1);
                ppfjet_twr_elmttype_.push_back(2);
                ppfjet_twr_clusterind_.push_back(-1);
                ppfjet_twr_candtrackind_.push_back(-1);
                float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                if (cv[0].phi() < cv[2].phi())
                  avgphi =
                      (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                ppfjet_twr_dR_.push_back(deltaR(ppfjet_eta_, ppfjet_phi_, avgeta, avgphi));
                ++ppfjet_ntwrs_;
                HFEM_E += (*ith).energy();
              }
            }
          } else if (elements[iEle].type() == reco::PFBlockElement::HO) {  // Element is HO
            reco::PFClusterRef clusterref = elements[iEle].clusterRef();
            reco::PFCluster cluster = *clusterref;
            double cluster_dR = deltaR(ppfjet_eta_, ppfjet_phi_, cluster.eta(), cluster.phi());
            if (ppfjet_clusters.count(cluster_dR) == 0) {
              ppfjet_clusters[cluster_dR] = ppfjet_cluster_n_;
              ppfjet_cluster_eta_.push_back(cluster.eta());
              ppfjet_cluster_phi_.push_back(cluster.phi());
              ppfjet_cluster_dR_.push_back(cluster_dR);
              ppfjet_cluster_n_++;
            }
            int cluster_ind = ppfjet_clusters[cluster_dR];

            std::vector<std::pair<DetId, float>> hitsAndFracs = cluster.hitsAndFractions();
            int nHits = hitsAndFracs.size();
            for (int iHit = 0; iHit < nHits; iHit++) {
              int etaPhiPF = getEtaPhi(hitsAndFracs[iHit].first);

              for (edm::SortedCollection<HORecHit, edm::StrictWeakOrdering<HORecHit>>::const_iterator ith =
                       horeco->begin();
                   ith != horeco->end();
                   ++ith) {
                int etaPhiRecHit = getEtaPhi((*ith).id());
                if (etaPhiPF == etaPhiRecHit) {
                  ppfjet_had_ntwrs_.at(ppfjet_had_n_ - 1)++;
                  if (ppfjet_rechits.count((*ith).id()) == 0) {
                    ppfjet_twr_ieta_.push_back((*ith).id().ieta());
                    ppfjet_twr_iphi_.push_back((*ith).id().iphi());
                    ppfjet_twr_depth_.push_back((*ith).id().depth());
                    ppfjet_twr_subdet_.push_back((*ith).id().subdet());
                    ppfjet_twr_hade_.push_back((*ith).energy());
                    ppfjet_twr_frac_.push_back(hitsAndFracs[iHit].second);
                    ppfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                    ppfjet_twr_hadind_.push_back(ppfjet_had_n_ - 1);
                    ppfjet_twr_elmttype_.push_back(3);
                    ppfjet_twr_clusterind_.push_back(cluster_ind);
                    if (hasTrack) {
                      ppfjet_twr_candtrackind_.push_back(ppfjet_ncandtracks_ - 1);
                    } else {
                      ppfjet_twr_candtrackind_.push_back(-1);
                    }
                    auto thisCell = HOGeom->getGeometry((*ith).id().rawId());
                    const CaloCellGeometry::CornersVec& cv = thisCell->getCorners();
                    float avgeta = (cv[0].eta() + cv[2].eta()) / 2.0;
                    float avgphi = (static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) / 2.0;
                    if (cv[0].phi() < cv[2].phi())
                      avgphi =
                          (2.0 * 3.141592653 + static_cast<double>(cv[0].phi()) + static_cast<double>(cv[2].phi())) /
                          2.0;
                    ppfjet_twr_dR_.push_back(deltaR(ppfjet_eta_, ppfjet_phi_, avgeta, avgphi));
                    ppfjet_rechits[(*ith).id()].first = ppfjet_ntwrs_;
                    ++ppfjet_ntwrs_;
                  } else if (ppfjet_rechits[(*ith).id()].second.count(hitsAndFracs[iHit].second) == 0) {
                    ppfjet_twr_frac_.at(ppfjet_rechits[(*ith).id()].first) += hitsAndFracs[iHit].second;
                    if (cluster_dR <
                        ppfjet_cluster_dR_.at(ppfjet_twr_clusterind_.at(ppfjet_rechits[(*ith).id()].first))) {
                      ppfjet_twr_clusterind_.at(ppfjet_rechits[(*ith).id()].first) = cluster_ind;
                    }
                    ppfjet_rechits[(*ith).id()].second.insert(hitsAndFracs[iHit].second);
                  }
                }  // Test if ieta,iphi match
              }    // Loop over rechits
            }      // Loop over hits
          }        // Test if element is from HO
        }          // Test for right element index
      }            // Loop over elements
    }              // Loop over elements in blocks
    switch (candidateType) {
      case reco::PFCandidate::h_HF:
        ppfjet_had_emf_.push_back(HFEM_E / (HFEM_E + HFHAD_E));
        break;
      case reco::PFCandidate::egamma_HF:
        ppfjet_had_emf_.push_back(-1);
        break;
      default:
        ppfjet_had_emf_.push_back(-1);
        break;
    }
  }  // Loop over PF constitutents

  int probe_had_EcalE = 0;
  int probe_had_rawHcalE = 0;
  for (int i = 0; i < ppfjet_had_n_; i++) {
    probe_had_EcalE += ppfjet_had_EcalE_[i];
    probe_had_rawHcalE += ppfjet_had_rawHcalE_[i];
  }
  ppfjet_EMfrac_ = 1.0 - probe_had_rawHcalE / (probe_had_rawHcalE + probe_had_EcalE + ppfjet_unkown_E_ +
                                               ppfjet_electron_E_ + ppfjet_muon_E_ + ppfjet_photon_E_);
  ppfjet_hadEcalEfrac_ = probe_had_EcalE / (probe_had_rawHcalE + probe_had_EcalE + ppfjet_unkown_E_ +
                                            ppfjet_electron_E_ + ppfjet_muon_E_ + ppfjet_photon_E_);

  // fill dijet variables
  pf_dijet_deta_ = std::fabs(std::fabs(pf_tag.jet()->eta()) - std::fabs(pf_probe.jet()->eta()));
  pf_dijet_dphi_ = pf_tag.jet()->phi() - pf_probe.jet()->phi();
  if (pf_dijet_dphi_ > 3.1415)
    pf_dijet_dphi_ = 6.2832 - pf_dijet_dphi_;
  pf_dijet_balance_ = (tpfjet_pt_ - ppfjet_pt_) / (tpfjet_pt_ + ppfjet_pt_);

  tree_->Fill();

  return;
}

// ------------ method called once each job just before starting event loop  ------------
void DiJetAnalyzer::beginJob() {
  // book histograms
  rootfile_ = new TFile(rootHistFilename_.c_str(), "RECREATE");

  h_PassSelPF_ = new TH1D("h_PassSelectionPF", "Selection Pass Failures PFJets", 200, -0.5, 199.5);

  tree_ = new TTree("dijettree", "tree for dijet balancing");

  tree_->Branch("tpfjet_pt", &tpfjet_pt_, "tpfjet_pt/F");
  tree_->Branch("tpfjet_p", &tpfjet_p_, "tpfjet_p/F");
  tree_->Branch("tpfjet_E", &tpfjet_E_, "tpfjet_E/F");
  tree_->Branch("tpfjet_eta", &tpfjet_eta_, "tpfjet_eta/F");
  tree_->Branch("tpfjet_phi", &tpfjet_phi_, "tpfjet_phi/F");
  tree_->Branch("tpfjet_EMfrac", &tpfjet_EMfrac_, "tpfjet_EMfrac/F");
  tree_->Branch("tpfjet_hadEcalEfrac", &tpfjet_hadEcalEfrac_, "tpfjet_hadEcalEfrac/F");
  tree_->Branch("tpfjet_scale", &tpfjet_scale_, "tpfjet_scale/F");
  tree_->Branch("tpfjet_area", &tpfjet_area_, "tpfjet_area/F");
  tree_->Branch("tpfjet_jetID", &tpfjet_jetID_, "tpfjet_jetID/I");
  tree_->Branch("tpfjet_unkown_E", &tpfjet_unkown_E_, "tpfjet_unkown_E/F");
  tree_->Branch("tpfjet_electron_E", &tpfjet_electron_E_, "tpfjet_electron_E/F");
  tree_->Branch("tpfjet_muon_E", &tpfjet_muon_E_, "tpfjet_muon_E/F");
  tree_->Branch("tpfjet_photon_E", &tpfjet_photon_E_, "tpfjet_photon_E/F");
  tree_->Branch("tpfjet_unkown_px", &tpfjet_unkown_px_, "tpfjet_unkown_px/F");
  tree_->Branch("tpfjet_electron_px", &tpfjet_electron_px_, "tpfjet_electron_px/F");
  tree_->Branch("tpfjet_muon_px", &tpfjet_muon_px_, "tpfjet_muon_px/F");
  tree_->Branch("tpfjet_photon_px", &tpfjet_photon_px_, "tpfjet_photon_px/F");
  tree_->Branch("tpfjet_unkown_py", &tpfjet_unkown_py_, "tpfjet_unkown_py/F");
  tree_->Branch("tpfjet_electron_py", &tpfjet_electron_py_, "tpfjet_electron_py/F");
  tree_->Branch("tpfjet_muon_py", &tpfjet_muon_py_, "tpfjet_muon_py/F");
  tree_->Branch("tpfjet_photon_py", &tpfjet_photon_py_, "tpfjet_photon_py/F");
  tree_->Branch("tpfjet_unkown_pz", &tpfjet_unkown_pz_, "tpfjet_unkown_pz/F");
  tree_->Branch("tpfjet_electron_pz", &tpfjet_electron_pz_, "tpfjet_electron_pz/F");
  tree_->Branch("tpfjet_muon_pz", &tpfjet_muon_pz_, "tpfjet_muon_pz/F");
  tree_->Branch("tpfjet_photon_pz", &tpfjet_photon_pz_, "tpfjet_photon_pz/F");
  tree_->Branch("tpfjet_unkown_EcalE", &tpfjet_unkown_EcalE_, "tpfjet_unkown_EcalE/F");
  tree_->Branch("tpfjet_electron_EcalE", &tpfjet_electron_EcalE_, "tpfjet_electron_EcalE/F");
  tree_->Branch("tpfjet_muon_EcalE", &tpfjet_muon_EcalE_, "tpfjet_muon_EcalE/F");
  tree_->Branch("tpfjet_photon_EcalE", &tpfjet_photon_EcalE_, "tpfjet_photon_EcalE/F");
  tree_->Branch("tpfjet_unkown_n", &tpfjet_unkown_n_, "tpfjet_unkown_n/I");
  tree_->Branch("tpfjet_electron_n", &tpfjet_electron_n_, "tpfjet_electron_n/I");
  tree_->Branch("tpfjet_muon_n", &tpfjet_muon_n_, "tpfjet_muon_n/I");
  tree_->Branch("tpfjet_photon_n", &tpfjet_photon_n_, "tpfjet_photon_n/I");
  tree_->Branch("tpfjet_had_n", &tpfjet_had_n_, "tpfjet_had_n/I");
  tree_->Branch("tpfjet_had_E", &tpfjet_had_E_);
  tree_->Branch("tpfjet_had_px", &tpfjet_had_px_);
  tree_->Branch("tpfjet_had_py", &tpfjet_had_py_);
  tree_->Branch("tpfjet_had_pz", &tpfjet_had_pz_);
  tree_->Branch("tpfjet_had_EcalE", &tpfjet_had_EcalE_);
  tree_->Branch("tpfjet_had_rawHcalE", &tpfjet_had_rawHcalE_);
  tree_->Branch("tpfjet_had_emf", &tpfjet_had_emf_);
  tree_->Branch("tpfjet_had_id", &tpfjet_had_id_);
  tree_->Branch("tpfjet_had_candtrackind", &tpfjet_had_candtrackind_);
  tree_->Branch("tpfjet_had_ntwrs", &tpfjet_had_ntwrs_);
  tree_->Branch("tpfjet_ntwrs", &tpfjet_ntwrs_, "tpfjet_ntwrs/I");
  tree_->Branch("tpfjet_twr_ieta", &tpfjet_twr_ieta_);
  tree_->Branch("tpfjet_twr_iphi", &tpfjet_twr_iphi_);
  tree_->Branch("tpfjet_twr_depth", &tpfjet_twr_depth_);
  tree_->Branch("tpfjet_twr_subdet", &tpfjet_twr_subdet_);
  tree_->Branch("tpfjet_twr_hade", &tpfjet_twr_hade_);
  tree_->Branch("tpfjet_twr_frac", &tpfjet_twr_frac_);
  tree_->Branch("tpfjet_twr_candtrackind", &tpfjet_twr_candtrackind_);
  tree_->Branch("tpfjet_twr_hadind", &tpfjet_twr_hadind_);
  tree_->Branch("tpfjet_twr_elmttype", &tpfjet_twr_elmttype_);
  tree_->Branch("tpfjet_twr_dR", &tpfjet_twr_dR_);
  tree_->Branch("tpfjet_twr_clusterind", &tpfjet_twr_clusterind_);
  tree_->Branch("tpfjet_cluster_n", &tpfjet_cluster_n_, "tpfjet_cluster_n/I");
  tree_->Branch("tpfjet_cluster_eta", &tpfjet_cluster_eta_);
  tree_->Branch("tpfjet_cluster_phi", &tpfjet_cluster_phi_);
  tree_->Branch("tpfjet_cluster_dR", &tpfjet_cluster_dR_);
  tree_->Branch("tpfjet_ncandtracks", &tpfjet_ncandtracks_, "tpfjet_ncandtracks/I");
  tree_->Branch("tpfjet_candtrack_px", &tpfjet_candtrack_px_);
  tree_->Branch("tpfjet_candtrack_py", &tpfjet_candtrack_py_);
  tree_->Branch("tpfjet_candtrack_pz", &tpfjet_candtrack_pz_);
  tree_->Branch("tpfjet_candtrack_EcalE", &tpfjet_candtrack_EcalE_);
  tree_->Branch("ppfjet_pt", &ppfjet_pt_, "ppfjet_pt/F");
  tree_->Branch("ppfjet_p", &ppfjet_p_, "ppfjet_p/F");
  tree_->Branch("ppfjet_E", &ppfjet_E_, "ppfjet_E/F");
  tree_->Branch("ppfjet_eta", &ppfjet_eta_, "ppfjet_eta/F");
  tree_->Branch("ppfjet_phi", &ppfjet_phi_, "ppfjet_phi/F");
  tree_->Branch("ppfjet_EMfrac", &ppfjet_EMfrac_, "ppfjet_EMfrac/F");
  tree_->Branch("ppfjet_hadEcalEfrac", &ppfjet_hadEcalEfrac_, "ppfjet_hadEcalEfrac/F");
  tree_->Branch("ppfjet_scale", &ppfjet_scale_, "ppfjet_scale/F");
  tree_->Branch("ppfjet_area", &ppfjet_area_, "ppfjet_area/F");
  tree_->Branch("ppfjet_jetID", &ppfjet_jetID_, "ppfjet_jetID/I");
  tree_->Branch("ppfjet_unkown_E", &ppfjet_unkown_E_, "ppfjet_unkown_E/F");
  tree_->Branch("ppfjet_electron_E", &ppfjet_electron_E_, "ppfjet_electron_E/F");
  tree_->Branch("ppfjet_muon_E", &ppfjet_muon_E_, "ppfjet_muon_E/F");
  tree_->Branch("ppfjet_photon_E", &ppfjet_photon_E_, "ppfjet_photon_E/F");
  tree_->Branch("ppfjet_unkown_px", &ppfjet_unkown_px_, "ppfjet_unkown_px/F");
  tree_->Branch("ppfjet_electron_px", &ppfjet_electron_px_, "ppfjet_electron_px/F");
  tree_->Branch("ppfjet_muon_px", &ppfjet_muon_px_, "ppfjet_muon_px/F");
  tree_->Branch("ppfjet_photon_px", &ppfjet_photon_px_, "ppfjet_photon_px/F");
  tree_->Branch("ppfjet_unkown_py", &ppfjet_unkown_py_, "ppfjet_unkown_py/F");
  tree_->Branch("ppfjet_electron_py", &ppfjet_electron_py_, "ppfjet_electron_py/F");
  tree_->Branch("ppfjet_muon_py", &ppfjet_muon_py_, "ppfjet_muon_py/F");
  tree_->Branch("ppfjet_photon_py", &ppfjet_photon_py_, "ppfjet_photon_py/F");
  tree_->Branch("ppfjet_unkown_pz", &ppfjet_unkown_pz_, "ppfjet_unkown_pz/F");
  tree_->Branch("ppfjet_electron_pz", &ppfjet_electron_pz_, "ppfjet_electron_pz/F");
  tree_->Branch("ppfjet_muon_pz", &ppfjet_muon_pz_, "ppfjet_muon_pz/F");
  tree_->Branch("ppfjet_photon_pz", &ppfjet_photon_pz_, "ppfjet_photon_pz/F");
  tree_->Branch("ppfjet_unkown_EcalE", &ppfjet_unkown_EcalE_, "ppfjet_unkown_EcalE/F");
  tree_->Branch("ppfjet_electron_EcalE", &ppfjet_electron_EcalE_, "ppfjet_electron_EcalE/F");
  tree_->Branch("ppfjet_muon_EcalE", &ppfjet_muon_EcalE_, "ppfjet_muon_EcalE/F");
  tree_->Branch("ppfjet_photon_EcalE", &ppfjet_photon_EcalE_, "ppfjet_photon_EcalE/F");
  tree_->Branch("ppfjet_unkown_n", &ppfjet_unkown_n_, "ppfjet_unkown_n/I");
  tree_->Branch("ppfjet_electron_n", &ppfjet_electron_n_, "ppfjet_electron_n/I");
  tree_->Branch("ppfjet_muon_n", &ppfjet_muon_n_, "ppfjet_muon_n/I");
  tree_->Branch("ppfjet_photon_n", &ppfjet_photon_n_, "ppfjet_photon_n/I");
  tree_->Branch("ppfjet_had_n", &ppfjet_had_n_, "ppfjet_had_n/I");
  tree_->Branch("ppfjet_had_E", &ppfjet_had_E_);
  tree_->Branch("ppfjet_had_px", &ppfjet_had_px_);
  tree_->Branch("ppfjet_had_py", &ppfjet_had_py_);
  tree_->Branch("ppfjet_had_pz", &ppfjet_had_pz_);
  tree_->Branch("ppfjet_had_EcalE", &ppfjet_had_EcalE_);
  tree_->Branch("ppfjet_had_rawHcalE", &ppfjet_had_rawHcalE_);
  tree_->Branch("ppfjet_had_emf", &ppfjet_had_emf_);
  tree_->Branch("ppfjet_had_id", &ppfjet_had_id_);
  tree_->Branch("ppfjet_had_candtrackind", &ppfjet_had_candtrackind_);
  tree_->Branch("ppfjet_had_ntwrs", &ppfjet_had_ntwrs_);
  tree_->Branch("ppfjet_ntwrs", &ppfjet_ntwrs_, "ppfjet_ntwrs/I");
  tree_->Branch("ppfjet_twr_ieta", &ppfjet_twr_ieta_);
  tree_->Branch("ppfjet_twr_iphi", &ppfjet_twr_iphi_);
  tree_->Branch("ppfjet_twr_depth", &ppfjet_twr_depth_);
  tree_->Branch("ppfjet_twr_subdet", &ppfjet_twr_subdet_);
  tree_->Branch("ppfjet_twr_hade", &ppfjet_twr_hade_);
  tree_->Branch("ppfjet_twr_frac", &ppfjet_twr_frac_);
  tree_->Branch("ppfjet_twr_candtrackind", &ppfjet_twr_candtrackind_);
  tree_->Branch("ppfjet_twr_hadind", &ppfjet_twr_hadind_);
  tree_->Branch("ppfjet_twr_elmttype", &ppfjet_twr_elmttype_);
  tree_->Branch("ppfjet_twr_dR", &ppfjet_twr_dR_);
  tree_->Branch("ppfjet_twr_clusterind", &ppfjet_twr_clusterind_);
  tree_->Branch("ppfjet_cluster_n", &ppfjet_cluster_n_, "ppfjet_cluster_n/I");
  tree_->Branch("ppfjet_cluster_eta", &ppfjet_cluster_eta_);
  tree_->Branch("ppfjet_cluster_phi", &ppfjet_cluster_phi_);
  tree_->Branch("ppfjet_cluster_dR", &ppfjet_cluster_dR_);
  tree_->Branch("ppfjet_ncandtracks", &ppfjet_ncandtracks_, "ppfjet_ncandtracks/I");
  tree_->Branch("ppfjet_candtrack_px", &ppfjet_candtrack_px_);
  tree_->Branch("ppfjet_candtrack_py", &ppfjet_candtrack_py_);
  tree_->Branch("ppfjet_candtrack_pz", &ppfjet_candtrack_pz_);
  tree_->Branch("ppfjet_candtrack_EcalE", &ppfjet_candtrack_EcalE_);
  tree_->Branch("pf_dijet_deta", &pf_dijet_deta_, "pf_dijet_deta/F");
  tree_->Branch("pf_dijet_dphi", &pf_dijet_dphi_, "pf_dijet_dphi/F");
  tree_->Branch("pf_dijet_balance", &pf_dijet_balance_, "pf_dijet_balance/F");
  tree_->Branch("pf_thirdjet_px", &pf_thirdjet_px_, "pf_thirdjet_px/F");
  tree_->Branch("pf_thirdjet_py", &pf_thirdjet_py_, "pf_thirdjet_py/F");
  tree_->Branch("pf_realthirdjet_px", &pf_realthirdjet_px_, "pf_realthirdjet_px/F");
  tree_->Branch("pf_realthirdjet_py", &pf_realthirdjet_py_, "pf_realthirdjet_py/F");
  tree_->Branch("pf_realthirdjet_scale", &pf_realthirdjet_scale_, "pf_realthirdjet_scale/F");
  tree_->Branch("pf_Run", &pf_Run_, "pf_Run/I");
  tree_->Branch("pf_Lumi", &pf_Lumi_, "pf_Lumi/I");
  tree_->Branch("pf_Event", &pf_Event_, "pf_Event/I");
  tree_->Branch("pf_NPV", &pf_NPV_, "pf_NPV/I");

  return;
}

// ------------ method called once each job just after ending the event loop  ------------
void DiJetAnalyzer::endJob() {
  // write histograms
  rootfile_->cd();

  h_PassSelPF_->Write();
  tree_->Write();

  rootfile_->Close();
}

// helper function

double DiJetAnalyzer::deltaR(const reco::Jet* j1, const reco::Jet* j2) {
  double deta = j1->eta() - j2->eta();
  double dphi = std::fabs(j1->phi() - j2->phi());
  if (dphi > 3.1415927)
    dphi = 2 * 3.1415927 - dphi;
  return std::sqrt(deta * deta + dphi * dphi);
}

double DiJetAnalyzer::deltaR(const double eta1, const double phi1, const double eta2, const double phi2) {
  double deta = eta1 - eta2;
  double dphi = std::fabs(phi1 - phi2);
  if (dphi > 3.1415927)
    dphi = 2 * 3.1415927 - dphi;
  return std::sqrt(deta * deta + dphi * dphi);
}

int DiJetAnalyzer::getEtaPhi(const DetId id) {
  return id.rawId() & 0x3FFF;  // Get 14 least-significant digits
}

int DiJetAnalyzer::getEtaPhi(const HcalDetId id) {
  return id.rawId() & 0x3FFF;  // Get 14 least-significant digits
}

//define this as a plug-in
DEFINE_FWK_MODULE(DiJetAnalyzer);
