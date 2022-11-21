// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "JetMETCorrections/MinBias/interface/MinBias.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

namespace cms {
  MinBias::MinBias(const edm::ParameterSet& iConfig)
      : caloGeometryESToken_(esConsumes<edm::Transition::BeginRun>()), geo_(nullptr) {
    // get names of modules, producing object collections
    hbheLabel_ = iConfig.getParameter<std::string>("hbheInput");
    hoLabel_ = iConfig.getParameter<std::string>("hoInput");
    hfLabel_ = iConfig.getParameter<std::string>("hfInput");
    hbheToken_ = mayConsume<HBHERecHitCollection>(edm::InputTag(hbheLabel_));
    hoToken_ = mayConsume<HORecHitCollection>(edm::InputTag(hoLabel_));
    hfToken_ = mayConsume<HFRecHitCollection>(edm::InputTag(hfLabel_));
    allowMissingInputs_ = iConfig.getUntrackedParameter<bool>("AllowMissingInputs", false);
  }

  void MinBias::beginJob() {
    edm::Service<TFileService> fs;
    myTree_ = fs->make<TTree>("RecJet", "RecJet Tree");
    myTree_->Branch("mydet", &mydet, "mydet/I");
    myTree_->Branch("mysubd", &mysubd, "mysubd/I");
    myTree_->Branch("depth", &depth, "depth/I");
    myTree_->Branch("ieta", &ieta, "ieta/I");
    myTree_->Branch("iphi", &iphi, "iphi/I");
    myTree_->Branch("eta", &eta, "eta/F");
    myTree_->Branch("phi", &phi, "phi/F");
    myTree_->Branch("mom1", &mom1, "mom1/F");
    myTree_->Branch("mom2", &mom2, "mom2/F");
    myTree_->Branch("mom3", &mom3, "mom3/F");
    myTree_->Branch("mom4", &mom4, "mom4/F");
  }

  void MinBias::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {
    geo_ = &iSetup.getData(caloGeometryESToken_);
    std::vector<DetId> did = geo_->getValidDetIds();

    for (auto const& id : did) {
      if ((id).det() == DetId::Hcal) {
        theFillDetMap0_[id] = 0.;
        theFillDetMap1_[id] = 0.;
        theFillDetMap2_[id] = 0.;
        theFillDetMap3_[id] = 0.;
        theFillDetMap4_[id] = 0.;
      }
    }
  }

  void MinBias::endRun(edm::Run const&, edm::EventSetup const& iSetup) {}

  void MinBias::endJob() {
    const HcalGeometry* hgeo = static_cast<const HcalGeometry*>(geo_->getSubdetectorGeometry(DetId::Hcal, 1));
    const std::vector<DetId>& did = hgeo->getValidDetIds();
    int i = 0;
    for (const auto& id : did) {
      //      if( id.det() == DetId::Hcal ) {
      GlobalPoint pos = hgeo->getPosition(id);
      mydet = (int)(id.det());
      mysubd = (id.subdetId());
      depth = HcalDetId(id).depth();
      ieta = HcalDetId(id).ieta();
      iphi = HcalDetId(id).iphi();
      phi = pos.phi();
      eta = pos.eta();
      if (theFillDetMap0_[id] > 0.) {
        mom1 = theFillDetMap1_[id] / theFillDetMap0_[id];
        mom2 = theFillDetMap2_[id] / theFillDetMap0_[id] - (mom1 * mom1);
        mom3 = theFillDetMap3_[id] / theFillDetMap0_[id] - 3. * mom1 * theFillDetMap2_[id] / theFillDetMap0_[id] +
               2. * pow(mom2, 3);
        mom4 = (theFillDetMap4_[id] - 4. * mom1 * theFillDetMap3_[id] + 6. * pow(mom1, 2) * theFillDetMap2_[id]) /
                   theFillDetMap0_[id] -
               3. * pow(mom1, 4);

      } else {
        mom1 = 0.;
        mom2 = 0.;
        mom3 = 0.;
        mom4 = 0.;
      }
      edm::LogWarning("MinBias") << " Detector " << id.rawId() << " mydet " << mydet << " " << mysubd << " " << depth
                                 << " " << ieta << " " << iphi << " " << pos.eta() << " " << pos.phi();
      edm::LogWarning("MinBias") << " Energy " << mom1 << " " << mom2 << std::endl;
      myTree_->Fill();
      i++;
      //      }
    }
    edm::LogWarning("MinBias") << " The number of CaloDet records " << did.size();
    edm::LogWarning("MinBias") << " The number of Hcal records " << i << std::endl;

    /*
  std::cout << "===== Start writing user histograms =====" << std::endl;
  hOutputFile->SetCompressionLevel(2);
  hOutputFile->cd();
  myTree->Write();
  hOutputFile->Close() ;
  std::cout << "===== End writing user histograms =======" << std::endl;
  */
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void MinBias::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
    if (!hbheLabel_.empty()) {
      edm::Handle<HBHERecHitCollection> hbhe;
      iEvent.getByToken(hbheToken_, hbhe);
      if (!hbhe.isValid()) {
        // can't find it!
        if (!allowMissingInputs_) {
          *hbhe;  // will throw the proper exception
        }
      } else {
        for (auto const& hbheItr : (HBHERecHitCollection)(*hbhe)) {
          DetId id = (hbheItr).detid();
          if (hbheItr.energy() > 0.)
            edm::LogWarning("MinBias") << " Energy = " << hbheItr.energy();
          theFillDetMap0_[id] += 1.;
          theFillDetMap1_[id] += hbheItr.energy();
          theFillDetMap2_[id] += pow(hbheItr.energy(), 2);
          theFillDetMap3_[id] += pow(hbheItr.energy(), 3);
          theFillDetMap4_[id] += pow(hbheItr.energy(), 4);
        }
      }
    }

    if (!hoLabel_.empty()) {
      edm::Handle<HORecHitCollection> ho;
      iEvent.getByToken(hoToken_, ho);
      if (!ho.isValid()) {
        // can't find it!
        if (!allowMissingInputs_) {
          *ho;  // will throw the proper exception
        }
      } else {
        for (auto const& hoItr : (HORecHitCollection)(*ho)) {
          DetId id = hoItr.detid();
          theFillDetMap0_[id] += 1.;
          theFillDetMap1_[id] += hoItr.energy();
          theFillDetMap2_[id] += pow(hoItr.energy(), 2);
          theFillDetMap3_[id] += pow(hoItr.energy(), 3);
          theFillDetMap4_[id] += pow(hoItr.energy(), 4);
        }
      }
    }

    if (!hfLabel_.empty()) {
      edm::Handle<HFRecHitCollection> hf;
      iEvent.getByToken(hfToken_, hf);
      if (!hf.isValid()) {
        // can't find it!
        if (!allowMissingInputs_) {
          *hf;  // will throw the proper exception
        }
      } else {
        for (auto const hfItr : (HFRecHitCollection)(*hf)) {
          DetId id = hfItr.detid();
          theFillDetMap0_[id] += 1.;
          theFillDetMap1_[id] += hfItr.energy();
          theFillDetMap2_[id] += pow(hfItr.energy(), 2);
          theFillDetMap3_[id] += pow(hfItr.energy(), 3);
          theFillDetMap4_[id] += pow(hfItr.energy(), 4);
        }
      }
    }
  }
}  // namespace cms
