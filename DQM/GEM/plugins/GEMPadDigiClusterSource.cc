#include "DQM/GEM/interface/GEMPadDigiClusterSource.h"

using namespace std;
using namespace edm;

GEMPadDigiClusterSource::GEMPadDigiClusterSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagPadDigiCluster_ = consumes<GEMPadDigiClusterCollection>(cfg.getParameter<edm::InputTag>("padDigiClusterInputLabel"));
  lumiScalers_ = consumes<LumiScalersCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
  nBXMin_ = cfg.getParameter<int>("bxMin");
  nBXMax_ = cfg.getParameter<int>("bxMax");
  nCLSMax_ = cfg.getParameter<int>("clsMax");
  nClusterSizeBinNum_ = cfg.getParameter<int>("ClusterSizeBinNum");
}

void GEMPadDigiClusterSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("padDigiClusterInputLabel", edm::InputTag("muonCSCDigis","MuonGEMPadDigiCluster"));
  desc.addUntracked<std::string>("runType", "online");
  desc.addUntracked<std::string>("logCategory", "GEMPadDigiClusterSource");
  desc.add<int>("bxMin", -15);
  desc.add<int>("bxMax", 15);
  desc.add<int>("clsMax", 10);
  desc.add<int>("ClusterSizeBinNum", 9);
  descriptions.add("GEMPadDigiClusterSource", desc);
}

void GEMPadDigiClusterSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  strFolderMain_ = "GEM/PadDigiCluster";

  fRadiusMin_ = 120.0;
  fRadiusMax_ = 250.0;
  float radS = -5.0 / 180 * M_PI;
  float radL = 355.0 / 180 * M_PI;

  mapTotalDigi_layer_ = MEMap3Inf(this, "occ", "Digi Occupancy", 36, 0.5, 36.5, 24, -0.5, 24 - 0.5, "Chamber", "VFAT");
  mapDigiWheel_layer_ = MEMap3Inf(
      this, "occ_rphi", "Digi R-Phi Occupancy", 360, radS, radL, 8, fRadiusMin_, fRadiusMax_, "#phi (rad)", "R [cm]");
  mapDigiOcc_ieta_ = MEMap3Inf(this, "occ_ieta", "Digi iEta Occupancy", 8, 0.5, 8.5, "iEta", "Number of fired digis");
  mapDigiOcc_phi_ =
      MEMap3Inf(this, "occ_phi", "Digi Phi Occupancy", 72, -5, 355, "#phi (degree)", "Number of fired digis");
  mapTotalDigiPerEvtLayer_ = MEMap3Inf(this,
                                       "digis_per_layer",
                                       "Total number of digis per event for each layers",
                                       50,
                                       -0.5,
                                       99.5,
                                       "Number of fired digis",
                                       "Events");
  mapTotalDigiPerEvtLayer_.SetNoUnderOverflowBin();
  mapTotalDigiPerEvtIEta_ = MEMap3Inf(this,
                                      "digis_per_ieta",
                                      "Total number of digis per event for each eta partitions",
                                      50,
                                      -0.5,
                                      99.5,
                                      "Number of fired digis",
                                      "Events");
  mapTotalDigiPerEvtIEta_.SetNoUnderOverflowBin();

  mapBX_ = MEMap2Inf(this, "bx", "Digi Bunch Crossing", 21, nBXMin_ - 0.5, nBXMax_ + 0.5, "Bunch crossing");

  mapPadDiffPerCh_=MEMap3Inf(this, "occ", "Pad Digi Difference", 41,  - 0.5, 40 + 0.5, "Pad Digi Difference");
  mapBXDiffPerCh_=MEMap3Inf(this, "bx", "BX Difference", 10,  - 0.5, 10 + 0.5, "BX Difference");
  mapPadBXDiffPerCh_ = MEMap3Inf(this,"delta_pad_bx","Pad difference over time difference",41,  - 0.5, 40 + 0.5,10,  - 0.5, 10 + 0.5, "Delta Pads","Delta BX");

  mapPadDigiOccPerCh_ = MEMap4Inf(this, "occ", "Pad Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Pads", "iEta");
  mapPadBxPerCh_ = MEMap4Inf(this, "bx", "GEM Pads Hits in Time", 1536, 0.5, 1536.5, 15, -0.5, 15 - 0.5, "Pads", "Time Bins");
  mapPadCLSPerCh_= MEMap4Inf(this, "cls", "Cluster size of Pad Digi", nCLSMax_, 0.5, nCLSMax_ + 0.5, 1, 0.5, 1.5, "Cluster size", "iEta");
  if (nRunType_ == GEMDQM_RUNTYPE_OFFLINE) {
    mapDigiWheel_layer_.TurnOff();
    mapBX_.TurnOff();
  }

  if (nRunType_ == GEMDQM_RUNTYPE_RELVAL) {
    mapDigiWheel_layer_.TurnOff();
    mapPadDigiOccPerCh_.TurnOff();
    mapTotalDigi_layer_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS && nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
    mapDigiOcc_ieta_.TurnOff();
    mapDigiOcc_phi_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS) {
    mapTotalDigiPerEvtLayer_.TurnOff();
    mapTotalDigiPerEvtIEta_.TurnOff();
  }

  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);
  GenerateMEPerChamber(ibooker);
}

int GEMPadDigiClusterSource::ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) {
  mapBX_.bookND(bh, key);

 

  return 0;
}

int GEMPadDigiClusterSource::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  mapTotalDigiPerEvtIEta_.bookND(bh, key);

  return 0;
}

int GEMPadDigiClusterSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  //int nNumCh = stationInfo.nNumDigi_;
  
  





  mapTotalDigi_layer_.SetBinConfX(stationInfo.nNumChambers_);
  mapTotalDigi_layer_.SetBinConfY(stationInfo.nMaxVFAT_, -0.5);
  mapTotalDigi_layer_.bookND(bh, key);
  mapTotalDigi_layer_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);
  mapTotalDigi_layer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapDigiWheel_layer_.SetBinLowEdgeX(stationInfo.fMinPhi_);
  mapDigiWheel_layer_.SetBinHighEdgeX(stationInfo.fMinPhi_ + 2 * M_PI);
  mapDigiWheel_layer_.SetNbinsX(nNumVFATPerEta * stationInfo.nNumChambers_);
  mapDigiWheel_layer_.SetNbinsY(stationInfo.nNumEtaPartitions_);
  mapDigiWheel_layer_.bookND(bh, key);

  mapDigiOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapDigiOcc_ieta_.bookND(bh, key);
  mapDigiOcc_ieta_.SetLabelForIEta(key, 1);

  mapDigiOcc_phi_.SetBinLowEdgeX(stationInfo.fMinPhi_ * 180 / M_PI);
  mapDigiOcc_phi_.SetBinHighEdgeX(stationInfo.fMinPhi_ * 180 / M_PI + 360);
  mapDigiOcc_phi_.bookND(bh, key);
  mapTotalDigiPerEvtLayer_.bookND(bh, key);

  

  return 0;
}
int GEMPadDigiClusterSource::ProcessWithMEMap2WithChamber(BookingHelper& bh, ME3IdsKey key) {
  //ME2IdsKey key2 = key3Tokey2(key);
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/Pad_Difference_");

  mapPadDiffPerCh_.bookND(bh,key);
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/BX_Difference_");

  mapBXDiffPerCh_.bookND(bh,key);
  
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/Pad_BX_Difference_");
  mapPadBXDiffPerCh_.bookND(bh,key);
  bh.getBooker()->setCurrentFolder(strFolderMain_);
   
  return 0;
}
int GEMPadDigiClusterSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_" + getNameDirLayer(key3));

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  int nNumCh = stationInfo.nNumDigi_;

  mapPadDigiOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta / 2, -0.5);
  mapPadDigiOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapPadDigiOccPerCh_.bookND(bh, key);
  mapPadDigiOccPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/bx_" + getNameDirLayer(key3));
  mapPadBxPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta * stationInfo.nNumEtaPartitions_/ 2, -0.5);
  mapPadBxPerCh_.bookND(bh, key);
  
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/cluster size_" + getNameDirLayer(key3));

  mapPadCLSPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapPadCLSPerCh_.bookND(bh, key);
  mapPadCLSPerCh_.SetLabelForIEta(key, 2);


  bh.getBooker()->setCurrentFolder(strFolderMain_);

  return 0;
}

void GEMPadDigiClusterSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMPadDigiClusterCollection> gemPadDigiClusters;
  event.getByToken(this->tagPadDigiCluster_, gemPadDigiClusters);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);


  for (auto it = gemPadDigiClusters->begin(); it != gemPadDigiClusters->end(); it++) {
    auto range = gemPadDigiClusters->get((*it).first);
    //const int type = ((*it).first).station() - 1;
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {
        // ignore data clusters in BX's other than BX0
       // if (usegemPadDigiClustersOnlyInBX0_ and cluster->bx() != 0)
       //   continue;
       /* std::cout << "Cluster front: " << cluster->pads().front() << " , Cluster size:  "<< cluster->pads().size() 
        << " , Cluster BX: " <<  cluster->bx() << " , Cluster station: " << ((*it).first).station()
        << " , Cluster chamber: " << ((*it).first).chamber() << " , Cluster layer: " << ((*it).first).layer()
        << " , Cluster eta: " << ((*it).first).roll()<< std::endl;*/
       
        ME4IdsKey key4Ch{((*it).first).region(), ((*it).first).station(), ((*it).first).layer(), ((*it).first).chamber()};
        ME3IdsKey key3Ch{((*it).first).region(), ((*it).first).station(),((*it).first).chamber()};  

        for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
          mapPadDigiOccPerCh_.Fill(key4Ch, pad , ((*it).first).roll());
          mapPadBxPerCh_.Fill(key4Ch, pad + (192 * (8 - ((*it).first).roll())), cluster->bx());
          //std::cout << "pad:"<< pad << "  BX:"<< cluster->bx() <<"  chamber:"<< ((*it).first).chamber() << "  layer:"<< ((*it).first).layer()<<std::endl;
          
          
        }

        for (auto it2 = gemPadDigiClusters->begin(); it2 != gemPadDigiClusters->end(); it2++) {
          auto range2 = gemPadDigiClusters->get((*it2).first);
          for (auto cluster2 = range2.first; cluster2 != range2.second; cluster2++) {
            if (cluster2->isValid()) {
                 if (((*it).first).chamber()==((*it2).first).chamber() && ((*it).first).station()==((*it2).first).station() && ((*it).first).region()==((*it2).first).region() && ((*it).first).layer()==1 &&((*it2).first).layer()==2 ){
                    if(abs(cluster->bx() - cluster2->bx()) <10 ){
                      mapBXDiffPerCh_.Fill(key3Ch, abs(cluster->bx() - cluster2->bx()));
                    }
                    for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
                      for (auto pad2=cluster2->pads().front(); pad2 < (cluster2->pads().front() + cluster2->pads().size()); pad2++  ){
                        if (abs(pad - pad2) < 40 ){
                          mapPadDiffPerCh_.Fill(key3Ch, abs(pad-pad2));
                          mapPadBXDiffPerCh_.Fill(key3Ch,abs(pad-pad2), abs(cluster->bx() - cluster2->bx()));
                        }
                      }
                    }
                  } 
            }
          }
        }
        Int_t nCLS = cluster->pads().size();
        Int_t nCLSCutOff = std::min(nCLS, nCLSMax_); 
        mapPadCLSPerCh_.Fill(key4Ch, nCLSCutOff, ((*it).first).roll() );
        //std::cout << "Cluster Size:"<< cluster->pads().size() << "  Eta:"<< ((*it).first).roll()  <<std::endl;
        
      }
    }

  }
 
/*
  std::map<ME3IdsKey, Int_t> total_digi_layer;
  std::map<ME3IdsKey, Int_t> total_digi_eta;
  for (auto gid : listChamberId_) {
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    std::map<Int_t, bool> bTagVFAT;
    bTagVFAT.clear();
    MEStationInfo& stationInfo = mapStationInfo_[key3];
    const BoundPlane& surface = GEMGeometry_->idToDet(gid)->surface();
    if (total_digi_layer.find(key3) == total_digi_layer.end())
      total_digi_layer[key3] = 0;
    for (auto iEta : mapEtaPartition_[gid]) {
      GEMDetId eId = iEta->id();
      ME3IdsKey key3IEta{gid.region(), gid.station(), eId.ieta()};
      if (total_digi_eta.find(key3IEta) == total_digi_eta.end())
        total_digi_eta[key3IEta] = 0;
      const auto& digis_in_det = gemDigis->get(eId);
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        // Filling of digi occupancy
        Int_t nIdxVFAT = getVFATNumberByDigi(gid.station(), eId.ieta(), d->strip());
        mapTotalDigi_layer_.Fill(key3, gid.chamber(), nIdxVFAT);

        // Filling of digi
        mapDigiOcc_ieta_.Fill(key3, eId.ieta());  // Eta (partition)

        GlobalPoint digi_global_pos = surface.toGlobal(iEta->centreOfStrip(d->strip()));
        Float_t fPhi = (Float_t)digi_global_pos.phi();
        Float_t fPhiShift = restrictAngle(fPhi, stationInfo.fMinPhi_);
        Float_t fPhiDeg = fPhiShift * 180.0 / M_PI;
        mapDigiOcc_phi_.Fill(key3, fPhiDeg);  // Phi

        // Filling of R-Phi occupancy
        Float_t fR = fRadiusMin_ + (fRadiusMax_ - fRadiusMin_) * (eId.ieta() - 0.5) / stationInfo.nNumEtaPartitions_;
        mapDigiWheel_layer_.Fill(key3, fPhiShift, fR);

        mapDigiOccPerCh_.Fill(key4Ch, d->strip(), eId.ieta());  // Per chamber ***********************

        // For total digis
        total_digi_layer[key3]++;
        total_digi_eta[key3IEta]++;

        // Filling of bx
        Int_t nBX = std::min(std::max((Int_t)d->bx(), nBXMin_), nBXMax_);  // For under/overflow
        if (bTagVFAT.find(nIdxVFAT) == bTagVFAT.end()) {
          mapBX_.Fill(key2, nBX);
        }

        bTagVFAT[nIdxVFAT] = true;
      }
    }
  }
  for (auto [key, num_total_digi] : total_digi_layer)
    mapTotalDigiPerEvtLayer_.Fill(key, num_total_digi);
  for (auto [key, num_total_digi] : total_digi_eta)
    mapTotalDigiPerEvtIEta_.Fill(key, num_total_digi);*/
}

DEFINE_FWK_MODULE(GEMPadDigiClusterSource);
