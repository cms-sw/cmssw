#include "DQM/GEM/interface/GEMDigiPadAnalysis.h"

using namespace std;
using namespace edm;

GEMDigiPadAnalysis::GEMDigiPadAnalysis(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
    tagPadDigiCluster_ = consumes<GEMPadDigiClusterCollection>(cfg.getParameter<edm::InputTag>("padDigiClusterInputLabel"));
    tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
    lumiScalers_ = consumes<LumiScalersCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
    nBXMin_ = cfg.getParameter<int>("bxMin");
    nBXMax_ = cfg.getParameter<int>("bxMax");
    nCLSMax_ = cfg.getParameter<int>("clsMax");
    nClusterSizeBinNum_ = cfg.getParameter<int>("ClusterSizeBinNum");
}

void GEMDigiPadAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
    desc.add<edm::InputTag>("padDigiClusterInputLabel", edm::InputTag("muonCSCDigis","MuonGEMPadDigiCluster"));
    desc.addUntracked<std::string>("runType", "online");
    desc.addUntracked<std::string>("logCategory", "GEMDigiPadAnalysis");
    desc.add<int>("bxMin", -15);
    desc.add<int>("bxMax", 15);
    desc.add<int>("clsMax", 10);
    desc.add<int>("ClusterSizeBinNum", 9);
    descriptions.add("GEMDigiPadAnalysis", desc);
}

void GEMDigiPadAnalysis::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  strFolderMain_ = "GEM/DigiPadComparison";

  mapPadDigiMatchOccPerCh_ = MEMap4Inf(this, "occ", "Corresponding Pad Digi Occupancy ", 1, -0.5, 1.5, 1, 0.5, 1.5, "Pad", "iEta");
  mapPadDigiMisMatchOccPerCh_ = MEMap4Inf(this, "occ", "Noncorresponding Pad Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Pad", "iEta");
  mapBXMisMatchPerCh_ = MEMap4Inf(this, "bx", "Noncorresponding Pad Digi Bunch Crossing", 16, 0 - 0.5, nBXMax_ + 0.5, "Bunch crossing");
  mapDigiPadMisMatchOccPerCh_ = MEMap4Inf(this, "occ", "Noncorresponding Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Digi", "iEta");
  mapDigiPadMatchOccPerCh_ = MEMap4Inf(this, "occ", "Corresponding Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Digi", "iEta");
  //mapDigiOccPerCh_ = MEMap4Inf(this, "occ", "Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Digi", "iEta");

  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);
  GenerateMEPerChamber(ibooker);
}
int GEMDigiPadAnalysis::ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) {
  return 0;
}
int GEMDigiPadAnalysis::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  return 0;
}
int GEMDigiPadAnalysis::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  
  return 0;
}

int GEMDigiPadAnalysis::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_pad_correspondence" + getNameDirLayer(key3));

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  int nNumCh = stationInfo.nNumDigi_;

  mapPadDigiMatchOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta / 2, -0.5);
  mapPadDigiMatchOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapPadDigiMatchOccPerCh_.bookND(bh, key);
  mapPadDigiMatchOccPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_pad_no_correspondence" + getNameDirLayer(key3));


  mapPadDigiMisMatchOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta / 2, -0.5);
  mapPadDigiMisMatchOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapPadDigiMisMatchOccPerCh_.bookND(bh, key);
  mapPadDigiMisMatchOccPerCh_.SetLabelForIEta(key, 2);


  bh.getBooker()->setCurrentFolder(strFolderMain_+"/bx_no_correspondence" + getNameDirLayer(key3));
  mapBXMisMatchPerCh_.bookND(bh,key);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_digi_no_correspondence" + getNameDirLayer(key3));


  mapDigiPadMisMatchOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta , -0.5);
  mapDigiPadMisMatchOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapDigiPadMisMatchOccPerCh_.bookND(bh, key);
  mapDigiPadMisMatchOccPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_digi_correspondence" + getNameDirLayer(key3));


  mapDigiPadMatchOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta , -0.5);
  mapDigiPadMatchOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapDigiPadMatchOccPerCh_.bookND(bh, key);
  mapDigiPadMatchOccPerCh_.SetLabelForIEta(key, 2);

  
  bh.getBooker()->setCurrentFolder(strFolderMain_);
  return 0;


  }



void GEMDigiPadAnalysis::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMPadDigiClusterCollection> gemPadDigiClusters;
  event.getByToken(this->tagPadDigiCluster_, gemPadDigiClusters);
  edm::Handle<GEMDigiCollection> gemDigis;
  event.getByToken(tagDigi_, gemDigis);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);



  bool isMatched_digi;
  for (auto gid : listChamberId_){
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    
    for (auto iEta : mapEtaPartition_[gid]) {
      GEMDetId eId = iEta->id();
      const auto& digis_in_det = gemDigis->get(eId);
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        //mapDigiOccPerCh_.Fill(key4Ch, d->strip(), eId.ieta());  

        isMatched_digi = false;
        for (auto it = gemPadDigiClusters->begin(); it != gemPadDigiClusters->end(); it++) {
          auto range = gemPadDigiClusters->get((*it).first);
          for (auto cluster = range.first; cluster != range.second; cluster++) {
            if (cluster->isValid()) {
              if (((*it).first).chamber()==gid.chamber() && ((*it).first).station()==gid.station() && ((*it).first).region()==gid.region() && ((*it).first).layer()==gid.layer() ){

                for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
                  if((pad * 2 == d->strip() || (pad * 2 + 1)== d->strip()) && ((*it).first).roll() == eId.ieta()){
                    
                    isMatched_digi = true;
                  }                 
                }
              }
            }
          }
        }
        if(isMatched_digi == false){
                   
          //Occupancy 2D (x-axis = digi number, y-axis = eta) for Digis that DO NOT have a correspondence to a Pad
          mapDigiPadMisMatchOccPerCh_.Fill(key4Ch, d->strip(), eId.ieta()); 
        }else{
          mapDigiPadMatchOccPerCh_.Fill(key4Ch, d->strip(), eId.ieta());
                    
        }
      }
            
    }  

      
  } 
  
  bool isMatched_pad;
  for (auto it2 = gemPadDigiClusters->begin(); it2 != gemPadDigiClusters->end(); it2++) {
    
    ME4IdsKey key4Ch{((*it2).first).region(), ((*it2).first).station(), ((*it2).first).layer(), ((*it2).first).chamber()};
    auto range2 = gemPadDigiClusters->get((*it2).first);
    for (auto cluster2 = range2.first; cluster2 != range2.second; cluster2++) {
      if (cluster2->isValid()) {
          for (auto pad2=cluster2->pads().front(); pad2 < (cluster2->pads().front() + cluster2->pads().size()); pad2++  ) {
            isMatched_pad =false;
            for (auto gid2 : listChamberId_){
              for (auto iEta2 : mapEtaPartition_[gid2]) {
                GEMDetId eId2 = iEta2->id();
                const auto& digis_in_det2 = gemDigis->get(eId2);
                for (auto d2 = digis_in_det2.first; d2 != digis_in_det2.second; ++d2) {
                  if (((*it2).first).chamber()==gid2.chamber() && ((*it2).first).station()==gid2.station() && ((*it2).first).region()==gid2.region() && ((*it2).first).layer()==gid2.layer() ){

                    if((pad2 * 2 == d2->strip() || (pad2 * 2 + 1)== d2->strip()) && ((*it2).first).roll() == eId2.ieta()){
                      isMatched_pad =true;
                    }
                  }
                }
              }
            }
            if(isMatched_pad == false){
            //Occupancy 2D (x-axis = pad number, y-axis = eta) for pads that DO NOT have a correspondence to a Digi
              mapPadDigiMisMatchOccPerCh_.Fill(key4Ch, pad2 , ((*it2).first).roll()); 
            //1D plot (x-axis = bx) for pads that DO NOT have a correspondence to a Digi
              mapBXMisMatchPerCh_.Fill(key4Ch, cluster2->bx());
            }else{
            //Occupancy 2D (x-axis = pad number, y-axis = eta) for pads that have a correspondence to a Digi  
              mapPadDigiMatchOccPerCh_.Fill(key4Ch, pad2 , ((*it2).first).roll());

            }
            

          }
        }
      }
      
    
  }




}


DEFINE_FWK_MODULE(GEMDigiPadAnalysis);
