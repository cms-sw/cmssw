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
  desc.add<int>("clsMax", 9);
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
  
  mapPadBXDiffPerCh_ = MEMap3Inf(this,"delta_pad_bx","Pad and BX Difference of closed Pads ",81, -40 - 0.5, 40 + 0.5,21,  -10 - 0.5, 10 + 0.5, "Delta Pads","Delta BX");
  mapBXCLSPerCh_ = MEMap4Inf(this, "bx", "Pad Bunch Crossing Per Cluster", 14 , -0.5, 13.5, "Bunch crossing");
  mapPadDigiOccPerCh_ = MEMap4Inf(this, "occ", "Pad Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Pads", "iEta");
  mapPadBxPerCh_ = MEMap4Inf(this, "pad", "GEM Pads Hits in Time", 1536, 0.5, 1536.5, 15, -0.5, 15 - 0.5, "Pads", "Time Bins");
  mapPadCLSPerCh_= MEMap4Inf(this, "cls", "Cluster size of Pad Digi", 9, 0.5, 9 + 0.5, 1, 0.5, 1.5, "Cluster Size", "iEta");
  
  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);
  GenerateMEPerChamber(ibooker);
}

int GEMPadDigiClusterSource::ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) {
  return 0;
}

int GEMPadDigiClusterSource::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  return 0;
}

int GEMPadDigiClusterSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  return 0;
}
int GEMPadDigiClusterSource::ProcessWithMEMap2WithChamber(BookingHelper& bh, ME3IdsKey key) {
    
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/pad_bx_difference_");
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

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/pads in time_" + getNameDirLayer(key3));
  mapPadBxPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta * stationInfo.nNumEtaPartitions_/ 2, -0.5);
  mapPadBxPerCh_.bookND(bh, key);
  
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/cluster size_" + getNameDirLayer(key3));

  mapPadCLSPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapPadCLSPerCh_.bookND(bh, key);
  mapPadCLSPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/bx_cluster" + getNameDirLayer(key3));
  mapBXCLSPerCh_ .bookND(bh,key);
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
      
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {

        ME4IdsKey key4Ch{((*it).first).region(), ((*it).first).station(), ((*it).first).layer(), ((*it).first).chamber()};
        ME4IdsKey key4EtaCh{((*it).first).region(), ((*it).first).station(),((*it).first).roll(),((*it).first).chamber()};
        ME3IdsKey key3Ch{((*it).first).region(), ((*it).first).station(),((*it).first).chamber()};  
        //Plot the bx of each cluster.
        mapBXCLSPerCh_.Fill(key4Ch,cluster->bx());
        for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
          mapPadDigiOccPerCh_.Fill(key4Ch, pad , ((*it).first).roll());
          mapPadBxPerCh_.Fill(key4Ch, pad + (192 * (8 - ((*it).first).roll())), cluster->bx());
      
          for (auto it2 = gemPadDigiClusters->begin(); it2 != gemPadDigiClusters->end(); it2++) {
            auto range2 = gemPadDigiClusters->get((*it2).first);
            for (auto cluster2 = range2.first; cluster2 != range2.second; cluster2++) {
              if (cluster2->isValid()) {
             
             //Plot the pad differnce and bx difference for closed pads per chamber.
                if (((*it).first).chamber()==((*it2).first).chamber() && ((*it).first).station()==((*it2).first).station() && ((*it).first).region()==((*it2).first).region() && ((*it).first).layer()==1 &&((*it2).first).layer()==2 ){
                    
                  for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
                    for (auto pad2=cluster2->pads().front(); pad2 < (cluster2->pads().front() + cluster2->pads().size()); pad2++  ){
                      if (abs(pad - pad2) < 40 && abs(((*it).first).roll() - ((*it2).first).roll())<=1){
                          mapPadBXDiffPerCh_.Fill(key3Ch, pad2-pad,  cluster2->bx()-cluster->bx() );
                      }
                    }
                  }
                } 
              }
            }
          }
          //Plot the size of clusters for each chamber and layer 
          Int_t nCLS = cluster->pads().size();
          Int_t nCLSCutOff = std::min(nCLS, nCLSMax_); 
          mapPadCLSPerCh_.Fill(key4Ch, nCLSCutOff, ((*it).first).roll() );
        
        }
      }  
    }
  }
}

DEFINE_FWK_MODULE(GEMPadDigiClusterSource);

