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
  
  mapPadBXDiffPerCh_ = MEMap3Inf(this,"delta_pad_bx","Difference of Cluster Pad Number and BX for ",81, -40 - 0.5, 40 + 0.5,21,  -10 - 0.5, 10 + 0.5, "Lay2 - Lay1 cluster central pad","Lay2 - Lay1 cluster BX");
  mapPadDiffPerCh_ = MEMap3Inf(this,"delta_pad","Difference of Cluster Pad Number for ",81, -40 - 0.5, 40 + 0.5, "Lay2 - Lay1 cluster central pad");
  mapBXDiffPerCh_ = MEMap3Inf(this,"delta_bx","Difference of Cluster BX for",21,  -10 - 0.5, 10 + 0.5, "Lay2 - Lay1 cluster BX");

  mapBXCLSPerCh_ = MEMap4Inf(this, "bx", "Cluster Bunch Crossing ", 14 , -0.5, 13.5, "Bunch crossing");
  mapPadDigiOccPerCh_ = MEMap4Inf(this, "occ", "Pad Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Pad number", "i#eta");
  mapPadBxPerCh_ = MEMap4Inf(this, "pad", "Cluster BX and Pad number for ", 1536, 0.5, 1536.5, 15, -0.5, 15 - 0.5, "Pad number", "Cluster BX");
  mapPadCLSPerCh_= MEMap4Inf(this, "cls", "Cluster Size of Pad ", 9, 0.5, 9 + 0.5, 1, 0.5, 1.5, "Cluster Size", "i#eta");
  
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
    
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/pad_bx_difference");
  mapPadBXDiffPerCh_.bookND(bh,key);
  bh.getBooker()->setCurrentFolder(strFolderMain_);
   
  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/pad_difference");
  mapPadDiffPerCh_.bookND(bh,key);
  bh.getBooker()->setCurrentFolder(strFolderMain_);

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/bx_difference");
  mapBXDiffPerCh_.bookND(bh,key);
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

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/bx_cluster_" + getNameDirLayer(key3));
  mapBXCLSPerCh_ .bookND(bh,key);
  bh.getBooker()->setCurrentFolder(strFolderMain_);
  return 0;
}

void GEMPadDigiClusterSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMPadDigiClusterCollection> gemPadDigiClusters;
  event.getByToken(this->tagPadDigiCluster_, gemPadDigiClusters);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);

  std::vector<std::vector<int>> pad_bx_layer1;
  std::vector<std::vector<int>> pad_bx_layer2;
  int med_pad;

  for (auto it = gemPadDigiClusters->begin(); it != gemPadDigiClusters->end(); it++) {
    auto range = gemPadDigiClusters->get((*it).first);
      
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {

        ME4IdsKey key4Ch{((*it).first).region(), ((*it).first).station(), ((*it).first).layer(), ((*it).first).chamber()};
        //Plot the bx of each cluster.  
        mapBXCLSPerCh_.Fill(key4Ch,cluster->bx());
      }
    } 
  }    

    for (auto it = gemPadDigiClusters->begin(); it != gemPadDigiClusters->end(); it++) {
      auto range = gemPadDigiClusters->get((*it).first);
      
        for (auto cluster = range.first; cluster != range.second; cluster++) {
          if (cluster->isValid()) { 
            ME4IdsKey key4Ch{((*it).first).region(), ((*it).first).station(), ((*it).first).layer(), ((*it).first).chamber()};
    
            for (auto pad=cluster->pads().front(); pad < (cluster->pads().front() + cluster->pads().size()); pad++  ) {
              //Plot of pad and bx for each chamber and layer
              mapPadDigiOccPerCh_.Fill(key4Ch, pad , ((*it).first).roll());
              mapPadBxPerCh_.Fill(key4Ch, pad + (192 * (8 - ((*it).first).roll())), cluster->bx());

              //Plot the size of clusters for each chamber and layer 
              Int_t nCLS = cluster->pads().size();
              Int_t nCLSCutOff = std::min(nCLS, nCLSMax_); 
              mapPadCLSPerCh_.Fill(key4Ch, nCLSCutOff, ((*it).first).roll() );
            }
          }
        }
    }  


  for (auto it = gemPadDigiClusters->begin(); it != gemPadDigiClusters->end(); it++) {
    auto range = gemPadDigiClusters->get((*it).first);
    
    for (auto cluster = range.first; cluster != range.second; cluster++) {
      if (cluster->isValid()) {    
        med_pad = cluster->pads().front() + floor(cluster->pads().size()/2);
        //push_back the pad and bx information for different layers.
        if(((*it).first).layer() == 1){
          pad_bx_layer1.push_back({med_pad , cluster->bx(), ((*it).first).region(), ((*it).first).station(), 1, ((*it).first).chamber(), ((*it).first).roll()});
        }  
        if(((*it).first).layer() == 2){
          pad_bx_layer2.push_back({med_pad, cluster->bx(),((*it).first).region(), ((*it).first).station(), 2,((*it).first).chamber(), ((*it).first).roll()});
        }
      }
    }
  }   

  for (unsigned i = 0; i < pad_bx_layer1.size(); i++){  

    ME3IdsKey key3Ch{pad_bx_layer1[i][2], pad_bx_layer1[i][3],pad_bx_layer1[i][5]}; 

    for (unsigned j = 0; j < pad_bx_layer2.size(); j++){
      if(pad_bx_layer1[i][2]==pad_bx_layer2[j][2] && pad_bx_layer1[i][3]==pad_bx_layer2[j][3] && pad_bx_layer1[i][5]==pad_bx_layer2[j][5]){
             
        //Plot the pad differnce and bx difference for closed pads per chamber.
        if (abs(pad_bx_layer1[i][0] - pad_bx_layer2[j][0]) <= 40 && abs(pad_bx_layer1[i][6] - pad_bx_layer2[j][6])<=1){
          mapPadBXDiffPerCh_.Fill(key3Ch, pad_bx_layer1[i][0] - pad_bx_layer2[j][0],  pad_bx_layer1[i][1] - pad_bx_layer2[j][1] );
          mapPadDiffPerCh_.Fill(key3Ch, pad_bx_layer1[i][0] - pad_bx_layer2[j][0] );
          mapBXDiffPerCh_.Fill(key3Ch, pad_bx_layer1[i][1] - pad_bx_layer2[j][1] );
        }
      }
    }
  }
  pad_bx_layer1.clear();
  pad_bx_layer2.clear(); 
}
          
        
        

DEFINE_FWK_MODULE(GEMPadDigiClusterSource);

