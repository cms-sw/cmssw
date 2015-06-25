// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1RecoTreeProducer
//
/**\class L1RecoTreeProducer L1RecoTreeProducer.cc L1Trigger/L1TNtuples/src/L1RecoTreeProducer.cc

 Description: Produces tree containing reco quantities
        Merging of former L1JetRecoTree & L1EgammaRecoTree analyzers (Jim Brooke)

*/
//
// Original Author:  Anne-Catherine Le Bihan
//         Created:
// $Id: L1RecoTreeProducer.cc,v 1.17 2011/05/24 17:13:07 econte Exp $
//
//


// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// cond formats
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

// data formats
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/METReco/interface/HcalNoiseHPD.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"
#include "TF1.h"

//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoJet.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoMet.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoCluster.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoVertex.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoTrack.h"

//
// class declaration
//

class L1RecoTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1RecoTreeProducer(const edm::ParameterSet&);
  ~L1RecoTreeProducer();


private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  L1Analysis::L1AnalysisRecoJet*        jet;
  L1Analysis::L1AnalysisRecoMet*        met;
  L1Analysis::L1AnalysisRecoCluster*    superClusters;
  L1Analysis::L1AnalysisRecoCluster*    basicClusters;
  L1Analysis::L1AnalysisRecoVertex*     vertices;
  L1Analysis::L1AnalysisRecoTrack*      tracks;

  L1Analysis::L1AnalysisRecoJetDataFormat*              jet_data;
  L1Analysis::L1AnalysisRecoMetDataFormat*              met_data;
  L1Analysis::L1AnalysisRecoClusterDataFormat*          superClusters_data;
  L1Analysis::L1AnalysisRecoClusterDataFormat*          basicClusters_data;
  L1Analysis::L1AnalysisRecoVertexDataFormat*           vertices_data;
  L1Analysis::L1AnalysisRecoTrackDataFormat*            tracks_data;

private:

  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree * tree_;

  // EDM input tags
  edm::InputTag jetTag_;
  edm::InputTag jetIdTag_;
  edm::InputTag metTag_;
  edm::InputTag ebRecHitsTag_;
  edm::InputTag eeRecHitsTag_;
  edm::InputTag superClustersBarrelTag_;
  edm::InputTag superClustersEndcapTag_;
  edm::InputTag basicClustersBarrelTag_;
  edm::InputTag basicClustersEndcapTag_;
  edm::InputTag verticesTag_;
  edm::InputTag tracksTag_;
  std::string jetCorrectorServiceName_;

  // jet corrector

  // debug stuff
  bool jetsMissing_;
  double jetptThreshold_;
  unsigned int maxCl_;
  unsigned int maxJet_;
  unsigned int maxVtx_;
  unsigned int maxTrk_;
};



L1RecoTreeProducer::L1RecoTreeProducer(const edm::ParameterSet& iConfig):
  jetTag_(iConfig.getUntrackedParameter("jetTag",edm::InputTag("ak5CaloJets"))),
  jetIdTag_(iConfig.getUntrackedParameter("jetIdTag",edm::InputTag("ak5JetID"))),
  metTag_(iConfig.getUntrackedParameter("metTag",edm::InputTag("met"))),
  ebRecHitsTag_(iConfig.getUntrackedParameter("ebRecHitsTag",edm::InputTag("ecalRecHit:EcalRecHitsEB"))),
  eeRecHitsTag_(iConfig.getUntrackedParameter("eeRecHitsTag",edm::InputTag("ecalRecHit:EcalRecHitsEE"))),
  superClustersBarrelTag_(iConfig.getUntrackedParameter("superClustersBarrelTag",edm::InputTag("hybridSuperClusters"))),
  superClustersEndcapTag_(iConfig.getUntrackedParameter("superClustersEndcapTag",edm::InputTag("multi5x5SuperClusters:multi5x5EndcapSuperClusters"))),
  basicClustersBarrelTag_(iConfig.getUntrackedParameter("basicClustersBarrelTag",edm::InputTag("multi5x5BasicClusters:multi5x5BarrelBasicClusters"))),
  basicClustersEndcapTag_(iConfig.getUntrackedParameter("basicClustersEndcapTag",edm::InputTag("multi5x5BasicClusters:multi5x5EndcapBasicClusters"))),
  verticesTag_(iConfig.getUntrackedParameter("verticesTag",edm::InputTag("offlinePrimaryVertices"))),
  tracksTag_(iConfig.getUntrackedParameter("tracksTag",edm::InputTag("generalTracks"))),
  jetCorrectorServiceName_(iConfig.getUntrackedParameter<std::string>("jetCorrectorServiceName","ak5CaloL2L3Residual")),
  jetsMissing_(false)
{

  jetptThreshold_ = iConfig.getParameter<double>      ("jetptThreshold");
  maxCl_          = iConfig.getParameter<unsigned int>("maxCl");
  maxJet_         = iConfig.getParameter<unsigned int>("maxJet");
  maxVtx_         = iConfig.getParameter<unsigned int>("maxVtx");
  maxTrk_         = iConfig.getParameter<unsigned int>("maxTrk");

  jet           = new L1Analysis::L1AnalysisRecoJet();
  met           = new L1Analysis::L1AnalysisRecoMet();
  superClusters = new L1Analysis::L1AnalysisRecoCluster();
  basicClusters = new L1Analysis::L1AnalysisRecoCluster();
  vertices      = new L1Analysis::L1AnalysisRecoVertex();
  tracks        = new L1Analysis::L1AnalysisRecoTrack();

  jet_data           = jet->getData();
  met_data           = met->getData();
  superClusters_data = superClusters->getData();
  basicClusters_data = basicClusters->getData();
  vertices_data      = vertices->getData();
  tracks_data        = tracks->getData();

  // set up output
  tree_=fs_->make<TTree>("RecoTree", "RecoTree");
  tree_->Branch("Jet",           "L1Analysis::L1AnalysisRecoJetDataFormat",         &jet_data,                32000, 3);
  tree_->Branch("Met",           "L1Analysis::L1AnalysisRecoMetDataFormat",         &met_data,                32000, 3);
  tree_->Branch("SuperClusters", "L1Analysis::L1AnalysisRecoClusterDataFormat",     &superClusters_data,      32000, 3);
  tree_->Branch("BasicClusters", "L1Analysis::L1AnalysisRecoClusterDataFormat",     &basicClusters_data,      32000, 3);
  tree_->Branch("Vertices",      "L1Analysis::L1AnalysisRecoVertexDataFormat",      &vertices_data,           32000, 3);
  tree_->Branch("Tracks",        "L1Analysis::L1AnalysisRecoTrackDataFormat",       &tracks_data,             32000, 3);

}


L1RecoTreeProducer::~L1RecoTreeProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void L1RecoTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  jet->Reset();
  met->Reset();
  superClusters->Reset();
  basicClusters->Reset();
  vertices->Reset();
  tracks->Reset();

  // ES objects
  edm::ESHandle<EcalChannelStatus> ecalChStatus;
  iSetup.get<EcalChannelStatusRcd>().get(ecalChStatus);

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlv);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();


  const JetCorrector* jetCorrector = JetCorrector::getJetCorrector(jetCorrectorServiceName_,iSetup);


  // get jets  & co...
  edm::Handle<reco::CaloJetCollection> recoCaloJets;
  edm::Handle<edm::ValueMap<reco::JetID> > jetsID;


  edm::Handle<EBRecHitCollection> ebRecHits;
  edm::Handle<EERecHitCollection> eeRecHits;
  edm::Handle<reco::SuperClusterCollection> recoSuperClustersBarrel; // uncorrected ecal superclusters
  edm::Handle<reco::SuperClusterCollection> recoSuperClustersEndcap; // uncorrected ecal superclusters
  edm::Handle<reco::CaloClusterCollection> recoBasicClustersBarrel;  // ecal 5x5 basic clusters
  edm::Handle<reco::CaloClusterCollection> recoBasicClustersEndcap;  // ecal 5x5 basic clusters
  edm::Handle<reco::CaloMETCollection> recoMet; // missing transverse energy
  edm::Handle<reco::VertexCollection> recoVertices;
  edm::Handle<reco::TrackCollection>  recoTrkRef;


  iEvent.getByLabel(jetTag_, recoCaloJets);
  iEvent.getByLabel(jetIdTag_,jetsID);
  iEvent.getByLabel(ebRecHitsTag_,ebRecHits);
  iEvent.getByLabel(eeRecHitsTag_,eeRecHits);
  iEvent.getByLabel(superClustersBarrelTag_, recoSuperClustersBarrel);
  iEvent.getByLabel(superClustersEndcapTag_, recoSuperClustersEndcap);
  iEvent.getByLabel(basicClustersBarrelTag_, recoBasicClustersBarrel);
  iEvent.getByLabel(basicClustersEndcapTag_, recoBasicClustersEndcap);
  iEvent.getByLabel(metTag_, recoMet);
  iEvent.getByLabel(verticesTag_, recoVertices);
  iEvent.getByLabel(tracksTag_, recoTrkRef);


  if (recoCaloJets.isValid()) {
    jet->SetCaloJet(iEvent, iSetup, recoCaloJets, jetsID, jetCorrector, maxJet_);
    met->SetHtMht(recoCaloJets, jetptThreshold_);
  }
  else {
    if (!jetsMissing_) {edm::LogWarning("MissingProduct") << "CaloJets not found.  Branch will not be filled" << std::endl;}
    jetsMissing_ = true;
  }

  if (ebRecHits.isValid()) {
    met->SetECALFlags(ecalChStatus, ebRecHits, eeRecHits, sevLevel);
  }
  else {
    edm::LogWarning("MissingProduct") << "EB RecHits not found.  Branch will not be filled" << std::endl;
  }

  if (recoSuperClustersBarrel.isValid()) {
    reco::SuperClusterCollection theclusters (*recoSuperClustersBarrel.product());
    superClusters->Set(theclusters, maxCl_);
    }
   else {
   edm::LogWarning("MissingProduct") << "superClusters in barrel not found.  Branch will not be filled" << std::endl;
    }

  if (recoSuperClustersEndcap.isValid()) {
    reco::SuperClusterCollection theclusters (*recoSuperClustersEndcap.product());
    std::cout << "SuperClusters = "<<theclusters.size()<<std::endl;
     superClusters->Set(theclusters, maxCl_);
    }
   else {
   edm::LogWarning("MissingProduct") << "superClusters in endcap not found.  Branch will not be filled" << std::endl;
    }

  if (recoBasicClustersBarrel.isValid()) {
    reco::CaloClusterCollection theclusters (*recoBasicClustersBarrel.product());
    std::cout << "BasicClusters = "<<theclusters.size()<<std::endl;
    basicClusters->Set(theclusters, maxCl_);
   }
  else {
    edm::LogWarning("MissingProduct") << "basicClusters in barrel not found.  Branch will not be filled **" << basicClustersBarrelTag_<<std::endl;
    }


  if (recoBasicClustersEndcap.isValid()) {
    reco::CaloClusterCollection theclusters (*recoBasicClustersEndcap.product());
    std::cout << "BasicClusters = "<<theclusters.size()<<std::endl;
    basicClusters->Set(theclusters, maxCl_);
   }
  else {
    edm::LogWarning("MissingProduct") << "basicClusters in endcap not found.  Branch will not be filled"<<std::endl;
    }


  if (recoMet.isValid()) {
    met->SetMet(recoMet);
   }
   else {
     edm::LogWarning("MissingProduct") << "met not found.  Branch will not be filled"<<std::endl;
    }


  if (recoTrkRef.isValid()) {
    const reco::TrackCollection recoTracks(*recoTrkRef.product());
    tracks->SetTracks(recoTracks, maxTrk_);
   }
   else {
     edm::LogWarning("MissingProduct") << "tracks not found.  Branch will not be filled"<<std::endl;
    }


   if (recoVertices.isValid()) {
    vertices->SetVertices(recoVertices, maxVtx_);
   }
   else {
     edm::LogWarning("MissingProduct") << "vertices not found.  Branch will not be filled"<<std::endl;
    }



  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void
L1RecoTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1RecoTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1RecoTreeProducer);
