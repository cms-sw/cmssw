// -*- C++ -*-
//
// Package:    BlockAnalyzer
// Class:      BlockAnalyzer
//
/**\class ElectronAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Benedetti

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>

//
// class decleration
//

using namespace edm;
using namespace reco;
using namespace std;
class BlockAnalyzer : public edm::EDAnalyzer {
public:
  explicit BlockAnalyzer(const edm::ParameterSet&);
  ~BlockAnalyzer() override;

private:
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  double InvMass(const vector<TLorentzVector>& par);

  ParameterSet conf_;

  std::string outputfile_;

  TFile* tf1;
  TTree* s;

  //TTree
  float pt_, eta_, phi_;

  unsigned int ev;
  bool debug_;

  // ----------member data ---------------------------

  edm::InputTag pfblocks_;
  edm::InputTag trackCollection_;
  edm::InputTag primVtxLabel_;
};

//, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BlockAnalyzer::BlockAnalyzer(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      pfblocks_(iConfig.getParameter<edm::InputTag>("blockCollection")),
      trackCollection_(iConfig.getParameter<edm::InputTag>("trackCollection")),
      primVtxLabel_(iConfig.getParameter<edm::InputTag>("PrimaryVertexLabel")) {
  //now do what ever initialization is needed
  outputfile_ = conf_.getParameter<std::string>("OutputFile");

  // here a simple tree can be saved

  tf1 = new TFile(outputfile_.c_str(), "RECREATE");
  s = new TTree("s", " Tree Shared");
  s->Branch("pt", &pt_, "pt/F");
  s->Branch("eta", &eta_, "eta/F");
  s->Branch("phi", &phi_, "phi/F");

  //  s->Branch("",&_,"/F");

  // here histograms can be saved

  edm::Service<TFileService> fs;

  // histograms
  // h_myhisto  = fs->make<TH1F>("h_myhisto"," ",10,0.,10.);
}

BlockAnalyzer::~BlockAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void BlockAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ev++;
  // Just examples
  // the track collection
  //  Handle<reco::TrackCollection> theTracks;
  //Primary Vertexes
  //  Handle<reco::VertexCollection> thePrimaryVertexColl;
  // PFBlocks
  Handle<PFBlockCollection> thePFBlockCollection;

  //  iEvent.getByLabel( trackCollection_, theTracks );
  //  iEvent.getByLabel(primVtxLabel_,thePrimaryVertexColl);
  iEvent.getByLabel(pfblocks_, thePFBlockCollection);

  vector<reco::PFBlock> theBlocks = *(thePFBlockCollection.product());

  if (!theBlocks.empty()) {
    // loop over the pfblocks (for each event you have > 1 blocks)
    for (PFBlockCollection::const_iterator iBlock = theBlocks.begin(); iBlock != theBlocks.end(); iBlock++) {
      PFBlock::LinkData linkData = iBlock->linkData();
      const edm::OwnVector<reco::PFBlockElement>& elements = iBlock->elements();

      // loop over the pfblock elements
      for (unsigned int iEle = 0; iEle < elements.size(); iEle++) {
        PFBlockElement::Type type = elements[iEle].type();
        // example access the element tracks:
        if (type == reco::PFBlockElement::SC) {
          cout << " Found a SuperCluster.  Energy ";
          const reco::PFBlockElementSuperCluster* sc =
              dynamic_cast<const reco::PFBlockElementSuperCluster*>(&elements[iEle]);
          std::cout << sc->superClusterRef()->energy() << " Track/Ecal/Hcal Iso " << sc->trackIso() << " "
                    << sc->ecalIso();
          std::cout << " " << sc->hcalIso() << std::endl;
          // find the linked ECAL clusters
          std::multimap<double, unsigned int> ecalAssoPFClusters;
          iBlock->associatedElements(
              iEle, linkData, ecalAssoPFClusters, reco::PFBlockElement::ECAL, reco::PFBlock::LINKTEST_ALL);

          // loop over the ECAL clusters linked to the iEle
          if (!ecalAssoPFClusters.empty()) {
            // this just to get the first element (the closest)
            //	    unsigned int ecalTrack_index = ecalAssoPFClusters.begin()->second;

            // otherwise is possible to loop over all the elements associated

            for (std::multimap<double, unsigned int>::iterator itecal = ecalAssoPFClusters.begin();
                 itecal != ecalAssoPFClusters.end();
                 ++itecal) {
              // to get the reference to the PF clusters, this is needed.
              reco::PFClusterRef clusterRef = elements[itecal->second].clusterRef();

              // from the clusterRef get the energy, direction, etc
              float ClustRawEnergy = clusterRef->energy();
              float ClustEta = clusterRef->position().eta();
              float ClustPhi = clusterRef->position().phi();

              cout << " My cluster index " << itecal->second << " energy " << ClustRawEnergy << " eta " << ClustEta
                   << " phi " << ClustPhi << endl;

              // now retrieve the tracks associated to the PFClusters
              std::multimap<double, unsigned int> associatedTracks;
              iBlock->associatedElements(
                  itecal->second, linkData, associatedTracks, reco::PFBlockElement::TRACK, reco::PFBlock::LINKTEST_ALL);
              if (!associatedTracks.empty()) {
                for (std::multimap<double, unsigned int>::iterator ittrack = associatedTracks.begin();
                     ittrack != associatedTracks.end();
                     ++ittrack) {
                  cout << " Found a track.  Eenergy ";
                  // no need to dynamic_cast, the trackRef() methods exists for all PFBlockElements
                  std::cout << elements[ittrack->second].trackRef()->p() << std::endl;

                }  // loop on elements
              }    // associated tracks
            }      // loop on ECAL PF Clusters
          }        // there are ECAL PF Clusters
        }          // there is a SuperCluster
      }            // loop on the PFBlock Elements
    }              // loop on the blocks
  }                // there are blocks
}

// ------------ method called once each job just before starting event loop  ------------
void BlockAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) { ev = 0; }

// ------------ method called once each job just after ending the event loop  ------------
void BlockAnalyzer::endJob() {
  tf1->cd();
  s->Write();
  tf1->Write();
  tf1->Close();
  cout << " endJob:: #events " << ev << endl;
}
//define this as a plug-in
DEFINE_FWK_MODULE(BlockAnalyzer);
