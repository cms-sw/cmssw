// -*- C++ -*-
//
// Package:    GenParticleCounter
// Class:      GenParticleCounter
//
/**\class GenParticleCounter GenParticleCounter.cc CmsHi/GenParticleCounter/src/GenParticleCounter.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Fri Oct 29 12:18:14 CEST 2010
// $Id: GenParticleCounter.cc,v 1.3 2012/05/08 12:34:38 yjlee Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"
//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include <Math/VectorUtil.h>
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "TNtuple.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

//
// class declaration
//

class GenParticleCounter : public edm::EDAnalyzer {
public:
  explicit GenParticleCounter(const edm::ParameterSet&);
  ~GenParticleCounter();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::Service<TFileService> fs;

  TNtuple* nt;
  TTree* theTree;

  std::string mSrc;
  std::string vertexProducer_;      // vertecies producer
  int nPar;
  float recoVtxZ;
  float et[1000];
  float eta[1000];
  float phi[1000];
  int id[1000];
  int momId[1000];
  int status[1000];
  int collId[1000];
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GenParticleCounter::GenParticleCounter(const edm::ParameterSet& iConfig)

{
  mSrc = iConfig.getUntrackedParameter<std::string>("src", "hiGenParticles");
  vertexProducer_  = iConfig.getUntrackedParameter<std::string>("VertexProducer","hiSelectedVertex");

}


GenParticleCounter::~GenParticleCounter()

{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
GenParticleCounter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get the primary event vertex
  recoVtxZ = -100;
  //  edm::Handle<reco::VertexCollection> vertexHandle;
  //  iEvent.getByLabel(InputTag(vertexProducer_), vertexHandle);
  //  reco::VertexCollection vertexCollection = *(vertexHandle.product());
  //  if (vertexCollection.size()>0) {
  //   recoVtxZ = vertexCollection.begin()->position().Z();
  //   }


  edm::Handle<reco::GenParticleCollection> inputHandle;
  iEvent.getByLabel(InputTag(mSrc),inputHandle);
  const reco::GenParticleCollection *collection1 = inputHandle.product();

  int maxindex = (int)collection1->size();

  nPar = 0 ;
  for(int i = 0; i < maxindex; i++)
  {
    const reco::GenParticle &c1 = (*collection1)[i];

    if ( c1.et() < 14 )  continue;

    et[nPar] = c1.et();
    eta[nPar] = c1.eta();
    phi[nPar] = c1.phi();
    status[nPar] = c1.status();
    id[nPar]    = c1.pdgId();
    momId[nPar] = -1;
    if ( c1.mother()!=0 )
      momId[nPar] = c1.mother()->pdgId();
    collId[nPar] = c1.collisionId();
    nPar++;
  }
  theTree->Fill();

}


// ------------ method called once each job just before starting event loop  ------------
void
GenParticleCounter::beginJob()
{

  theTree  = fs->make<TTree>("photon","Tree of Rechits around photon");

  theTree->Branch("nPar",&nPar,"nPar/I");
  theTree->Branch("recoVtxZ",&recoVtxZ,"recoVtxZ/F");

  theTree->Branch("et",et,"et[nPar]/F");
  theTree->Branch("eta",eta,"eta[nPar]/F");
  theTree->Branch("phi",phi,"phi[nPar]/F");
  theTree->Branch("id",id,"id[nPar]/I");
  theTree->Branch("momId",momId,"momId[nPar]/I");
  theTree->Branch("status",status,"status[nPar]/I");
  theTree->Branch("collId",collId,"collId[nPar]/I");

  std::cout<<"done beginjob"<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void
GenParticleCounter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenParticleCounter);
