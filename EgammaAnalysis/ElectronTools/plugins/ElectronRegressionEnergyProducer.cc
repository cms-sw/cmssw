// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyRegressionEvaluate.h"

//
// class declaration
//

using namespace std;
using namespace reco;
using namespace edm;

class ElectronRegressionEnergyProducer : public edm::EDFilter {
public:
  explicit ElectronRegressionEnergyProducer(const edm::ParameterSet&);
  ~ElectronRegressionEnergyProducer();
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);

  // ----------member data ---------------------------
  bool printDebug_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;

  std::string regressionInputFile_;
  uint32_t energyRegressionType_;

  std::string nameEnergyReg_;
  std::string nameEnergyErrorReg_;

  bool geomInitialized_;

  const CaloTopology* ecalTopology_;
  const CaloGeometry* caloGeometry_;

  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;

  ElectronEnergyRegressionEvaluate *regressionEvaluator;

  edm::EDGetTokenT<reco::VertexCollection> hVertexToken_;
  edm::EDGetTokenT<double> hRhoKt6PFJetsToken_;

};


ElectronRegressionEnergyProducer::ElectronRegressionEnergyProducer(const edm::ParameterSet& iConfig) {
  printDebug_  = iConfig.getUntrackedParameter<bool>("printDebug", false);
  electronToken_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electronTag"));

  regressionInputFile_  = iConfig.getParameter<std::string>("regressionInputFile");
  energyRegressionType_ = iConfig.getParameter<uint32_t>("energyRegressionType");

  nameEnergyReg_      = iConfig.getParameter<std::string>("nameEnergyReg");
  nameEnergyErrorReg_ = iConfig.getParameter<std::string>("nameEnergyErrorReg");

  recHitCollectionEBToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitCollectionEB"));
  recHitCollectionEEToken_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitCollectionEE"));

  hVertexToken_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  hRhoKt6PFJetsToken_ = consumes<double>(edm::InputTag("kt6PFJets","rho"));

  produces<edm::ValueMap<double> >(nameEnergyReg_);
  produces<edm::ValueMap<double> >(nameEnergyErrorReg_);

  regressionEvaluator = new ElectronEnergyRegressionEvaluate();

  //set regression type
  ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionType type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  if (energyRegressionType_ == 1) type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  else if (energyRegressionType_ == 2) type = ElectronEnergyRegressionEvaluate::kWithSubCluVar;
  else if (energyRegressionType_ == 3) type = ElectronEnergyRegressionEvaluate::kWithTrkVarV1;
  else if (energyRegressionType_ == 4) type = ElectronEnergyRegressionEvaluate::kWithTrkVarV2;

  //load weights and initialize
  regressionEvaluator->initialize(regressionInputFile_.c_str(),type);

  geomInitialized_ = false;

}


ElectronRegressionEnergyProducer::~ElectronRegressionEnergyProducer()
{
  delete regressionEvaluator;
}

// ------------ method called on each new Event  ------------
bool ElectronRegressionEnergyProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  assert(regressionEvaluator->isInitialized());

  if (!geomInitialized_) {
    edm::ESHandle<CaloTopology> theCaloTopology;
    iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
    ecalTopology_ = & (*theCaloTopology);

    edm::ESHandle<CaloGeometry> theCaloGeometry;
    iSetup.get<CaloGeometryRecord>().get(theCaloGeometry);
    caloGeometry_ = & (*theCaloGeometry);
    geomInitialized_ = true;
  }

  std::auto_ptr<edm::ValueMap<double> > regrEnergyMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyFiller(*regrEnergyMap);

  std::auto_ptr<edm::ValueMap<double> > regrEnergyErrorMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyErrorFiller(*regrEnergyErrorMap);

  Handle<reco::GsfElectronCollection> egCollection;
  iEvent.getByToken(electronToken_,egCollection);
  const reco::GsfElectronCollection egCandidates = (*egCollection.product());

  std::vector<double> energyValues;
  std::vector<double> energyErrorValues;
  energyValues.reserve(egCollection->size());
  energyErrorValues.reserve(egCollection->size());
  //
  //**************************************************************************
  // Rechits
  //**************************************************************************
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  edm::Handle< EcalRecHitCollection > pEERecHits;
  iEvent.getByToken( recHitCollectionEBToken_, pEBRecHits );
  iEvent.getByToken( recHitCollectionEEToken_, pEERecHits );

  //**************************************************************************
  //Get Number of Vertices
  //**************************************************************************
  Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(hVertexToken_,hVertexProduct);
  const reco::VertexCollection inVertices = *(hVertexProduct.product());

  // loop through all vertices
  Int_t nvertices = 0;
  for (reco::VertexCollection::const_iterator inV = inVertices.begin();
       inV != inVertices.end(); ++inV) {

    // pass these vertex cuts
    if (inV->ndof() >= 4
        && inV->position().Rho() <= 2.0
        && fabs(inV->z()) <= 24.0
      ) {
      nvertices++;
    }
  }

  //**************************************************************************
  //Get Rho
  //**************************************************************************
  double rho = 0;
  Handle<double> hRhoKt6PFJets;
  iEvent.getByToken(hRhoKt6PFJetsToken_, hRhoKt6PFJets);
  rho = (*hRhoKt6PFJets);


  for ( reco::GsfElectronCollection::const_iterator egIter = egCandidates.begin();
        egIter != egCandidates.end(); ++egIter) {

    const EcalRecHitCollection * recHits=0;
    if(egIter->isEB())
        recHits = pEBRecHits.product();
    else
        recHits = pEERecHits.product();

    SuperClusterHelper mySCHelper(&(*egIter),recHits,ecalTopology_,caloGeometry_);

    double energy=regressionEvaluator->calculateRegressionEnergy(&(*egIter),
                                                          mySCHelper,
                                                          rho,nvertices,
                                                          printDebug_);

    double error=regressionEvaluator->calculateRegressionEnergyUncertainty(&(*egIter),
                                                                    mySCHelper,
                                                                    rho,nvertices,
                                                                    printDebug_);

    energyValues.push_back(energy);
    energyErrorValues.push_back(error);

  }

  energyFiller.insert( egCollection, energyValues.begin(), energyValues.end() );
  energyFiller.fill();

  energyErrorFiller.insert( egCollection, energyErrorValues.begin(), energyErrorValues.end() );
  energyErrorFiller.fill();

  iEvent.put(regrEnergyMap,nameEnergyReg_);
  iEvent.put(regrEnergyErrorMap,nameEnergyErrorReg_);

  return true;

}


//define this as a plug-in
DEFINE_FWK_MODULE(ElectronRegressionEnergyProducer);



