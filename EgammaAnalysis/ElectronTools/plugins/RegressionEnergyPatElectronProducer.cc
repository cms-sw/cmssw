#include "EgammaAnalysis/ElectronTools/plugins/RegressionEnergyPatElectronProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "EgammaAnalysis/ElectronTools/interface/SuperClusterHelper.h"

#include <iostream>

using namespace edm ;
using namespace std ;
using namespace reco ;
using namespace pat ;


RegressionEnergyPatElectronProducer::RegressionEnergyPatElectronProducer( const edm::ParameterSet & cfg )
{

  inputGsfElectronsToken_ = mayConsume<GsfElectronCollection>(cfg.getParameter<edm::InputTag>("inputElectronsTag"));
  inputPatElectronsToken_ = mayConsume<ElectronCollection>(cfg.getParameter<edm::InputTag>("inputElectronsTag"));
  inputCollectionType_ = cfg.getParameter<uint32_t>("inputCollectionType");
  rhoInputToken_ = consumes<double>(cfg.getParameter<edm::InputTag>("rhoCollection"));
  verticesInputToken_ = consumes<VertexCollection>(cfg.getParameter<edm::InputTag>("vertexCollection"));
  energyRegressionType_ = cfg.getParameter<uint32_t>("energyRegressionType");
  regressionInputFile_ = cfg.getParameter<std::string>("regressionInputFile");
  recHitCollectionEBToken_ = mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitCollectionEB"));
  recHitCollectionEEToken_ = mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitCollectionEE"));
  nameEnergyReg_      = cfg.getParameter<std::string>("nameEnergyReg");
  nameEnergyErrorReg_ = cfg.getParameter<std::string>("nameEnergyErrorReg");
  debug_ = cfg.getUntrackedParameter<bool>("debug");
  useReducedRecHits_ = cfg.getParameter<bool>("useRecHitCollections");
  produceValueMaps_ = cfg.getParameter<bool>("produceValueMaps");

  // if Gsf Electrons; useReducedRecHits should be used)
  if(inputCollectionType_ == 0 && !useReducedRecHits_) {
    throw cms::Exception("InconsistentParameters") << " *** Inconsistent configuration : if you read GsfElectrons, you should set useRecHitCollections to true and provide the correcte values to recHitCollectionEB and recHitCollectionEE (most probably reducedEcalRecHitsEB and reducedEcalRecHitsEE )" << std::endl;
  }

  if(inputCollectionType_ == 0 && !produceValueMaps_) {
    std::cout << " You are running on GsfElectrons and the producer is not configured to produce ValueMaps with the results. In that case, it does not nothing !! " << std::endl;
  }

  if (inputCollectionType_ == 0) {
    // do nothing
  } else if (inputCollectionType_ == 1) {
    produces<ElectronCollection>();
  } else {
    throw cms::Exception("InconsistentParameters")  << " inputCollectionType should be either 0 (GsfElectrons) or 1 (pat::Electrons) " << std::endl;
  }


  //set regression type
  ElectronEnergyRegressionEvaluate::ElectronEnergyRegressionType type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  if (energyRegressionType_ == 1) type = ElectronEnergyRegressionEvaluate::kNoTrkVar;
  else if (energyRegressionType_ == 2) type = ElectronEnergyRegressionEvaluate::kWithSubCluVar;
  else if (energyRegressionType_ == 3) type = ElectronEnergyRegressionEvaluate::kWithTrkVar;

  //load weights and initialize
  regressionEvaluator_ = new ElectronEnergyRegressionEvaluate();
  regressionEvaluator_->initialize(regressionInputFile_.c_str(),type);

  if(produceValueMaps_) {
    produces<edm::ValueMap<double> >(nameEnergyReg_);
    produces<edm::ValueMap<double> >(nameEnergyErrorReg_);
  }


  //****************************************************************************************
  //set up regression calculator
  //****************************************************************************************

  geomInitialized_ = false;

  std::cout << " Finished initialization " << std::endl;

}

RegressionEnergyPatElectronProducer::~RegressionEnergyPatElectronProducer()
{
  delete regressionEvaluator_;

}

void RegressionEnergyPatElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
{

  assert(regressionEvaluator_->isInitialized());
  if (!geomInitialized_) {
    edm::ESHandle<CaloTopology> theCaloTopology;
    setup.get<CaloTopologyRecord>().get(theCaloTopology);
    ecalTopology_ = & (*theCaloTopology);

    edm::ESHandle<CaloGeometry> theCaloGeometry;
    setup.get<CaloGeometryRecord>().get(theCaloGeometry);
    caloGeometry_ = & (*theCaloGeometry);
    geomInitialized_ = true;
  }


  //**************************************************************************
  //Get Number of Vertices
  //**************************************************************************
  Handle<reco::VertexCollection> hVertexProduct;
  event.getByToken(verticesInputToken_,hVertexProduct);
  const reco::VertexCollection inVertices = *(hVertexProduct.product());

  // loop through all vertices
  Int_t nvertices = 0;
  for (reco::VertexCollection::const_iterator inV = inVertices.begin();
       inV != inVertices.end(); ++inV) {

    //pass these vertex cuts
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
  event.getByToken(rhoInputToken_, hRhoKt6PFJets);
  rho = (*hRhoKt6PFJets);

  //*************************************************************************
  // Get the RecHits
  //************************************************************************

  edm::Handle< EcalRecHitCollection > pEBRecHits;
  edm::Handle< EcalRecHitCollection > pEERecHits;
  if (useReducedRecHits_) {
    event.getByToken( recHitCollectionEBToken_, pEBRecHits );
    event.getByToken( recHitCollectionEEToken_, pEERecHits );
  }

  edm::Handle<GsfElectronCollection> gsfCollectionH ;
  edm::Handle<ElectronCollection> patCollectionH;
  if ( inputCollectionType_ == 0 ) {
    event.getByToken ( inputGsfElectronsToken_,gsfCollectionH )  ;
    nElectrons_ = gsfCollectionH->size();
  }
  if ( inputCollectionType_ == 1 ) {
    event.getByToken ( inputPatElectronsToken_,patCollectionH )  ;
    nElectrons_ = patCollectionH->size();
  }

  // prepare the two even if only one is used
  std::auto_ptr<ElectronCollection> patElectrons( new ElectronCollection ) ;

  // Fillers for ValueMaps:
  std::auto_ptr<edm::ValueMap<double> > regrEnergyMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyFiller(*regrEnergyMap);

  std::auto_ptr<edm::ValueMap<double> > regrEnergyErrorMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyErrorFiller(*regrEnergyErrorMap);


  // Reserve the vectors with outputs
  std::vector<double> energyValues;
  std::vector<double> energyErrorValues;
  energyValues.reserve(nElectrons_);
  energyErrorValues.reserve(nElectrons_);


  for(unsigned iele=0; iele < nElectrons_ ; ++iele) {

    const GsfElectron * ele = ( inputCollectionType_ == 0 ) ? &(*gsfCollectionH)[iele] : &(*patCollectionH)[iele] ;
    if (debug_) {
      std::cout << "***********************************************************************\n";
      std::cout << "Run Lumi Event: " << event.id().run() << " " << event.luminosityBlock() << " " << event.id().event() << "\n";
      std::cout << "Pat Electron : " << ele->pt() << " " << ele->eta() << " " << ele->phi() << "\n";
    }

    pat::Electron * myPatElectron = (inputCollectionType_ == 0 ) ?  0 : new pat::Electron((*patCollectionH)[iele]);
    // Get RecHit Collection
    const EcalRecHitCollection * recHits=0;
    if (useReducedRecHits_) {
      if(ele->isEB()) {
	recHits = pEBRecHits.product();
      } else
	recHits = pEERecHits.product();
    } else {
      recHits = (*patCollectionH)[iele].recHits();
    }

    SuperClusterHelper * mySCHelper = 0 ;
    if ( inputCollectionType_ == 0 ) {
      mySCHelper = new SuperClusterHelper(&(*ele),recHits,ecalTopology_,caloGeometry_);
    } else if ( inputCollectionType_ == 1) {
      mySCHelper = new SuperClusterHelper( &(*patCollectionH)[iele], recHits,ecalTopology_,caloGeometry_);
    }

    // apply regression energy
    Double_t FinalMomentum = 0;
    Double_t FinalMomentumError = 0;
    Double_t RegressionMomentum = 0;
    Double_t RegressionMomentumError = 0;

    if (energyRegressionType_ == 1) {

	RegressionMomentum = regressionEvaluator_->regressionValueNoTrkVar( mySCHelper->rawEnergy(),
									    mySCHelper->eta(),
									    mySCHelper->phi(),
									    mySCHelper->r9(),
									    mySCHelper->etaWidth(),
									    mySCHelper->phiWidth(),
									    mySCHelper->clustersSize(),
									    mySCHelper->hadronicOverEm(),
									    rho,
									    nvertices,
									    mySCHelper->seedEta(),
									    mySCHelper->seedPhi(),
									    mySCHelper->seedEnergy(),
									    mySCHelper->e3x3(),
									    mySCHelper->e5x5(),
									    mySCHelper->sigmaIetaIeta(),
									    mySCHelper->spp(),
									    mySCHelper->sep(),
									    mySCHelper->eMax(),
									    mySCHelper->e2nd(),
									    mySCHelper->eTop(),
									    mySCHelper->eBottom(),
									    mySCHelper->eLeft(),
									    mySCHelper->eRight(),
									    mySCHelper->e2x5Max(),
									    mySCHelper->e2x5Top(),
									    mySCHelper->e2x5Bottom(),
									    mySCHelper->e2x5Left(),
									    mySCHelper->e2x5Right(),
									    mySCHelper->ietaSeed(),
									    mySCHelper->iphiSeed(),
									    mySCHelper->etaCrySeed(),
									    mySCHelper->phiCrySeed(),
									    mySCHelper->preshowerEnergyOverRaw(),
									    debug_);
	RegressionMomentumError = regressionEvaluator_->regressionUncertaintyNoTrkVar(
										      mySCHelper->rawEnergy(),
										      mySCHelper->eta(),
										      mySCHelper->phi(),
										      mySCHelper->r9(),
										      mySCHelper->etaWidth(),
										      mySCHelper->phiWidth(),
										      mySCHelper->clustersSize(),
										      mySCHelper->hadronicOverEm(),
										      rho,
										      nvertices,
										      mySCHelper->seedEta(),
										      mySCHelper->seedPhi(),
										      mySCHelper->seedEnergy(),
										      mySCHelper->e3x3(),
										      mySCHelper->e5x5(),
										      mySCHelper->sigmaIetaIeta(),
										      mySCHelper->spp(),
										      mySCHelper->sep(),
										      mySCHelper->eMax(),
										      mySCHelper->e2nd(),
										      mySCHelper->eTop(),
										      mySCHelper->eBottom(),
										      mySCHelper->eLeft(),
										      mySCHelper->eRight(),
										      mySCHelper->e2x5Max(),
										      mySCHelper->e2x5Top(),
										      mySCHelper->e2x5Bottom(),
										      mySCHelper->e2x5Left(),
										      mySCHelper->e2x5Right(),
										      mySCHelper->ietaSeed(),
										      mySCHelper->iphiSeed(),
										      mySCHelper->etaCrySeed(),
										      mySCHelper->phiCrySeed(),
										      mySCHelper->preshowerEnergyOverRaw(),
										      debug_);

	// PAT method
	if(inputCollectionType_ == 1) {
	  myPatElectron->setEcalRegressionEnergy(RegressionMomentum, RegressionMomentumError);
	}
	energyValues.push_back(RegressionMomentum);
	energyErrorValues.push_back(RegressionMomentumError);


    } else if (energyRegressionType_ == 2) {// ECAL regression with subcluster information
        RegressionMomentum = regressionEvaluator_->regressionValueWithSubClusters(
                mySCHelper->rawEnergy(),
                mySCHelper->eta(),
                mySCHelper->phi(),
                mySCHelper->r9(),
                mySCHelper->etaWidth(),
                mySCHelper->phiWidth(),
                mySCHelper->clustersSize(),
                mySCHelper->hadronicOverEm(),
                rho,
                nvertices,
                mySCHelper->seedEta(),
                mySCHelper->seedPhi(),
                mySCHelper->seedEnergy(),
                mySCHelper->e3x3(),
                mySCHelper->e5x5(),
                mySCHelper->sigmaIetaIeta(),
                mySCHelper->spp(),
                mySCHelper->sep(),
                mySCHelper->eMax(),
                mySCHelper->e2nd(),
                mySCHelper->eTop(),
                mySCHelper->eBottom(),
                mySCHelper->eLeft(),
                mySCHelper->eRight(),
                mySCHelper->e2x5Max(),
                mySCHelper->e2x5Top(),
                mySCHelper->e2x5Bottom(),
                mySCHelper->e2x5Left(),
                mySCHelper->e2x5Right(),
                mySCHelper->ietaSeed(),
                mySCHelper->iphiSeed(),
                mySCHelper->etaCrySeed(),
                mySCHelper->phiCrySeed(),
                mySCHelper->preshowerEnergyOverRaw(),
                ele->ecalDrivenSeed(),
                ele->isEBEtaGap(),
                ele->isEBPhiGap(),
                ele->isEEDeeGap(),
                mySCHelper->eSubClusters(),
                mySCHelper->subClusterEnergy(1),
                mySCHelper->subClusterEta(1),
                mySCHelper->subClusterPhi(1),
                mySCHelper->subClusterEmax(1),
                mySCHelper->subClusterE3x3(1),
                mySCHelper->subClusterEnergy(2),
                mySCHelper->subClusterEta(2),
                mySCHelper->subClusterPhi(2),
                mySCHelper->subClusterEmax(2),
                mySCHelper->subClusterE3x3(2),
                mySCHelper->subClusterEnergy(3),
                mySCHelper->subClusterEta(3),
                mySCHelper->subClusterPhi(3),
                mySCHelper->subClusterEmax(3),
                mySCHelper->subClusterE3x3(3),
		        mySCHelper->nPreshowerClusters(),
		        mySCHelper->eESClusters(),
		        mySCHelper->esClusterEnergy(0),
		        mySCHelper->esClusterEta(0),
		        mySCHelper->esClusterPhi(0),
		        mySCHelper->esClusterEnergy(1),
		        mySCHelper->esClusterEta(1),
		        mySCHelper->esClusterPhi(1),
		        mySCHelper->esClusterEnergy(2),
		        mySCHelper->esClusterEta(2),
		        mySCHelper->esClusterPhi(2),
                ele->isEB(),
                debug_);
        RegressionMomentumError = regressionEvaluator_->regressionUncertaintyWithSubClusters(
                mySCHelper->rawEnergy(),
                mySCHelper->eta(),
                mySCHelper->phi(),
                mySCHelper->r9(),
                mySCHelper->etaWidth(),
                mySCHelper->phiWidth(),
                mySCHelper->clustersSize(),
                mySCHelper->hadronicOverEm(),
                rho,
                nvertices,
                mySCHelper->seedEta(),
                mySCHelper->seedPhi(),
                mySCHelper->seedEnergy(),
                mySCHelper->e3x3(),
                mySCHelper->e5x5(),
                mySCHelper->sigmaIetaIeta(),
                mySCHelper->spp(),
                mySCHelper->sep(),
                mySCHelper->eMax(),
                mySCHelper->e2nd(),
                mySCHelper->eTop(),
                mySCHelper->eBottom(),
                mySCHelper->eLeft(),
                mySCHelper->eRight(),
                mySCHelper->e2x5Max(),
                mySCHelper->e2x5Top(),
                mySCHelper->e2x5Bottom(),
                mySCHelper->e2x5Left(),
                mySCHelper->e2x5Right(),
                mySCHelper->ietaSeed(),
                mySCHelper->iphiSeed(),
                mySCHelper->etaCrySeed(),
                mySCHelper->phiCrySeed(),
                mySCHelper->preshowerEnergyOverRaw(),
                ele->ecalDrivenSeed(),
                ele->isEBEtaGap(),
                ele->isEBPhiGap(),
                ele->isEEDeeGap(),
                mySCHelper->eSubClusters(),
                mySCHelper->subClusterEnergy(1),
                mySCHelper->subClusterEta(1),
                mySCHelper->subClusterPhi(1),
                mySCHelper->subClusterEmax(1),
                mySCHelper->subClusterE3x3(1),
                mySCHelper->subClusterEnergy(2),
                mySCHelper->subClusterEta(2),
                mySCHelper->subClusterPhi(2),
                mySCHelper->subClusterEmax(2),
                mySCHelper->subClusterE3x3(2),
                mySCHelper->subClusterEnergy(3),
                mySCHelper->subClusterEta(3),
                mySCHelper->subClusterPhi(3),
                mySCHelper->subClusterEmax(3),
                mySCHelper->subClusterE3x3(3),
		        mySCHelper->nPreshowerClusters(),
		        mySCHelper->eESClusters(),
		        mySCHelper->esClusterEnergy(0),
		        mySCHelper->esClusterEta(0),
		        mySCHelper->esClusterPhi(0),
		        mySCHelper->esClusterEnergy(1),
		        mySCHelper->esClusterEta(1),
		        mySCHelper->esClusterPhi(1),
		        mySCHelper->esClusterEnergy(2),
		        mySCHelper->esClusterEta(2),
		        mySCHelper->esClusterPhi(2),
                ele->isEB(),
                debug_);

        // PAT method
        if(inputCollectionType_ == 1) {
            myPatElectron->setEcalRegressionEnergy(RegressionMomentum, RegressionMomentumError);
        }
        energyValues.push_back(RegressionMomentum);
        energyErrorValues.push_back(RegressionMomentumError);


    }

      else if (energyRegressionType_ == 3) {
	RegressionMomentum = regressionEvaluator_->regressionValueWithTrkVar(ele->p(),
									    mySCHelper->rawEnergy(),
									    mySCHelper->eta(),
									    mySCHelper->phi(),
									    mySCHelper->etaWidth(),
									    mySCHelper->phiWidth(),
									    mySCHelper->clustersSize(),
									    mySCHelper->hadronicOverEm(),
									    mySCHelper->r9(),
									    rho,
									    nvertices,
									    mySCHelper->seedEta(),
									    mySCHelper->seedPhi(),
									    mySCHelper->seedEnergy(),
									    mySCHelper->e3x3(),
									    mySCHelper->e5x5(),
									    mySCHelper->sigmaIetaIeta(),
									    mySCHelper->spp(),
									    mySCHelper->sep(),
									    mySCHelper->eMax(),
									    mySCHelper->e2nd(),
									    mySCHelper->eTop(),
									    mySCHelper->eBottom(),
									    mySCHelper->eLeft(),
									    mySCHelper->eRight(),
									    mySCHelper->e2x5Max(),
									    mySCHelper->e2x5Top(),
									    mySCHelper->e2x5Bottom(),
									    mySCHelper->e2x5Left(),
									    mySCHelper->e2x5Right(),
									    ele->pt(),
									    ele->trackMomentumAtVtx().R(),
									    ele->fbrem(),
									    ele->charge(),
									    ele->eSuperClusterOverP(),
									    mySCHelper->ietaSeed(),
									    mySCHelper->iphiSeed(),
									    mySCHelper->etaCrySeed(),
									    mySCHelper->phiCrySeed(),
									    mySCHelper->preshowerEnergy(),
									    debug_);
	RegressionMomentumError = regressionEvaluator_->regressionUncertaintyWithTrkVar(
										       ele->p(),
										       mySCHelper->rawEnergy(),
										       mySCHelper->eta(),
										       mySCHelper->phi(),
										       mySCHelper->etaWidth(),
										       mySCHelper->phiWidth(),
										       mySCHelper->clustersSize(),
										       mySCHelper->hadronicOverEm(),
										       mySCHelper->r9(),
										       rho,
										       nvertices,
										       mySCHelper->seedEta(),
										       mySCHelper->seedPhi(),
										       mySCHelper->seedEnergy(),
										       mySCHelper->e3x3(),
										       mySCHelper->e5x5(),
										       mySCHelper->sigmaIetaIeta(),
										       mySCHelper->spp(),
										       mySCHelper->sep(),
										       mySCHelper->eMax(),
										       mySCHelper->e2nd(),
										       mySCHelper->eTop(),
										       mySCHelper->eBottom(),
										       mySCHelper->eLeft(),
										       mySCHelper->eRight(),
										       mySCHelper->e2x5Max(),
										       mySCHelper->e2x5Top(),
										       mySCHelper->e2x5Bottom(),
										       mySCHelper->e2x5Left(),
										       mySCHelper->e2x5Right(),
										       ele->pt(),
										       ele->trackMomentumAtVtx().R(),
										       ele->fbrem(),
										       ele->charge(),
										       ele->eSuperClusterOverP(),
										       mySCHelper->ietaSeed(),
										       mySCHelper->iphiSeed(),
										       mySCHelper->etaCrySeed(),
										       mySCHelper->phiCrySeed(),
										       mySCHelper->preshowerEnergy(),
										       debug_);
	FinalMomentum = RegressionMomentum;
	FinalMomentumError = RegressionMomentumError;
	math::XYZTLorentzVector oldMomentum = ele->p4();
	math::XYZTLorentzVector newMomentum = math::XYZTLorentzVector
	  ( oldMomentum.x()*FinalMomentum/oldMomentum.t(),
	    oldMomentum.y()*FinalMomentum/oldMomentum.t(),
	    oldMomentum.z()*FinalMomentum/oldMomentum.t(),
	    FinalMomentum ) ;

	myPatElectron->correctEcalEnergy(RegressionMomentum, RegressionMomentumError);
        myPatElectron->correctMomentum(newMomentum,ele->trackMomentumError(),FinalMomentumError);

	energyValues.push_back(RegressionMomentum);
	energyErrorValues.push_back(RegressionMomentumError);
      } else {
	cout << "Error: RegressionType = " << energyRegressionType_ << " is not supported.\n";
      }

      if(inputCollectionType_ == 1) {
	patElectrons->push_back(*myPatElectron);
     }
      if (myPatElectron) delete myPatElectron;
      if (mySCHelper) delete mySCHelper;
  } // loop on electrons

  // Write the new collection in the event (AOD case)
  if(inputCollectionType_ == 1) {
    event.put(patElectrons) ;
  }

  // now AOD case: write ValueMaps
  if (produceValueMaps_) {

    if ( inputCollectionType_ ==0 ) {
      energyFiller.insert( gsfCollectionH, energyValues.begin(), energyValues.end() );
      energyErrorFiller.insert( gsfCollectionH, energyErrorValues.begin(), energyErrorValues.end() );
    } else if ( inputCollectionType_ ==1 ) {
      energyFiller.insert( patCollectionH, energyValues.begin(), energyValues.end() );
      energyErrorFiller.insert( patCollectionH, energyErrorValues.begin(), energyErrorValues.end() );
    }

    energyFiller.fill();
    energyErrorFiller.fill();
    event.put(regrEnergyMap,nameEnergyReg_);
    event.put(regrEnergyErrorMap,nameEnergyErrorReg_);
  }

}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(RegressionEnergyPatElectronProducer);
