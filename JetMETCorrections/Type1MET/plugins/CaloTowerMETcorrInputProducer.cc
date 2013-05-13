#include "JetMETCorrections/Type1MET/plugins/CaloTowerMETcorrInputProducer.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

CaloTowerMETcorrInputProducer::CaloTowerMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
    residualCorrectorFromFile_(0)
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  if ( cfg.exists("binning") ) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for ( vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin();
	  cfgBinningEntry != cfgBinning.end(); ++cfgBinningEntry ) {
      binning_.push_back(new binningEntryType(*cfgBinningEntry));
    }
  } else {
    binning_.push_back(new binningEntryType());
  }
  
  residualCorrLabel_ = cfg.getParameter<std::string>("residualCorrLabel");
  residualCorrEtaMax_ = cfg.getParameter<double>("residualCorrEtaMax");
  residualCorrOffset_ = cfg.getParameter<double>("residualCorrOffset");
  extraCorrFactor_ = cfg.exists("extraCorrFactor") ? 
    cfg.getParameter<double>("extraCorrFactor") : 1.;
  if ( cfg.exists("residualCorrFileName") ) {
    edm::FileInPath residualCorrFileName = cfg.getParameter<edm::FileInPath>("residualCorrFileName");
    if ( !residualCorrFileName.isLocal()) 
      throw cms::Exception("calibUnclusteredEnergy") 
	<< " Failed to find File = " << residualCorrFileName << " !!\n";
    JetCorrectorParameters residualCorr(residualCorrFileName.fullPath().data());
    std::vector<JetCorrectorParameters> jetCorrections;
    jetCorrections.push_back(residualCorr);
    residualCorrectorFromFile_ = new FactorizedJetCorrector(jetCorrections);
  }
  isMC_ = cfg.getParameter<bool>("isMC");

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");
  noHF_ = cfg.getParameter<bool>("noHF");
  
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
    produces<CorrMETData>((*binningEntry)->binLabel_em_);
    produces<CorrMETData>((*binningEntry)->binLabel_had_);
  }
}

CaloTowerMETcorrInputProducer::~CaloTowerMETcorrInputProducer()
{
  delete residualCorrectorFromFile_;
  for ( std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	it != binning_.end(); ++it ) {
    delete (*it);
  }
}

namespace
{
  DetId find_DetId_of_HCAL_cell_in_constituent_of(const CaloTower& calotower)
  {
    // CV: function copied from RecoMET/METAlgorithms/src/CaloSpecificAlgo.cc
    DetId ret;
    for ( int cell = calotower.constituentsSize() - 1; cell >= 0; --cell ) {
      DetId id = calotower.constituent(cell);
      if( id.det() == DetId::Hcal ) {
	ret = id;
	break;
      }
    }
    return ret;
  }
}

void CaloTowerMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) { 
    std::cout << "<CaloTowerMETcorrInputProducer::produce>:" << std::endl;
    std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
  }

  for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    (*binningEntry)->binUnclEnergySum_     = CorrMETData();
    (*binningEntry)->binUnclEnergySum_em_  = CorrMETData();
    (*binningEntry)->binUnclEnergySum_had_ = CorrMETData();
  }

  const JetCorrector* residualCorrectorFromDB = 0;
  if ( !residualCorrectorFromFile_ && residualCorrLabel_ != "" ) {
    residualCorrectorFromDB = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrectorFromDB )  
      throw cms::Exception("CaloTowerMETcorrInputProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
  }
  
  typedef edm::View<CaloTower> CaloTowerView;
  edm::Handle<CaloTowerView> caloTowers;
  evt.getByLabel(src_, caloTowers);
  
  int idxCaloTower = 0;
  for ( CaloTowerView::const_iterator caloTower = caloTowers->begin();
	caloTower != caloTowers->end(); ++caloTower ) {
    if ( verbosity_ ) { 
      std::cout << "CaloTower #" << idxCaloTower << " (raw): Pt = " << caloTower->pt() << "," 
    	        << " eta = " << caloTower->eta() << ", phi = " << caloTower->phi() << std::endl;
    }
        
    double residualCorrFactor = 1.;
    if ( fabs(caloTower->eta()) < residualCorrEtaMax_ ) {
      if ( residualCorrectorFromFile_ ) {
	residualCorrectorFromFile_->setJetEta(caloTower->eta());
	residualCorrectorFromFile_->setJetPt(10.);
	residualCorrectorFromFile_->setJetA(0.25);
	residualCorrectorFromFile_->setRho(10.); 
	residualCorrFactor = residualCorrectorFromFile_->getCorrection();
      } else if ( residualCorrectorFromDB ) {
	residualCorrFactor = residualCorrectorFromDB->correction(caloTower->p4());
      }
      if ( verbosity_ ) std::cout << " residualCorrFactor = " << residualCorrFactor << " (extraCorrFactor = " << extraCorrFactor_ << ")" << std::endl;
    }
    residualCorrFactor *= extraCorrFactor_;
    if ( isMC_ && residualCorrFactor != 0. ) residualCorrFactor = 1./residualCorrFactor;
    
    if ( (residualCorrFactor*caloTower->et()) < globalThreshold_ ) continue;
    
    if ( noHF_ ) {
      DetId detId_hcal = find_DetId_of_HCAL_cell_in_constituent_of(*caloTower);
      if( !detId_hcal.null() ) {
	HcalSubdetector subdet = HcalDetId(detId_hcal).subdet();
	if( subdet == HcalForward ) continue;
      }
    }

    for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	  binningEntry != binning_.end(); ++binningEntry ) {
      if ( !(*binningEntry)->binSelection_ || (*(*binningEntry)->binSelection_)(caloTower->p4()) ) {
	if ( verbosity_ ) std::cout << "adding CaloTower." << std::endl;
	(*binningEntry)->binUnclEnergySum_.mex         += ((residualCorrFactor - residualCorrOffset_)*caloTower->px());
	(*binningEntry)->binUnclEnergySum_.mey         += ((residualCorrFactor - residualCorrOffset_)*caloTower->py());
	(*binningEntry)->binUnclEnergySum_.sumet       += ((residualCorrFactor - residualCorrOffset_)*caloTower->et());
	if ( caloTower->energy() > 0. ) {
	  double emEnFrac = caloTower->emEnergy()/caloTower->energy();
	  (*binningEntry)->binUnclEnergySum_em_.mex    += (emEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->px());
	  (*binningEntry)->binUnclEnergySum_em_.mey    += (emEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->py());
	  (*binningEntry)->binUnclEnergySum_em_.sumet  += (emEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->et());
	}	
	if ( caloTower->energy() > 0. ) {
	  double hadEnFrac = caloTower->hadEnergy()/caloTower->energy();
	  (*binningEntry)->binUnclEnergySum_had_.mex   += (hadEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->px());
	  (*binningEntry)->binUnclEnergySum_had_.mey   += (hadEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->py());
	  (*binningEntry)->binUnclEnergySum_had_.sumet += (hadEnFrac*(residualCorrFactor - residualCorrOffset_)*caloTower->et());
	}
      }
    }
    ++idxCaloTower;
  }

//--- add momentum sum of CaloTowers not within jets ("unclustered energy") to the event
  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_)), (*binningEntry)->binLabel_);
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_em_)), (*binningEntry)->binLabel_em_);
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_had_)), (*binningEntry)->binLabel_had_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CaloTowerMETcorrInputProducer);
