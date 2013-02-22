#include "JetMETCorrections/Type1MET/plugins/CaloTowerMETcorrInputProducer.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

CaloTowerMETcorrInputProducer::CaloTowerMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
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

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");
  noHF_ = cfg.getParameter<bool>("noHF");
  
  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    produces<CorrMETData>((*binningEntry)->binLabel_);
  }
}

CaloTowerMETcorrInputProducer::~CaloTowerMETcorrInputProducer()
{
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
  //std::cout << "<CaloTowerMETcorrInputProducer::produce>:" << std::endl;

  for ( std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    (*binningEntry)->binUnclEnergySum_ = CorrMETData();
  }

  const JetCorrector* residualCorrector = 0;
  if ( residualCorrLabel_ != "" ) {
    residualCorrector = JetCorrector::getJetCorrector(residualCorrLabel_, es);
    if ( !residualCorrector )  
      throw cms::Exception("CaloTowerMETcorrInputProducer")
	<< "Failed to access Residual corrections = " << residualCorrLabel_ << " !!\n";
  }
  
  typedef edm::View<CaloTower> CaloTowerView;
  edm::Handle<CaloTowerView> caloTowers;
  evt.getByLabel(src_, caloTowers);
  
  int caloTowerIndex = 0;
  for ( CaloTowerView::const_iterator caloTower = caloTowers->begin();
	caloTower != caloTowers->end(); ++caloTower ) {
    //std::cout << "CaloTower #" << caloTowerIndex << " (raw): Pt = " << CaloTower->pt() << "," 
    //	        << " eta = " << CaloTower->eta() << ", phi = " << CaloTower->phi() << std::endl;
        
    double residualCorrFactor = 1.;
    if ( residualCorrector && fabs(caloTower->eta()) < residualCorrEtaMax_ ) {
      residualCorrFactor = residualCorrector->correction(caloTower->p4());
      //std::cout << " residualCorrFactor = " << residualCorrFactor << " (extraCorrFactor = " << extraCorrFactor_ << ")" << std::endl;
    }
    residualCorrFactor *= extraCorrFactor_;
    
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
	(*binningEntry)->binUnclEnergySum_.mex   += ((residualCorrFactor - residualCorrOffset_)*caloTower->px());
	(*binningEntry)->binUnclEnergySum_.mey   += ((residualCorrFactor - residualCorrOffset_)*caloTower->py());
	(*binningEntry)->binUnclEnergySum_.sumet += ((residualCorrFactor - residualCorrOffset_)*caloTower->et());
      }
    }
    ++caloTowerIndex;
  }

//--- add momentum sum of PFCandidates not within jets ("unclustered energy") to the event
  for ( std::vector<binningEntryType*>::const_iterator binningEntry = binning_.begin();
	binningEntry != binning_.end(); ++binningEntry ) {
    evt.put(std::auto_ptr<CorrMETData>(new CorrMETData((*binningEntry)->binUnclEnergySum_)), (*binningEntry)->binLabel_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CaloTowerMETcorrInputProducer);
