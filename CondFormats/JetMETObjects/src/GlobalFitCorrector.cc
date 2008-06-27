#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrector.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrectorParameters.h"


GlobalFitCorrector::GlobalFitCorrector(const std::string& file) : 
  params_(new GlobalFitCorrectorParameters(file)) {}

GlobalFitCorrector::~GlobalFitCorrector()
{ 
  delete params_; 
}

double GlobalFitCorrector::correction(const reco::CaloJet& jet) const
{
  double correction=1.;
  reco::Particle::LorentzVector towerCorrected;

  // -------------------------------------------
  // prepare calibration maps
  // -------------------------------------------
  CalibMap towerParams, jetParams;
  for(unsigned int idx=0; idx<params_->size(); ++idx){
    for(unsigned int par=0; par<params_->record(idx).towerParameters().size(); ++par)
      towerParams[CalibKey(params_->record(idx).iEta(),params_->record(idx).iPhi())].push_back(params_->record(idx).towerParameter(par));
    for(unsigned int par=0; par<params_->record(idx).jetParameters().size(); ++par)
      jetParams  [CalibKey(params_->record(idx).iEta(),params_->record(idx).iPhi())].push_back(params_->record(idx).jetParameter(par)  );
  }

  // -------------------------------------------
  // apply calo tower correction and
  // sum tower corrected constituents
  // -------------------------------------------
  std::vector<double> towerDeps;
  for(std::vector <CaloTowerPtr>::const_iterator tower = jet.getCaloConstituents().begin() ;
      tower!=jet.getCaloConstituents().end(); ++tower) {
    const CaloTowerDetId& id = (*tower)->id();

    float scale = 1.;
    if( (*tower)->et()>0. ){
      // prepare dependencies
      towerDeps.push_back( (*tower)->et()      );
      towerDeps.push_back( (*tower)->emEt()    );
      towerDeps.push_back( (*tower)->hadEt()   );
      towerDeps.push_back( (*tower)->outerEt() );
      
      // read tower parameters back
      // and apply tower calibration
      CalibMap::const_iterator pars = towerParams.find(CalibKey(id.ieta(),id.iphi()));
      if( pars != towerParams.end() ){
	CalibVal val=pars->second;
	scale = params_->parametrization().correctedTowerEt(&towerDeps[0], &val[0])/(*tower)->et();
      }
    }
    // create calibrated tower
    towerCorrected += scale*(*tower)->p4();
  }

  // -------------------------------------------
  // find closest tower to jet in DeltaR  
  // for subsequent jet correction
  // -------------------------------------------
  int jetIdx = 0;
  double dist=-1;
  int closestTower=-1;
  for(std::vector <CaloTowerPtr>::const_iterator tower = jet.getCaloConstituents().begin() ;
      tower!=jet.getCaloConstituents().end(); ++tower, ++jetIdx) {
    double dR=deltaR(towerCorrected.eta(), towerCorrected.phi(), (*tower)->eta(), (*tower)->phi());
    if( dist<0 || dR<dist ){
      dist = dR;
      closestTower = jetIdx;
    }
  }

  // -------------------------------------------
  // apply subsequent jet correction
  // -------------------------------------------
  std::vector<double> jetDeps;
  jetDeps.push_back( towerCorrected.pt() );
  const CaloTowerDetId& id = jet.getCaloConstituent( closestTower )->id();
  CalibMap::const_iterator pars = jetParams.find(CalibKey(id.ieta(),id.iphi()));
  if( pars != jetParams.end() ){
    CalibVal val=pars->second;
    correction = params_->parametrization().correctedJetEt(&jetDeps[0], &val[0])/towerCorrected.pt();
  }
  return correction;
}
