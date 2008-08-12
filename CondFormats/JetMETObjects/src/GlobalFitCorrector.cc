#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrector.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitParameters.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"


GlobalFitCorrector::GlobalFitCorrector(const CaloSubdetectorGeometry* geom, const std::string& file) : 
  geom_(geom), params_(new SimpleJetCorrectorParameters(file))
{
  // -------------------------------------------
  // determine number of expected tower and jet
  // parameters from file
  // ------------------------------------------
  std::string param = file.substr(file.rfind("_"), file.rfind(".txt")-1);
  if( param.compare( "MyParametrization" ) ) parametrization_= new MyParametrization();
  else if( param.compare( "StepParametrization" ) ) parametrization_= new StepParametrization();   
  else if( param.compare( "JetMETParametrization" ) ) parametrization_= new JetMETParametrization();
  else if( param.compare( "StepEfracParametrization" ) ) parametrization_= new StepEfracParametrization();
  else{
    throw cms::Exception("GlobalFitCorrector") 
      << "cannot instantiate a Parametrization of name " << param << "\n";    
  }

  // -------------------------------------------
  // prepare calibration maps
  // -------------------------------------------
  for(unsigned int idx=0; idx<params_->size(); ++idx){
    const SimpleJetCorrectorParameters::Record& record = params_->record(idx);
    if(record.parameters().size() != parametrization_->nTowerPars()+parametrization_->nJetPars() ){
      throw cms::Exception("Parameter Missmatch") 
	<< "expect" << parametrization_->nTowerPars()+parametrization_->nJetPars()
	<< " for parametrization " << param << " but received " 
	<< record.parameters().size() 
	<< " from file \n";       
    }
    int iEta = indexEta(record.etaMiddle());
    for(unsigned int par=0; par<parametrization_->nTowerPars(); ++par) // phi is set to 0 for the moment
      towerParams_[CalibKey(iEta,0)].push_back(record.parameter(par));
    for(unsigned int par=0; par<parametrization_->nJetPars();   ++par)
      jetParams_  [CalibKey(iEta,0)].push_back(record.parameter(par+parametrization_->nTowerPars()));
  }
}

GlobalFitCorrector::~GlobalFitCorrector()
{
  delete params_; 
  delete parametrization_;
}

double GlobalFitCorrector::correction(const reco::CaloJet& jet) const
{
  double correction=1.;
  reco::Particle::LorentzVector towerCorrected;
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
      CalibMap::const_iterator pars = towerParams_.find(CalibKey(id.ieta(),0)); // phi is set to 0 for the moment
      if( pars != towerParams_.end() ){
	CalibVal val=pars->second;
	scale = parametrization_->correctedTowerEt(&towerDeps[0], &val[0])/(*tower)->et();
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
  CalibMap::const_iterator pars = jetParams_.find(CalibKey(id.ieta(),0)); // phi is set to 0 for the moment
  if( pars != jetParams_.end() ){
    CalibVal val=pars->second;
    correction = parametrization_->correctedJetEt(&jetDeps[0], &val[0])/towerCorrected.pt();
  }
  return correction;
}

int
GlobalFitCorrector::indexEta(double eta)
{
  double theta = 2*std::atan(-std::exp(eta));
  const GlobalPoint p( GlobalPoint::Polar(theta, 0., CaloBoundaries::hcalRadius) );
  CaloTowerDetId towerId = geom_->getClosestCell(p); 
  return towerId.ieta();
}
