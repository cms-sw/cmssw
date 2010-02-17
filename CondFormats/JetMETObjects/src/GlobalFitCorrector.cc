#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitCorrector.h"
#include "CondFormats/JetMETObjects/interface/GlobalFitParameters.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectorParameters.h"


GlobalFitCorrector::GlobalFitCorrector(const std::string& file) : 
  params_(new SimpleJetCorrectorParameters(file))
{
  // -------------------------------------------
  // determine number of expected tower and jet
  // parameters from file
  // ------------------------------------------
  std::string param = file.substr(file.rfind("_")+1, file.rfind(".txt")-file.rfind("_")-1);
  if( param == "StepParametrization" ) parametrization_= new StepParametrization();
  else if( param == "StepParametrizationEnergy" ) parametrization_= new StepParametrizationEnergy();   
  else if( param == "StepEfracParametrization"  ) parametrization_= new StepEfracParametrization();
  else if( param == "StepJetParametrization"    ) parametrization_= new StepJetParametrization();
  else if( param == "MyParametrization"         ) parametrization_= new MyParametrization();
  else if( param == "SimpleParametrization"     ) parametrization_= new SimpleParametrization();
  else if( param == "JetMETParametrization"     ) parametrization_= new JetMETParametrization();
  else{
    throw cms::Exception("GlobalFitCorrector") 
      << "cannot instantiate a Parametrization of name " << param << "\n";    
  }

  // -------------------------------------------
  // fill calibration maps
  // -------------------------------------------  
  for(unsigned int idx=0; idx<params_->size(); ++idx){
    const SimpleJetCorrectorParameters::Record& record = params_->record(idx);
    if(record.parameters().size() != parametrization_->nTowerPars()+parametrization_->nJetPars() ){
      throw cms::Exception("Parameter Missmatch") 
	<< "expect " << parametrization_->nTowerPars()+parametrization_->nJetPars()
	<< " for parametrization " << parametrization_->name() << " but received " 
	<< record.parameters().size() 
	<< " from file \n";       
    }
    int iPhi = idx%72;
    int iEta = indexEta(record.etaMiddle());
    for(unsigned int par=0; par<parametrization_->nTowerPars(); ++par){
      towerParams_[CalibKey(iEta,iPhi)].push_back(record.parameter(par));
    }
    for(unsigned int par=0; par<parametrization_->nJetPars();   ++par)
      jetParams_  [CalibKey(iEta,iPhi)].push_back(record.parameter(par+parametrization_->nTowerPars()));
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
  for(unsigned twr=0; twr<jet.getCaloConstituents().size(); ++twr){
    CaloTowerPtr tower = (jet.getCaloConstituents())[twr];
    CaloTowerDetId id = tower->id();
    
    float scale = 1.;
    if( tower->et()>0. ){
      // prepare dependencies
      TMeasurement* measured = new TMeasurement();
      measured->pt   = tower->et(); 
      measured->EMF  = tower->emEt(); 
      measured->HadF = tower->hadEt(); 
      measured->OutF = tower->outerEt(); 
      measured->E    = tower->energy(); 
      measured->eta  = tower->eta(); 
      measured->phi  = tower->phi(); 
      
      // read tower parameters back
      // and apply tower calibration
      CalibMap::const_iterator pars = towerParams_.find(CalibKey(id.ieta(),id.iphi()));
      if( pars != towerParams_.end() ){
	CalibVal val=pars->second;
	scale = parametrization_->correctedTowerEt(measured, &val[0])/tower->et();
      }
      delete measured;
    }
    // create calibrated tower
    towerCorrected += scale*tower->p4();
  }

  // -------------------------------------------
  // find closest tower to jet in DeltaR  
  // for subsequent jet correction
  // -------------------------------------------
  double dist=-1;
  int closestTower=-1;
  for(unsigned twr=0; twr<jet.getCaloConstituents().size(); ++twr){
    CaloTowerPtr tower = (jet.getCaloConstituents())[twr];
    double dR=deltaR(towerCorrected.eta(), towerCorrected.phi(), tower->eta(), tower->phi());
    if( dist<0 || dR<dist ){
      dist = dR;
      closestTower = (int)twr;
    }
  }
  // save intermediate result (to catch
  // cases without subsequent jet correction)
  correction = towerCorrected.pt()/jet.pt();

  // -------------------------------------------
  // apply subsequent jet correction
  // -------------------------------------------
  CaloTowerDetId id = jet.getCaloConstituent( closestTower )->id();
  CalibMap::const_iterator pars = jetParams_.find(CalibKey(id.ieta(),id.iphi()));
  if( pars != jetParams_.end() ){
    CalibVal val=pars->second;
    
    // prepare dependencies
    TMeasurement* measured = new TMeasurement();
    measured->pt   = towerCorrected.pt(); 
    measured->EMF  = jet.emEnergyFraction()*jet.et(); 
    measured->HadF = towerCorrected.pt()-jet.emEnergyFraction()*jet.et()-jet.hadEnergyInHO(); 
    measured->OutF = jet.hadEnergyInHO(); 
    measured->E    = towerCorrected.energy(); 
    measured->eta  = towerCorrected.eta(); 
    measured->phi  = towerCorrected.phi(); 

    // apply subsequent jet correction
    correction = parametrization_->correctedJetEt(measured, &val[0])/towerCorrected.pt();
    delete measured;
  }
  return correction;
}

int
GlobalFitCorrector::indexEta(double eta) const
{
  if( -5.191<= eta && eta < -4.889) return -41;
  if( -4.889<= eta && eta < -4.716) return -40;
  if( -4.716<= eta && eta < -4.538) return -39;
  if( -4.538<= eta && eta < -4.363) return -38;
  if( -4.363<= eta && eta < -4.191) return -37;
  if( -4.191<= eta && eta < -4.013) return -36;
  if( -4.013<= eta && eta < -3.839) return -35;
  if( -3.839<= eta && eta < -3.664) return -34;
  if( -3.664<= eta && eta < -3.489) return -33;
  if( -3.489<= eta && eta < -3.314) return -32;
  if( -3.314<= eta && eta < -3.139) return -31;
  if( -3.139<= eta && eta < -2.964) return -30;
  if( -2.964<= eta && eta < -2.853) return -29;
  if( -2.853<= eta && eta < -2.65 ) return -28;
  if( -2.65 <= eta && eta < -2.5  ) return -27;
  if( -2.5  <= eta && eta < -2.322) return -26;
  if( -2.322<= eta && eta < -2.172) return -25;
  if( -2.172<= eta && eta < -2.043) return -24;
  if( -2.043<= eta && eta < -1.93 ) return -23;
  if( -1.93 <= eta && eta < -1.83 ) return -22;
  if( -1.83 <= eta && eta < -1.74 ) return -21;
  if( -1.74 <= eta && eta < -1.653) return -20;
  if( -1.653<= eta && eta < -1.566) return -19;
  if( -1.566<= eta && eta < -1.479) return -18;
  if( -1.479<= eta && eta < -1.392) return -17;
  if( -1.392<= eta && eta < -1.305) return -16;
  if( -1.305<= eta && eta < -1.218) return -15;
  if( -1.218<= eta && eta < -1.131) return -14;
  if( -1.131<= eta && eta < -1.044) return -13;
  if( -1.044<= eta && eta < -0.957) return -12;
  if( -0.957<= eta && eta < -0.879) return -11;
  if( -0.879<= eta && eta < -0.783) return -10;
  if( -0.783<= eta && eta < -0.696) return  -9;
  if( -0.696<= eta && eta < -0.609) return  -8;
  if( -0.609<= eta && eta < -0.522) return  -7;
  if( -0.522<= eta && eta < -0.435) return  -6;
  if( -0.435<= eta && eta < -0.348) return  -5;
  if( -0.348<= eta && eta < -0.261) return  -4;
  if( -0.261<= eta && eta < -0.174) return  -3;
  if( -0.174<= eta && eta < -0.087) return  -2;
  if( -0.087<= eta && eta <  0.   ) return  -1;
  if(  0    <= eta && eta <  0.087) return   1;
  if(  0.087<= eta && eta <  0.174) return   2;
  if(  0.174<= eta && eta <  0.261) return   3;
  if(  0.261<= eta && eta <  0.348) return   4;
  if(  0.348<= eta && eta <  0.435) return   5;
  if(  0.435<= eta && eta <  0.522) return   6;
  if(  0.522<= eta && eta <  0.609) return   7;
  if(  0.609<= eta && eta <  0.696) return   8;
  if(  0.696<= eta && eta <  0.783) return   9;
  if(  0.783<= eta && eta <  0.879) return  10;
  if(  0.879<= eta && eta <  0.957) return  11;
  if(  0.957<= eta && eta <  1.044) return  12;
  if(  1.044<= eta && eta <  1.131) return  13;
  if(  1.131<= eta && eta <  1.218) return  14;
  if(  1.218<= eta && eta <  1.305) return  15;
  if(  1.305<= eta && eta <  1.392) return  16;
  if(  1.392<= eta && eta <  1.479) return  17;
  if(  1.479<= eta && eta <  1.566) return  18;
  if(  1.566<= eta && eta <  1.653) return  19;
  if(  1.653<= eta && eta <  1.74 ) return  20;
  if(  1.74 <= eta && eta <  1.83 ) return  21;
  if(  1.83 <= eta && eta <  1.93 ) return  22;
  if(  1.93 <= eta && eta <  2.043) return  23;
  if(  2.043<= eta && eta <  2.172) return  24;
  if(  2.172<= eta && eta <  2.322) return  25;
  if(  2.322<= eta && eta <  2.5  ) return  26;
  if(  2.5  <= eta && eta <  2.65 ) return  27;
  if(  2.65 <= eta && eta <  2.853) return  28;
  if(  2.853<= eta && eta <  2.964) return  29;
  if(  2.964<= eta && eta <  3.139) return  30;
  if(  3.139<= eta && eta <  3.314) return  31;
  if(  3.314<= eta && eta <  3.489) return  32;
  if(  3.489<= eta && eta <  3.644) return  33;
  if(  3.664<= eta && eta <  3.839) return  34;
  if(  3.839<= eta && eta <  4.013) return  35;
  if(  4.013<= eta && eta <  4.191) return  36;
  if(  4.191<= eta && eta <  4.363) return  37;
  if(  4.363<= eta && eta <  4.538) return  38;
  if(  4.538<= eta && eta <  4.716) return  39;
  if(  4.716<= eta && eta <  4.889) return  40;
  if(  4.889<= eta && eta <  5.191) return  41;
  return -999;// return error 
}
