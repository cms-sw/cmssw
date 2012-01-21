#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"


PhotonEnergyCorrector::PhotonEnergyCorrector( const edm::ParameterSet& config, const edm::EventSetup& theEventSetup  ) {


  minR9Barrel_        = config.getParameter<double>("minR9Barrel");
  minR9Endcap_        = config.getParameter<double>("minR9Endcap");
  // get the geometry from the event setup:
  theEventSetup.get<CaloGeometryRecord>().get(theCaloGeom_);
  //  candidateP4type_ = config.getParameter<std::string>("candidateP4type") ;


  // function to extract f(eta) correction
  scEnergyFunction_ = 0 ;
  std::string superClusterFunctionName = config.getParameter<std::string>("superClusterEnergyCorrFunction") ;
  scEnergyFunction_ = EcalClusterFunctionFactory::get()->create(superClusterFunctionName,config) ;
  scEnergyFunction_->init(theEventSetup); 

  // function to extract corrections to cracks
  scCrackEnergyFunction_ = 0 ;
  std::string superClusterCrackFunctionName = config.getParameter<std::string>("superClusterCrackEnergyCorrFunction") ;
  scCrackEnergyFunction_ = EcalClusterFunctionFactory::get()->create(superClusterCrackFunctionName,config) ;
  scCrackEnergyFunction_->init(theEventSetup); 

  // function to extract the error on the sc ecal correction
  scEnergyErrorFunction_ = 0 ;
  std::string superClusterErrorFunctionName = config.getParameter<std::string>("superClusterEnergyErrorFunction") ;
  scEnergyErrorFunction_ = EcalClusterFunctionFactory::get()->create(superClusterErrorFunctionName,config) ;
  scEnergyErrorFunction_->init(theEventSetup); 

  // function  to extract the error on the photon ecal correction
  photonEcalEnergyCorrFunction_=0;
  std::string photonEnergyFunctionName = config.getParameter<std::string>("photonEcalEnergyCorrFunction") ;
  photonEcalEnergyCorrFunction_ = EcalClusterFunctionFactory::get()->create(photonEnergyFunctionName, config);
  photonEcalEnergyCorrFunction_->init(theEventSetup);


  // ingredient for energy regression  
  w_file_ = config.getParameter<std::string>("energyRegressionWeightsLocation");
  regressionCorrector_ = new EGEnergyCorrector(); 
  if ( ! (w_file_ == "none") ) 
    if (!regressionCorrector_->IsInitialized()) regressionCorrector_->Initialize(theEventSetup,w_file_);
}


PhotonEnergyCorrector::~PhotonEnergyCorrector() {
  delete regressionCorrector_;
}



void PhotonEnergyCorrector::calculate(reco::Photon & thePhoton, int subdet) {
  
  double phoEcalEnergy = -9999.;
  double phoEcalEnergyError = -9999.;
  double phoRegr1Energy = -9999.;
  double phoRegr1EnergyError = -9999.;
  const CaloSubdetectorGeometry* subDetGeometry =0 ;
  subDetGeometry =  theCaloGeom_->getSubdetectorGeometry(DetId::Ecal, subdet);

  double minR9=0;
  if (subdet==EcalBarrel) {
    minR9=minR9Barrel_;
  } else if  (subdet==EcalEndcap) {
    minR9=minR9Endcap_;
  }

  ////////////// Here default Ecal corrections based on electrons  ////////////////////////
  if ( thePhoton.r9() > minR9 ) {
    // f(eta) correction to e5x5
    double deltaE = scEnergyFunction_->getValue(*(thePhoton.superCluster()), 1);
    float e5x5=thePhoton.e5x5();
    if (subdet==EcalBarrel) e5x5 = e5x5 * (1.0 +  deltaE/thePhoton.superCluster()->rawEnergy() );
    phoEcalEnergy =  e5x5    +  thePhoton.superCluster()->preshowerEnergy() ;  
  } else {
    phoEcalEnergy = thePhoton.superCluster()->energy();
  }
  // store the value in the Photon.h
  thePhoton.setCorrectedEnergy( reco::Photon::ecal_standard, phoEcalEnergy, phoEcalEnergyError,  false);

  ////////////// Here Ecal corrections specific for photons ////////////////////////
  // correction for low r9 
  if ( thePhoton.r9() > minR9 ) {
    // f(eta) correction to e5x5
    double deltaE = scEnergyFunction_->getValue(*(thePhoton.superCluster()), 1);
    float e5x5=thePhoton.e5x5();
    if (subdet==EcalBarrel) e5x5 = e5x5 * (1.0 +  deltaE/thePhoton.superCluster()->rawEnergy() );
    phoEcalEnergy =  e5x5    +  thePhoton.superCluster()->preshowerEnergy() ;  
  } else {
    phoEcalEnergy =  photonEcalEnergyCorrFunction_->getValue(*(thePhoton.superCluster()), 1);
  }
  
  // add correction for cracks
  phoEcalEnergy *=  scCrackEnergyFunction_->getValue(*(thePhoton.superCluster()));
  // as for the erros use the error on the SC 
  phoEcalEnergyError =   scEnergyErrorFunction_->getValue(*(thePhoton.superCluster()), 0);  
  // store the value in the Photon.h
  thePhoton.setCorrectedEnergy( reco::Photon::ecal_photons, phoEcalEnergy, phoEcalEnergyError,  false);

  //////////  Energy  Regression ////////////////////// 
  if ( !(w_file_ == "none") ) {
    std::pair<double,double> cor = regressionCorrector_->CorrectedEnergyWithError(thePhoton);
    phoRegr1Energy = cor.first;
    phoRegr1EnergyError = cor.second;
    // store the value in the Photon.h
    thePhoton.setCorrectedEnergy( reco::Photon::regression1, phoRegr1Energy, phoRegr1EnergyError,  false);
  }

  /*
  std::cout << " ------------------------- " << std::endl;
  std::cout << " Corrector " << std::endl;
  std::cout << " P4 Type " << thePhoton.getCandidateP4type() << " candidate p4 " << thePhoton.p4() << std::endl;
  std::cout << " photon ecalEnergy " << thePhoton.getCorrectedEnergy(reco::Photon::ecal_photons) << " error " << thePhoton.getCorrectedEnergyError(reco::Photon::ecal_photons) << std::endl;
  std::cout << " ecal p4 from accessor " << thePhoton.p4(reco::Photon::ecal_photons) <<  std::endl;
  std::cout << " ------------------------- " << std::endl;
  std::cout << " reg1 energy " << thePhoton.getCorrectedEnergy(reco::Photon::regression1)  << " error " <<  thePhoton.getCorrectedEnergyError(reco::Photon::regression1) << std::endl;
  std::cout << " New p4 from regression " <<  thePhoton.p4(reco::Photon::regression1)    << std::endl;
  std::cout << " ------------------------- " << std::endl;
  */


}
