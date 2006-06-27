#ifndef RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
/** \class PhotonCorrectionProducer
 **  
 **
 **  $Id: PhotonCorrectionProducer.h,v 1.1 2006/06/09 15:55:49 nancy Exp $ 
 **  $Date: 2006/06/09 15:55:49 $ 
 **  $Revision: 1.1 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgo.h"


// PhotonCorrectionProducer inherits from EDProducer, so it can be a module:
class PhotonCorrectionProducer : public edm::EDProducer {

 public:

  PhotonCorrectionProducer (const edm::ParameterSet& ps);
  ~PhotonCorrectionProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  

  
  std::string PhotonCollection_;
  std::string phoProducer_;
  std::string phoCollection_;
  edm::ParameterSet conf_;

  edm::ESHandle<MagneticField> theMF_;

  PhotonCorrectionAlgo* theCorrections_;


};
#endif
