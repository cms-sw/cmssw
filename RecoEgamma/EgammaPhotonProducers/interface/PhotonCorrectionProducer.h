#ifndef RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
/** \class PhotonCorrectionProducer
 **  
 **
 **  $Id: PhotonCorrectionProducer.h,v 1.2 2006/07/26 09:11:32 nancy Exp $ 
 **  $Date: 2006/07/26 09:11:32 $ 
 **  $Revision: 1.2 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonDummyCorrection.h"


// PhotonCorrectionProducer inherits from EDProducer, so it can be a module:
class PhotonCorrectionProducer : public edm::EDProducer {

 public:

  PhotonCorrectionProducer (const edm::ParameterSet& ps);
  ~PhotonCorrectionProducer();


  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  

  
  std::string CorrectedPhotonCollection_;
  std::string photonCorrectionProducer_;
  std::string uncorrectedPhotonCollection_;
  edm::ParameterSet conf_;

  bool applyDummyCorrection_;

  edm::ESHandle<MagneticField> theMF_;

  PhotonDummyCorrection* theDummyCorrection_;


};
#endif
