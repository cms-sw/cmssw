//
// $Id: ObjectSpatialResolution.h,v 1.3 2008/03/05 14:56:50 fronga Exp $
//

#ifndef PhysicsTools_PatAlgos_ObjectSpatialResolution_h
#define PhysicsTools_PatAlgos_ObjectSpatialResolution_h

/**
  \class    pat::ObjectSpatialResolution ObjectSpatialResolution.h "PhysicsTools/PatAlgos/interface/ObjectSpatialResolution.h"
  \brief    Energy scale shifting and smearing module

   This class provides angular smearing to objects with resolutions for
   systematic error studies.
   A detailed documentation is found in
     PhysicsTools/PatAlgos/data/ObjectSpatialResolution.cfi

  \author   Volker Adler
  \version  $Id: ObjectSpatialResolution.h,v 1.3 2008/03/05 14:56:50 fronga Exp $
*/


#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include <cmath>


namespace pat {


  template<class T>
  class ObjectSpatialResolution : public edm::EDProducer {

    public:

      explicit ObjectSpatialResolution(const edm::ParameterSet& iConfig);
      ~ObjectSpatialResolution();

    private:

      virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

      void smearAngles(T& object);

      edm::InputTag objects_;
      double        iniResPolar_,
                    worsenResPolar_,
                    iniResPhi_,
                    worsenResPhi_;
      bool          useDefaultIniRes_,
                    usePolarTheta_,
                    useWorsenResPolarByFactor_,
                    useWorsenResPhiByFactor_;

      CLHEP::RandGaussQ* gaussian_;

  };


}


template<class T>
pat::ObjectSpatialResolution<T>::ObjectSpatialResolution(const edm::ParameterSet& iConfig)
{
  objects_                   = iConfig.getParameter<edm::InputTag>("movedObject");
  useDefaultIniRes_          = iConfig.getParameter<bool>         ("useDefaultInitialResolutions");
  iniResPhi_                 = iConfig.getParameter<double>       ("initialResolutionPhi");
  worsenResPhi_              = iConfig.getParameter<double>       ("worsenResolutionPhi");
  useWorsenResPhiByFactor_   = iConfig.getParameter<bool>         ("worsenResolutionPhiByFactor");
  usePolarTheta_             = iConfig.getParameter<bool>         ("usePolarTheta");
  iniResPolar_               = iConfig.getParameter<double>       ("initialResolutionPolar");
  worsenResPolar_            = iConfig.getParameter<double>       ("worsenResolutionPolar");
  useWorsenResPolarByFactor_ = iConfig.getParameter<bool>         ("worsenResolutionPolarByFactor");

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  gaussian_ = new CLHEP::RandGaussQ(engine);

  produces<std::vector<T> >();
}


template<class T>
pat::ObjectSpatialResolution<T>::~ObjectSpatialResolution()
{
  delete gaussian_;
}


template<class T>
void pat::ObjectSpatialResolution<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T> > objectsHandle;
  iEvent.getByLabel(objects_, objectsHandle);
  std::vector<T> objects = *objectsHandle;
  std::auto_ptr<std::vector<T> > objectsVector(new std::vector<T>);
  objectsVector->reserve(objectsHandle->size());

  for ( unsigned int i = 0; i < objects.size(); i++ ) {
    smearAngles(objects[i]);
    objectsVector->push_back(objects[i]);
  }
  iEvent.put(objectsVector);
}


/// Sets initial resolution to resolution provided by input object if required,
/// smears eta/theta and phi and sets the 4-vector accordingl.
template<class T>
void pat::ObjectSpatialResolution<T>::smearAngles(T& object)
{
  // overwrite config file parameters 'initialResolution...' if required
  if ( useDefaultIniRes_ ) {
    // get initial resolutions from input object
    iniResPhi_   = object.resolutionPhi();    // overwrites config file parameter "initialResolution"
    iniResPolar_ = usePolarTheta_      ?
                   object.resolutionTheta():
                   object.resolutionEta();    // overwrites config file parameter "initialResolution"
  }
  // smear phi
  double finalResPhi = useWorsenResPhiByFactor_                            ?
                       (1.+fabs(1.-fabs(worsenResPhi_))) * fabs(iniResPhi_):
                       fabs(worsenResPhi_) + fabs(iniResPhi_);               // conversions as protection from "finalRes_<iniRes_"
  double smearedPhi = normalizedPhi( gaussian_->fire(object.phi(), sqrt(pow(finalResPhi,2)-pow(iniResPhi_,2))) );
  double finalResPolar = useWorsenResPolarByFactor_                              ?
                         (1.+fabs(1.-fabs(worsenResPolar_))) * fabs(iniResPolar_):
                         fabs(worsenResPolar_) + fabs(iniResPolar_);               // conversions as protection from "finalRes_<iniRes_"
  // smear theta/eta
  const double thetaMin = 2.*atan(exp(-ROOT::Math::etaMax<double>())); // to be on the safe side; however, etaMax=22765 ==> thetaMin=0 within double precision
  double smearedTheta,
         smearedEta;
  if ( usePolarTheta_ ) {
    smearedTheta = gaussian_->fire(object.theta(), sqrt(pow(finalResPolar,2)-pow(iniResPolar_,2)));
    // 0<theta<Pi needs to be assured to have proper calculation of eta
    while ( fabs(smearedTheta) > M_PI ) smearedTheta = smearedTheta < 0.     ?
                                                       smearedTheta + 2.*M_PI:
                                                       smearedTheta - 2.*M_PI;
    if ( smearedTheta < 0. ) {
      smearedTheta = -smearedTheta;
      smearedPhi   = normalizedPhi(smearedPhi+M_PI);
    }
    smearedEta = smearedTheta < thetaMin          ?
                 ROOT::Math::etaMax<double>()     :
                 ( smearedTheta > M_PI-thetaMin ?
                   -ROOT::Math::etaMax<double>():
                   -log(tan(smearedTheta/2.))    ); // eta not calculable for theta=0,Pi, which could occur
  } else {
    smearedEta = gaussian_->fire(object.eta(), sqrt(pow(finalResPolar,2)-pow(iniResPolar_,2)));
    if ( fabs(smearedEta) > ROOT::Math::etaMax<double>() ) smearedEta = smearedEta < 0.              ?
                                                                        -ROOT::Math::etaMax<double>():
                                                                         ROOT::Math::etaMax<double>();
    smearedTheta = 2. * atan(exp(-smearedEta)); // since exp(x)>0 && atan() returns solution closest to 0, 0<theta<Pi should be assured.
  }
  // set smeared new 4-vector
  math::PtEtaPhiELorentzVector newLorentzVector(object.p()*sin(smearedTheta),
                                                smearedEta,
                                                smearedPhi,
                                                object.energy());
  object.setP4(reco::Particle::LorentzVector(newLorentzVector.Px(),
                                             newLorentzVector.Py(),
                                             newLorentzVector.Pz(),
                                             newLorentzVector.E() ));
}


#endif
