//
// $Id: ObjectEnergyScale.h,v 1.4 2008/03/05 14:56:50 fronga Exp $
//

#ifndef PhysicsTools_PatAlgos_ObjectEnergyScale_h
#define PhysicsTools_PatAlgos_ObjectEnergyScale_h

/**
  \class    pat::ObjectEnergyScale ObjectEnergyScale.h "PhysicsTools/PatAlgos/interface/ObjectEnergyScale.h"
  \brief    Energy scale shifting and smearing module

   This class provides energy scale shifting & smearing to objects with
   resolutions for systematic error studies.
   A detailed documentation is found in
     PhysicsTools/PatAlgos/data/ObjectEnergyScale.cfi

  \author   Volker Adler
  \version  $Id: ObjectEnergyScale.h,v 1.4 2008/03/05 14:56:50 fronga Exp $
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"


namespace pat {


  template<class T>
  class ObjectEnergyScale : public edm::EDProducer {

    public:

      explicit ObjectEnergyScale(const edm::ParameterSet& iConfig);
      ~ObjectEnergyScale();

    private:

      virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup);

      float  getSmearing(T& object);
      void   setScale(T& object);

      edm::InputTag objects_;
      float         factor_,
                    shiftFactor_,
                    iniRes_,
                    worsenRes_;
      bool          useFixedMass_,
                    useDefaultIniRes_,
                    useIniResByFraction_,
                    useWorsenResByFactor_;

      CLHEP::RandGaussQ* gaussian_;

  };


}


template<class T>
pat::ObjectEnergyScale<T>::ObjectEnergyScale(const edm::ParameterSet& iConfig)
{
  objects_              = iConfig.getParameter<edm::InputTag>("scaledObject");
  useFixedMass_         = iConfig.getParameter<bool>         ("fixMass");
  shiftFactor_          = iConfig.getParameter<double>       ("shiftFactor");
  useDefaultIniRes_     = iConfig.getParameter<bool>         ("useDefaultInitialResolution");
  iniRes_               = iConfig.getParameter<double>       ("initialResolution");
  useIniResByFraction_  = iConfig.getParameter<bool>         ("initialResolutionByFraction");
  worsenRes_            = iConfig.getParameter<double>       ("worsenResolution");
  useWorsenResByFactor_ = iConfig.getParameter<bool>         ("worsenResolutionByFactor");

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  gaussian_ = new CLHEP::RandGaussQ(engine);

  produces<std::vector<T> >();
}


template<class T>
pat::ObjectEnergyScale<T>::~ObjectEnergyScale()
{
  delete gaussian_;
}


template<class T>
void pat::ObjectEnergyScale<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T> > objectsHandle;
  iEvent.getByLabel(objects_, objectsHandle);
  std::vector<T> objects = *objectsHandle;
  std::auto_ptr<std::vector<T> > objectsVector(new std::vector<T>);
  objectsVector->reserve(objectsHandle->size());

  for ( unsigned int i = 0; i < objects.size(); i++ ) {
    factor_ = shiftFactor_ * ( objects[i].energy() > 0. ?
                               getSmearing(objects[i])  :
                               0.);
    setScale(objects[i]);
    objectsVector->push_back(objects[i]);
  }
  iEvent.put(objectsVector);
}


/// Returns a smearing factor which is multiplied to the initial value then to get it smeared,
/// sets initial resolution to resolution provided by input object if required
/// and converts the 'worsenResolution' parameter to protect from meaningless final resolution values.
template<class T>
float pat::ObjectEnergyScale<T>::getSmearing(T& object)
{
  // overwrite config file parameter 'initialResolution' if required
  if ( useDefaultIniRes_ ) {
    // get initial resolution from input object (and calculate relative initial resolution from absolute value)
    iniRes_ = (1. / sin(object.theta()) * object.resolutionEt() - object.et() * cos(object.theta()) / pow(sin(object.theta()),2) * object.resolutionTheta()) / object.energy(); // conversion of resEt and resTheta into energy resolution
  } else if ( ! useIniResByFraction_ ) {
    // calculate relative initial resolution from absolute value
    iniRes_ = iniRes_ / object.energy();
  }
  // Is 'worsenResolution' a factor or a summand?
  float finalRes = useWorsenResByFactor_                            ?
                    (1.+fabs(1.-fabs(worsenRes_)))   * fabs(iniRes_) :
                    fabs(worsenRes_)/object.energy() + fabs(iniRes_); // conversion as protection from "finalRes_<iniRes_"
  // return smearing factor
  return std::max( gaussian_->fire(1., sqrt(pow(finalRes,2)-pow(iniRes_,2))), 0. ); // protection from negative smearing factors
}


/// Mutliplies the final factor (consisting of shifting and smearing factors) to the object's 4-vector
/// and takes care of preserved masses.
template<class T>
void pat::ObjectEnergyScale<T>::setScale(T& object)
{
  if ( factor_ < 0. ) {
    factor_ = 0.;
  }
  // calculate the momentum factor for fixed or not fixed mass
  float factorMomentum = useFixedMass_ && object.p() > 0.                                   ?
                          sqrt(pow(factor_*object.energy(),2)-object.massSqr()) / object.p() :
                          factor_;
  // set shifted & smeared new 4-vector
  object.setP4(reco::Particle::LorentzVector(factorMomentum*object.px(),
                                             factorMomentum*object.py(),
                                             factorMomentum*object.pz(),
                                             factor_       *object.energy()));
}


#endif
