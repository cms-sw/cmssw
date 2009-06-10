#include "RecoBTag/SoftLepton/plugins/SoftPFElectronProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <cmath>

SoftPFElectronProducer::SoftPFElectronProducer (const edm::ParameterSet& conf)
{
    gsfElectronTag_ = conf.getParameter<edm::InputTag>("Electrons");

    std::vector<double> dummy;
    dummy.push_back(-9999.0);
    dummy.push_back(9999.0);

    barrelPtCuts_                    = conf.getUntrackedParameter<std::vector<double> >("BarrelPtCuts", dummy);
    barreldRGsfTrackElectronCuts_    = conf.getUntrackedParameter<std::vector<double> >("BarreldRGsfTrackElectronCuts", dummy);
    barrelEemPinRatioCuts_           = conf.getUntrackedParameter<std::vector<double> >("BarrelEemPinRatioCuts", dummy);
    barrelMVACuts_                   = conf.getUntrackedParameter<std::vector<double> >("BarrelMVACuts", dummy);
    barrelInversedRFirstLastHitCuts_ = conf.getUntrackedParameter<std::vector<double> >("BarrelInversedRFirstLastHitCuts", dummy);
    barrelRadiusFirstHitCuts_        = conf.getUntrackedParameter<std::vector<double> >("BarrelRadiusFirstHitCuts", dummy);
    barrelZFirstHitCuts_             = conf.getUntrackedParameter<std::vector<double> >("BarrelZFirstHitCuts", dummy);

    forwardPtCuts_                   = conf.getUntrackedParameter<std::vector<double> >("ForwardPtCuts", dummy);
    forwardInverseFBremCuts_         = conf.getUntrackedParameter<std::vector<double> >("ForwardInverseFBremCuts", dummy);
    forwarddRGsfTrackElectronCuts_   = conf.getUntrackedParameter<std::vector<double> >("ForwarddRGsfTrackElectronCuts", dummy);
    forwardRadiusFirstHitCuts_       = conf.getUntrackedParameter<std::vector<double> >("ForwardRadiusFirstHitCuts", dummy);
    forwardZFirstHitCuts_            = conf.getUntrackedParameter<std::vector<double> >("ForwardZFirstHitCuts", dummy);
    forwardMVACuts_                  = conf.getUntrackedParameter<std::vector<double> >("ForwardMVACuts", dummy);

    produces<reco::GsfElectronCollection>();
}

SoftPFElectronProducer::~SoftPFElectronProducer()
{

}

void SoftPFElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<reco::GsfElectronCollection> output(new reco::GsfElectronCollection);

  edm::Handle<reco::GsfElectronCollection> gsfCandidates;
  iEvent.getByLabel(gsfElectronTag_, gsfCandidates); 
  if(!gsfCandidates.isValid())
    throw edm::Exception(edm::errors::NotFound) << "PFCandidates with InputTag" << gsfElectronTag_ << " not found!";

  reco::GsfElectronCollection gsfCollection = (*(gsfCandidates.product()));

  for(unsigned int i = 0 ; i != gsfCollection.size(); ++i)
  {
    if(!isClean(gsfCollection[i]))
      continue;

    output->push_back(gsfCollection[i]);
  }

  iEvent.put(output);
}

bool SoftPFElectronProducer::isClean(const reco::GsfElectron& gsfcandidate)
{
  const double ecalEnergy            = gsfcandidate.superCluster()->energy();
  const double pIn                   = gsfcandidate.gsfTrack()->innerMomentum().R();
  const double pOut                  = gsfcandidate.gsfTrack()->outerMomentum().R();
  const double EemPinRatio           = (ecalEnergy - pIn)/(ecalEnergy + pIn);
  //const double EemPoutRatio          = (ecalEnergy - pOut)/(ecalEnergy + pOut);
  const double pt                    = gsfcandidate.pt();
  const double fbrem                 = (pIn - pOut)/pIn;
  const double dRGsfTrackElectron    = deltaR(gsfcandidate.gsfTrack()->eta(), gsfcandidate.gsfTrack()->phi(), gsfcandidate.eta(), gsfcandidate.phi());
  const double mva                   = gsfcandidate.mva();
  const math::XYZPoint firstHit      = gsfcandidate.gsfTrack()->innerPosition();
  const math::XYZPoint lastHit       = gsfcandidate.gsfTrack()->outerPosition();
  const double inversedRFirstLastHit = 1.0/deltaR(firstHit.eta(), firstHit.phi(), lastHit.eta(), lastHit.phi());
  const double radiusFirstHit        = firstHit.Rho();
  const double zFirstHit             = firstHit.Z();
  /*std::cout << "This particle has " << pt << " "
                                    << 1.0/fbrem << " "
                                    << dRGsfTrackElectron << " "
                                    << EemPoutRatio << " "
                                    << mva << std::endl;*/
  if(fabs(gsfcandidate.eta()) < 1.5)
  {
    if( barrelPtCuts_.front()                    > pt                    || barrelPtCuts_.back()                    < pt) return false;
    if( barreldRGsfTrackElectronCuts_.front()    > dRGsfTrackElectron    || barreldRGsfTrackElectronCuts_.back()    < dRGsfTrackElectron)    return false;
    if( barrelEemPinRatioCuts_.front()           > EemPinRatio           || barrelEemPinRatioCuts_.back()           < EemPinRatio)           return false;
    if( barrelMVACuts_.front()                   > mva                   || barrelMVACuts_.back()                   < mva)                   return false;
    if( barrelInversedRFirstLastHitCuts_.front() > inversedRFirstLastHit || barrelInversedRFirstLastHitCuts_.back() < inversedRFirstLastHit) return false;
    if( barrelRadiusFirstHitCuts_.front()        > radiusFirstHit        || barrelRadiusFirstHitCuts_.back()        < radiusFirstHit)        return false;
    if( barrelZFirstHitCuts_.front()             > zFirstHit             || barrelZFirstHitCuts_.back()             < zFirstHit)             return false;
  }
  else
  {
    if( forwardPtCuts_.front()                 > pt                 || forwardPtCuts_.back()                 < pt)                 return false;
    if( forwardInverseFBremCuts_.front()       > 1.0/fbrem          || forwardInverseFBremCuts_.back()       < 1.0/fbrem)          return false;
    if( forwarddRGsfTrackElectronCuts_.front() > dRGsfTrackElectron || forwarddRGsfTrackElectronCuts_.back() < dRGsfTrackElectron) return false;
    if( forwardRadiusFirstHitCuts_.front()     > radiusFirstHit     || forwardRadiusFirstHitCuts_.back()     < radiusFirstHit)     return false;
    if( forwardZFirstHitCuts_.front()          > zFirstHit          || forwardZFirstHitCuts_.back()          < zFirstHit)          return false;
    if( forwardMVACuts_.front()                > mva                || forwardMVACuts_.back()                < mva)                return false;
  }

  //std::cout << "It passed!" << std::endl;
  return true;
}

