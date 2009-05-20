#include "RecoBTag/SoftLepton/plugins/SoftPFElectronProducer.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <cmath>

SoftPFElectronProducer::SoftPFElectronProducer (const edm::ParameterSet& conf)
{
    pfElectronTag_ = conf.getParameter<edm::InputTag>("PFElectrons");

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

    produces<reco::PFCandidateCollection>();
}

SoftPFElectronProducer::~SoftPFElectronProducer()
{

}

void SoftPFElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<reco::PFCandidateCollection> output(new reco::PFCandidateCollection);

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByLabel(pfElectronTag_, pfCandidates); 
  if(!pfCandidates.isValid())
    throw edm::Exception(edm::errors::NotFound) << "PFCandidates with InputTag" << pfElectronTag_ << " not found!";

  reco::PFCandidateCollection pfCollection = (*(pfCandidates.product()));

  for(unsigned int i = 0 ; i != pfCollection.size(); ++i)
  {
    if(!isClean(pfCollection[i]))
      continue;

    output->push_back(pfCollection[i]);
  }

  iEvent.put(output);
}

bool SoftPFElectronProducer::isClean(const reco::PFCandidate& pfcandidate)
{
  if(pfcandidate.particleId() != reco::PFCandidate::e) return false;

  const double ecalEnergy            = pfcandidate.ecalEnergy();
  const double pIn                   = pfcandidate.gsfTrackRef()->innerMomentum().R();
  const double pOut                  = pfcandidate.gsfTrackRef()->outerMomentum().R();
  const double EemPinRatio           = (ecalEnergy - pIn)/(ecalEnergy + pIn);
  //const double EemPoutRatio          = (ecalEnergy - pOut)/(ecalEnergy + pOut);
  const double pt                    = pfcandidate.pt();
  const double fbrem                 = (pIn - pOut)/pIn;
  const double dRGsfTrackElectron    = deltaR(pfcandidate.gsfTrackRef()->eta(), pfcandidate.gsfTrackRef()->phi(), pfcandidate.eta(), pfcandidate.phi());
  const double mva                   = pfcandidate.mva_e_pi();
  const math::XYZPoint firstHit      = pfcandidate.gsfTrackRef()->innerPosition();
  const math::XYZPoint lastHit       = pfcandidate.gsfTrackRef()->outerPosition();
  const double inversedRFirstLastHit = 1.0/deltaR(firstHit.eta(), firstHit.phi(), lastHit.eta(), lastHit.phi());
  const double radiusFirstHit        = firstHit.Rho();
  const double zFirstHit             = firstHit.Z();
  /*std::cout << "This particle has " << pt << " "
                                    << 1.0/fbrem << " "
                                    << dRGsfTrackElectron << " "
                                    << EemPoutRatio << " "
                                    << mva << std::endl;*/
  if(fabs(pfcandidate.eta()) < 1.5)
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

void SoftPFElectronProducer::findSeedClusterEnergy(const reco::PFCandidate& pfcandidate, double& energy)
{
  for(unsigned int i = 0; i != pfcandidate.elementsInBlocks().size(); ++i)
  {
    reco::PFBlockRef blockRef = pfcandidate.elementsInBlocks()[i].first;
    unsigned int elementIndex = pfcandidate.elementsInBlocks()[i].second;
    if(blockRef.isNull())
      continue;
    const edm::OwnVector< reco::PFBlockElement >&  elements = (*blockRef).elements();
    const reco::PFBlockElement & pfbe(elements[elementIndex]);
    if(pfbe.type()==reco::PFBlockElement::ECAL)
    {
       reco::PFClusterRef pfSeed = pfbe.clusterRef();
       if(pfSeed.isNull())
         continue;
       energy = pfSeed->energy();
       break;   
    }
  }
}
