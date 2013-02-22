#ifndef PhysicsTools_PatUtils_ShiftedParticleProducerT_h
#define PhysicsTools_PatUtils_ShiftedParticleProducerT_h

/** \class ShiftedParticleProducerT
 *
 * Vary energy of electrons/muons/tau-jets by +/- 1 standard deviation, 
 * in order to estimate resulting uncertainty on MET
 *
 * NOTE: energy scale uncertainties need to be specified in python config
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ShiftedParticleProducerT.h,v 1.2 2011/11/02 14:03:07 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>
#include <math.h>

template <typename T, typename TCollection = std::vector<T> >
class ShiftedParticleProducerT : public edm::EDProducer  
{
 public:

  explicit ShiftedParticleProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {
    src_ = cfg.getParameter<edm::InputTag>("src");

    shiftBy_ = cfg.getParameter<double>("shiftBy");

    if ( cfg.exists("binning") ) {
      typedef std::vector<edm::ParameterSet> vParameterSet;
      vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
      for ( vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin();
	    cfgBinningEntry != cfgBinning.end(); ++cfgBinningEntry ) {
	binning_.push_back(new binningEntryType(*cfgBinningEntry));
      }
    } else {
      double uncertainty = cfg.getParameter<double>("uncertainty");
      double offset = ( cfg.exists("offset") ) ?
	cfg.getParameter<double>("offset") : 0.;
      binning_.push_back(new binningEntryType(uncertainty, offset));
    }
    
    produces<TCollection>();
  }
  ~ShiftedParticleProducerT()
  {
    for ( typename std::vector<binningEntryType*>::const_iterator it = binning_.begin();
	  it != binning_.end(); ++it ) {
      delete (*it);
    }
  }
    
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es)
  {
    edm::Handle<TCollection> originalParticles;
    evt.getByLabel(src_, originalParticles);

    std::auto_ptr<TCollection> shiftedParticles(new TCollection());

    for ( typename TCollection::const_iterator originalParticle = originalParticles->begin();
	  originalParticle != originalParticles->end(); ++originalParticle ) {

      double uncertainty = 0.;
      double offset = 0.;
      for ( typename std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	    binningEntry != binning_.end(); ++binningEntry ) {
	if ( (!(*binningEntry)->binSelection_) || (*(*binningEntry)->binSelection_)(*originalParticle) ) {
	  uncertainty = (*binningEntry)->binUncertainty_;
	  offset = (*binningEntry)->binOffset_;
	  break;
	}
      }
      
      double shift = shiftBy_*uncertainty - offset;

      double shiftedParticlePx = (1. + shift)*originalParticle->px();
      double shiftedParticlePy = (1. + shift)*originalParticle->py();
      double shiftedParticlePz = (1. + shift)*originalParticle->pz();
      double shiftedParticleEn = sqrt(
         shiftedParticlePx*shiftedParticlePx 
       + shiftedParticlePy*shiftedParticlePy 
       + shiftedParticlePz*shiftedParticlePz
       + originalParticle->mass()*originalParticle->mass());
      T shiftedParticle(*originalParticle);      
      reco::Candidate::LorentzVector shiftedParticleP4(shiftedParticlePx, shiftedParticlePy, shiftedParticlePz, shiftedParticleEn);
      shiftedParticle.setP4(shiftedParticleP4);

      shiftedParticles->push_back(shiftedParticle);
    }

    evt.put(shiftedParticles);
  }

  std::string moduleLabel_;

  edm::InputTag src_; 

  struct binningEntryType
  {
    binningEntryType(double uncertainty, double offset)
      : binSelection_(0),
        binUncertainty_(uncertainty),
	binOffset_(offset)
    {}
    binningEntryType(const edm::ParameterSet& cfg)
    : binSelection_(new StringCutObjectSelector<T>(cfg.getParameter<std::string>("binSelection"))),
      binUncertainty_(cfg.getParameter<double>("binUncertainty"))
    {
      binOffset_ = ( cfg.exists("binOffset") ) ?
	cfg.getParameter<double>("binOffset") : 0.;
    }
    ~binningEntryType() 
    {
      delete binSelection_;
    }
    StringCutObjectSelector<T>* binSelection_;
    double binUncertainty_;
    double binOffset_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale
};

#endif


 

