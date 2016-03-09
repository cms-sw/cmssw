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
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <string>
#include <vector>

template <typename T>
class ShiftedParticleProducerT : public edm::stream::EDProducer<>
{
  typedef std::vector<T> ParticleCollection;

 public:

  explicit ShiftedParticleProducerT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
  {
    srcToken_ = consumes<ParticleCollection>(cfg.getParameter<edm::InputTag>("src"));

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
      binning_.push_back(new binningEntryType(uncertainty));
    }

    produces<ParticleCollection>();
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
    edm::Handle<ParticleCollection> originalParticles;
    evt.getByToken(srcToken_, originalParticles);

    std::auto_ptr<ParticleCollection> shiftedParticles(new ParticleCollection);

    for ( typename ParticleCollection::const_iterator originalParticle = originalParticles->begin();
	  originalParticle != originalParticles->end(); ++originalParticle ) {

      double uncertainty = 0.;
      for ( typename std::vector<binningEntryType*>::iterator binningEntry = binning_.begin();
	    binningEntry != binning_.end(); ++binningEntry ) {
	if ( (!(*binningEntry)->binSelection_) || (*(*binningEntry)->binSelection_)(*originalParticle) ) {
	  uncertainty = (*binningEntry)->binUncertainty_;
	  break;
	}
      }

      double shift = shiftBy_*uncertainty;

      reco::Candidate::LorentzVector shiftedParticleP4 = originalParticle->p4();
      //leave 0*nan = 0 
      if (! (edm::isNotFinite(shift) && shiftedParticleP4.mag2()==0)) shiftedParticleP4 *= (1. + shift);

      T shiftedParticle(*originalParticle);
      shiftedParticle.setP4(shiftedParticleP4);

      shiftedParticles->push_back(shiftedParticle);
    }

    evt.put(shiftedParticles);
  }

  std::string moduleLabel_;

  edm::EDGetTokenT<ParticleCollection> srcToken_;

  struct binningEntryType
  {
    binningEntryType(double uncertainty)
      : binSelection_(0),
        binUncertainty_(uncertainty)
    {}
    binningEntryType(const edm::ParameterSet& cfg)
    : binSelection_(new StringCutObjectSelector<T>(cfg.getParameter<std::string>("binSelection"))),
      binUncertainty_(cfg.getParameter<double>("binUncertainty"))
    {}
    ~binningEntryType()
    {
      delete binSelection_;
    }
    StringCutObjectSelector<T>* binSelection_;
    double binUncertainty_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale
};

#endif




