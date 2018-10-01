/*
 * ============================================================================
 *       Filename:  RecoTauMVATransform.cc
 *
 *    Description:  Transform TaNC output according to decay mode.
 *        Created:  10/22/2010 15:36:12
 *         Author:  Evan K. Friis (UC Davis), evan.klose.friis@cern.ch
 * ============================================================================
 */

#include <boost/foreach.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <memory>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TGraph.h"

namespace {

// Build the transformation function from the PSet format
std::unique_ptr<TGraph> buildTransform(const edm::ParameterSet &pset) {
  double min = pset.getParameter<double>("min");
  double max = pset.getParameter<double>("max");
  const std::vector<double> &values =
      pset.getParameter<std::vector<double> >("transform");
  double stepSize = (max - min)/(values.size()-1);
  std::unique_ptr<TGraph> output(new TGraph(values.size()));
  for (size_t step = 0; step < values.size(); ++step) {
    double x = min + step*stepSize;
    output->SetPoint(step, x, values[step]);
  }
  return output;
}

}

class RecoTauMVATransform : public PFTauDiscriminationProducerBase {
  public:
    explicit RecoTauMVATransform(const edm::ParameterSet& pset);
    ~RecoTauMVATransform() override {}

    void beginEvent(const edm::Event&, const edm::EventSetup&) override;
    double discriminate(const reco::PFTauRef&) const override;

  private:
    // Map a decay mode to a transformation
    typedef boost::ptr_map<reco::PFTau::hadronicDecayMode, TGraph> TransformMap;
    TransformMap transforms_;
    // Points to raw TaNC output
    edm::InputTag input_;
    edm::Handle<reco::PFTauDiscriminator> disc_;
};


RecoTauMVATransform::RecoTauMVATransform(const edm::ParameterSet& pset)
  :PFTauDiscriminationProducerBase(pset) {
  input_ = pset.getParameter<edm::InputTag>("toTransform");
  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& transforms = pset.getParameter<VPSet>("transforms");
  prediscriminantFailValue_ = -2.0;
  BOOST_FOREACH(const edm::ParameterSet &transform, transforms) {
    unsigned int nCharged = transform.getParameter<unsigned int>("nCharged");
    unsigned int nPiZeros = transform.getParameter<unsigned int>("nPiZeros");
    // Get the transform
    const edm::ParameterSet &transformImpl =
        transform.getParameter<edm::ParameterSet>("transform");
    // Get the acutal decay mode
    reco::PFTau::hadronicDecayMode decayMode =
        reco::tau::translateDecayMode(nCharged, nPiZeros);

    if (!transforms_.count(decayMode)) {
      // Add it
      transforms_.insert(decayMode, buildTransform(transformImpl).get());
    } else {
      edm::LogError("DecayModeNotUnique") << "The tau decay mode with "
        "nCharged/nPiZero = " << nCharged << "/" << nPiZeros <<
        " dm: " << decayMode <<
        " is associated to multiple MVA transforms, "
        "the second instantiation is being ignored!!!";
    }
  }
}

// Update our discriminator handle at the begninng of the event
void
RecoTauMVATransform::beginEvent(const edm::Event& evt, const edm::EventSetup&) {
  evt.getByLabel(input_, disc_);
}

double RecoTauMVATransform::discriminate(const reco::PFTauRef& tau) const {
  // Check if we support this decay mode:
  TransformMap::const_iterator transformIter =
      transforms_.find(tau->decayMode());
  // Unsupported DM
  if (transformIter == transforms_.end())
    return prediscriminantFailValue_;
  const TGraph *transform = transformIter->second;
  // Get the discriminator output to transform
  double value = (*disc_)[tau];
  double result = transform->Eval(value);
  return result;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauMVATransform);
