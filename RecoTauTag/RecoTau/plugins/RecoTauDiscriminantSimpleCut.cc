/*
 * RecoTauDiscriminantSimpleCut
 *
 * Author: A.J. Johnson, Colorado
 *
 * Apply a cut on the raw isolation pT-sum, given by the discriminator specified via the 'rawIsoPtSum' InputTag,
 * to compute Loose, Medium and Tight working-point discriminators.
 *
 */
#include <boost/foreach.hpp>
#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/TauReco/interface/PFTau.h"


#include <functional>
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"



using namespace reco;
using namespace edm;

class RecoTauDiscriminantSimpleCut : public PFTauDiscriminationProducerBase 
{
 public:
  explicit RecoTauDiscriminantSimpleCut(const edm::ParameterSet& pset);

  ~RecoTauDiscriminantSimpleCut();
  double discriminate(const reco::PFTauRef&) const override;
  void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  
 private:

  double cutValue;

  edm::InputTag rawIsoPtSum_;
  edm::Handle<reco::PFTauDiscriminator> rawIsoPtSumHandle_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> rawIsoPtSum_token;

};

RecoTauDiscriminantSimpleCut::RecoTauDiscriminantSimpleCut(const edm::ParameterSet& cfg)
  : PFTauDiscriminationProducerBase(cfg)
{
  
  rawIsoPtSum_ = cfg.getParameter<edm::InputTag>("rawIsoPtSum");
  rawIsoPtSum_token = consumes<reco::PFTauDiscriminator>(rawIsoPtSum_);
  cutValue = cfg.getParameter<double>("cut");

}

RecoTauDiscriminantSimpleCut::~RecoTauDiscriminantSimpleCut()
{
}

void RecoTauDiscriminantSimpleCut::beginEvent(const edm::Event& evt, const edm::EventSetup& es) 
{
  evt.getByToken(rawIsoPtSum_token, rawIsoPtSumHandle_);
}

double
RecoTauDiscriminantSimpleCut::discriminate(const reco::PFTauRef& tau) const
{
  double disc_result = (*rawIsoPtSumHandle_)[tau];

  // See if the discriminator passes our cuts
  bool passesCut = false;
  passesCut = (disc_result < cutValue);

  return passesCut;
}

DEFINE_FWK_MODULE(RecoTauDiscriminantSimpleCut);

