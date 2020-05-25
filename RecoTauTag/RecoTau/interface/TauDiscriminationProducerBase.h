#ifndef RecoTauTag_RecoTau_TauDiscriminationProducerBase_H_
#define RecoTauTag_RecoTau_TauDiscriminationProducerBase_H_

/* class TauDiscriminationProducerBase
 *
 * Base classes for producing PFTau and PatTau discriminators
 *
 * PFTaus   - inherit from PFTauDiscriminationProducerBase
 * PatTaus - inherit from PatTauDiscriminationProducerBase
 *
 * The base class takes a (PF/PAT)Tau collection and a collection of
 * associated (PF/PAT)TauDiscriminators.  Each tau is required to pass the given
 * set of prediscriminants.  Taus that pass these are then passed to the
 * pure virtual function
 *
 *      double discriminate(const TauRef&);
 *
 * The derived classes should implement this function and return a double
 * giving the specific discriminant result for this tau.
 *
 * The edm::Event and EventSetup are available to the derived classes
 * at the beginning of the event w/ the virtual function
 *
 *      void beginEvent(...)
 *
 * The derived classes can set the desired value for taus that fail the
 * prediscriminants by setting the protected variable prediscriminantFailValue_
 *
 * created :  Wed Aug 12 16:58:37 PDT 2009
 * Authors :  Evan Friis (UC Davis), Simone Gennai (SNS)
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"

template <class TauType,
          class TauDiscriminator,
          class TauDiscriminatorDataType = double,
          class ConsumeType = TauDiscriminator>
class TauDiscriminationProducerBase : public edm::stream::EDProducer<> {
public:
  // setup framework types for this tautype
  typedef std::vector<TauType> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  typedef edm::RefProd<TauCollection> TauRefProd;

  // standard constructor from PSet
  explicit TauDiscriminationProducerBase(const edm::ParameterSet& iConfig);

  // default constructor must not be called - it will throw an exception
  // derived!  classes must call the parameterset constructor.
  TauDiscriminationProducerBase();

  ~TauDiscriminationProducerBase() override {}

  void produce(edm::Event&, const edm::EventSetup&) override;

  // called at the beginning of every event - override if necessary.
  virtual void beginEvent(const edm::Event&, const edm::EventSetup&) {}

  // abstract functions implemented in derived classes.
  virtual TauDiscriminatorDataType discriminate(const TauRef& tau) const = 0;

  // called at the end of event processing - override if necessary.
  virtual void endEvent(edm::Event&) {}

  struct TauDiscInfo {
    edm::InputTag label;
    edm::Handle<ConsumeType> handle;
    edm::EDGetTokenT<ConsumeType> disc_token;
    // = consumes<ConsumeType>(label);
    double cut;
    void fill(const edm::Event& evt) {
      //	disc_token = consumes<ConsumeType>(label);
      evt.getByToken(disc_token, handle);
    };
  };

  static void fillProducerDescriptions(edm::ParameterSetDescription& desc);

  /// helper method to retrieve tau type name, e.g. to build correct cfi getter
  //string (i.e. PFTau/PATTauProducer)
  static std::string getTauTypeString();

protected:
  //value given to taus that fail prediscriminants
  double prediscriminantFailValue_;

  edm::InputTag TauProducer_;

  std::string moduleLabel_;
  edm::EDGetTokenT<TauCollection> Tau_token;

  // current tau
  size_t tauIndex_;

private:
  std::vector<TauDiscInfo> prediscriminants_;
  // select boolean operation on prediscriminants (and = 0x01, or = 0x00)
  uint8_t andPrediscriminants_;
};

// define our implementations
typedef TauDiscriminationProducerBase<reco::PFTau,
                                      reco::TauDiscriminatorContainer,
                                      reco::SingleTauDiscriminatorContainer,
                                      reco::PFTauDiscriminator>
    PFTauDiscriminationContainerProducerBase;
typedef TauDiscriminationProducerBase<reco::PFTau, reco::PFTauDiscriminator> PFTauDiscriminationProducerBase;
typedef TauDiscriminationProducerBase<pat::Tau,
                                      reco::TauDiscriminatorContainer,
                                      reco::SingleTauDiscriminatorContainer,
                                      pat::PATTauDiscriminator>
    PATTauDiscriminationContainerProducerBase;
typedef TauDiscriminationProducerBase<pat::Tau, pat::PATTauDiscriminator> PATTauDiscriminationProducerBase;

#endif
