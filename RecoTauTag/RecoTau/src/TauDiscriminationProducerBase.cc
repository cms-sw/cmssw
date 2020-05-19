#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

#include <string>

using namespace reco;

namespace {
  template <class TauDiscriminator, class TauCollection>
  struct helper {
    typedef edm::RefProd<TauCollection> TauRefProd;
    static std::unique_ptr<TauDiscriminator> init_result_object(edm::Handle<TauCollection> taus) {
      return std::make_unique<TauDiscriminator>(TauRefProd(taus));
    }
  };

  template <class TauCollection>
  struct helper<TauDiscriminatorContainer, TauCollection> {
    static std::unique_ptr<TauDiscriminatorContainer> init_result_object(edm::Handle<TauCollection> taus) {
      auto result_object = std::make_unique<TauDiscriminatorContainer>();
      TauDiscriminatorContainer::Filler filler(*result_object);
      std::vector<SingleTauDiscriminatorContainer> placeholder(taus->size());
      filler.insert(taus, placeholder.begin(), placeholder.end());
      filler.fill();
      return result_object;
    }
  };
};  // namespace

// default constructor; must not be called
template <class TauType, class TauDiscriminator, class TauDiscriminatorDataType, class ConsumeType>
TauDiscriminationProducerBase<TauType, TauDiscriminator, TauDiscriminatorDataType, ConsumeType>::
    TauDiscriminationProducerBase() {
  throw cms::Exception("TauDiscriminationProducerBase") << " -- default ctor called; derived classes must call "
                                                        << "TauDiscriminationProducerBase(const ParameterSet&)";
}

//--- standard constructor from PSet
template <class TauType, class TauDiscriminator, class TauDiscriminatorDataType, class ConsumeType>
TauDiscriminationProducerBase<TauType, TauDiscriminator, TauDiscriminatorDataType, ConsumeType>::
    TauDiscriminationProducerBase(const edm::ParameterSet& iConfig)
    : moduleLabel_(iConfig.getParameter<std::string>("@module_label")) {
  // tau collection to discriminate
  TauProducer_ = iConfig.getParameter<edm::InputTag>(getTauTypeString() + "Producer");
  Tau_token = consumes<TauCollection>(TauProducer_);

  // prediscriminant operator
  // require the tau to pass the following prediscriminants
  const edm::ParameterSet& prediscriminantConfig = iConfig.getParameter<edm::ParameterSet>("Prediscriminants");

  // determine boolean operator used on the prediscriminants
  std::string pdBoolOperator = prediscriminantConfig.getParameter<std::string>("BooleanOperator");
  // convert string to lowercase
  transform(pdBoolOperator.begin(), pdBoolOperator.end(), pdBoolOperator.begin(), ::tolower);

  if (pdBoolOperator == "and") {
    andPrediscriminants_ = 0x1;  //use chars instead of bools so we can do a bitwise trick later
  } else if (pdBoolOperator == "or") {
    andPrediscriminants_ = 0x0;
  } else {
    throw cms::Exception("TauDiscriminationProducerBase")
        << "PrediscriminantBooleanOperator defined incorrectly, options are: AND,OR";
  }

  // get the list of prediscriminants
  std::vector<std::string> prediscriminantsNames = prediscriminantConfig.getParameterNamesForType<edm::ParameterSet>();

  for (std::vector<std::string>::const_iterator iDisc = prediscriminantsNames.begin();
       iDisc != prediscriminantsNames.end();
       ++iDisc) {
    const edm::ParameterSet& iPredisc = prediscriminantConfig.getParameter<edm::ParameterSet>(*iDisc);
    const edm::InputTag& label = iPredisc.getParameter<edm::InputTag>("Producer");
    double cut = iPredisc.getParameter<double>("cut");

    TauDiscInfo thisDiscriminator;
    thisDiscriminator.label = label;
    thisDiscriminator.cut = cut;
    thisDiscriminator.disc_token = consumes<ConsumeType>(label);
    prediscriminants_.push_back(thisDiscriminator);
  }

  prediscriminantFailValue_ = 0.;

  // register product
  produces<TauDiscriminator>();
}

template <class TauType, class TauDiscriminator, class TauDiscriminatorDataType, class ConsumeType>
void TauDiscriminationProducerBase<TauType, TauDiscriminator, TauDiscriminatorDataType, ConsumeType>::produce(
    edm::Event& event, const edm::EventSetup& eventSetup) {
  tauIndex_ = 0;
  // setup function - does nothing in base, but can be overridden to retrieve PV or other stuff
  beginEvent(event, eventSetup);

  // retrieve the tau collection to discriminate
  edm::Handle<TauCollection> taus;
  event.getByToken(Tau_token, taus);

  edm::ProductID tauProductID = taus.id();

  // output product
  std::unique_ptr<TauDiscriminator> output = helper<TauDiscriminator, TauCollection>::init_result_object(taus);

  size_t nTaus = taus->size();

  // load prediscriminators
  size_t nPrediscriminants = prediscriminants_.size();
  for (size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc) {
    prediscriminants_[iDisc].fill(event);

    // Check to make sure the product is correct for the discriminator.
    // If not, throw a more informative exception.
    edm::ProductID discKeyId = prediscriminants_[iDisc].handle->keyProduct().id();
    if (tauProductID != discKeyId) {
      throw cms::Exception("MisconfiguredPrediscriminant")
          << "The tau collection with input tag " << TauProducer_ << " has product ID: " << tauProductID
          << " but the pre-discriminator with input tag " << prediscriminants_[iDisc].label
          << " is keyed with product ID: " << discKeyId << std::endl;
    }
  }

  // loop over taus
  for (size_t iTau = 0; iTau < nTaus; ++iTau) {
    // get reference to tau
    TauRef tauRef(taus, iTau);

    bool passesPrediscriminants = (andPrediscriminants_ ? 1 : 0);
    // check tau passes prediscriminants
    for (size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc) {
      // current discriminant result for this tau
      double discResult = (*prediscriminants_[iDisc].handle)[tauRef];
      uint8_t thisPasses = (discResult > prediscriminants_[iDisc].cut) ? 1 : 0;

      // if we are using the AND option, as soon as one fails,
      // the result is FAIL and we can quit looping.
      // if we are using the OR option as soon as one passes,
      // the result is pass and we can quit looping

      // truth table
      //        |   result (thisPasses)
      //        |     F     |     T
      //-----------------------------------
      // AND(T) | res=fails |  continue
      //        |  break    |
      //-----------------------------------
      // OR (F) |  continue | res=passes
      //        |           |  break

      if (thisPasses ^ andPrediscriminants_)  //XOR
      {
        passesPrediscriminants = (andPrediscriminants_ ? 0 : 1);  //NOR
        break;
      }
    }

    TauDiscriminatorDataType result = TauDiscriminatorDataType(prediscriminantFailValue_);
    if (passesPrediscriminants) {
      // this tau passes the prereqs, call our implemented discrimination function
      result = discriminate(tauRef);
      ++tauIndex_;
    }

    // store the result of this tau into our new discriminator
    (*output)[tauRef] = result;
  }
  event.put(std::move(output));

  // function to put additional information into the event - does nothing in base, but can be overridden in derived classes
  endEvent(event);
}

template <class TauType, class TauDiscriminator, class TauDiscriminatorDataType, class ConsumeType>
void TauDiscriminationProducerBase<TauType, TauDiscriminator, TauDiscriminatorDataType, ConsumeType>::
    fillProducerDescriptions(edm::ParameterSetDescription& desc) {
  // helper function, it fills the description of the Producers parameter
  desc.add<edm::InputTag>(getTauTypeString() + "Producer", edm::InputTag("fixme"));
  {
    edm::ParameterSetDescription pset_prediscriminants;
    pset_prediscriminants.add<std::string>("BooleanOperator", "AND");
    // optional producers // will it pass if I don't specify them?

    {
      edm::ParameterSetDescription producer_params;
      producer_params.add<double>("cut", 0.);
      producer_params.add<edm::InputTag>("Producer", edm::InputTag("fixme"));

      pset_prediscriminants.addOptional<edm::ParameterSetDescription>("leadTrack", producer_params);
      pset_prediscriminants.addOptional<edm::ParameterSetDescription>("decayMode", producer_params);
    }

    desc.add<edm::ParameterSetDescription>("Prediscriminants", pset_prediscriminants);
  }
}

template <class TauType, class TauDiscriminator, class TauDiscriminatorDataType, class ConsumeType>
std::string
TauDiscriminationProducerBase<TauType, TauDiscriminator, TauDiscriminatorDataType, ConsumeType>::getTauTypeString() {
  if (std::is_same<TauType, reco::PFTau>::value)
    return "PFTau";
  if (std::is_same<TauType, pat::Tau>::value)
    return "PATTau";
  throw cms::Exception("TauDiscriminationProducerBase")
      << "Unsupported TauType used. You must use either PFTau or PATTau.";
}

// compile our desired types and make available to linker
template class TauDiscriminationProducerBase<PFTau,
                                             TauDiscriminatorContainer,
                                             SingleTauDiscriminatorContainer,
                                             PFTauDiscriminator>;
template class TauDiscriminationProducerBase<PFTau, PFTauDiscriminator>;
template class TauDiscriminationProducerBase<pat::Tau,
                                             TauDiscriminatorContainer,
                                             SingleTauDiscriminatorContainer,
                                             pat::PATTauDiscriminator>;
template class TauDiscriminationProducerBase<pat::Tau, pat::PATTauDiscriminator>;
