// -*- C++ -*-
//
// Package:    RecoJets/FFTJetProducers
// Class:      FFTJetCorrectionProducer
//
/**\class FFTJetCorrectionProducer FFTJetCorrectionProducer.cc RecoJets/FFTJetProducers/plugins/FFTJetCorrectionProducer.cc

 Description: producer for correcting jets created by FFTJetProducer

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Mon Aug  6 11:03:38 CDT 2012
//
//

// system include files
#include <iostream>
#include <memory>
#include <cfloat>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/FFTJetAlgorithms/interface/adjustForPileup.h"
#include "RecoJets/FFTJetProducers/interface/JetType.h"
#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceTypemap.h"

#define PILEUP_CALCULATION_MASK 0x200
#define PILEUP_SUBTRACTION_MASK_4VEC 0x400
#define PILEUP_SUBTRACTION_MASK_PT 0x800
#define PILEUP_SUBTRACTION_MASK_ANY (PILEUP_SUBTRACTION_MASK_4VEC | PILEUP_SUBTRACTION_MASK_PT)

using namespace fftjetcms;

//
// A generic switch statement based on jet type
//
#define jet_type_switch(method, arg1, arg2)                              \
  do {                                                                   \
    switch (jetType) {                                                   \
      case CALOJET:                                                      \
        method<reco::CaloJet>(arg1, arg2);                               \
        break;                                                           \
      case PFJET:                                                        \
        method<reco::PFJet>(arg1, arg2);                                 \
        break;                                                           \
      case GENJET:                                                       \
        method<reco::GenJet>(arg1, arg2);                                \
        break;                                                           \
      case TRACKJET:                                                     \
        method<reco::TrackJet>(arg1, arg2);                              \
        break;                                                           \
      case BASICJET:                                                     \
        method<reco::BasicJet>(arg1, arg2);                              \
        break;                                                           \
      case JPTJET:                                                       \
        method<reco::JPTJet>(arg1, arg2);                                \
        break;                                                           \
      default:                                                           \
        assert(!"ERROR in FFTJetCorrectionProducer : invalid jet type."\
               " This is a bug. Please report."); \
    }                                                                    \
  } while (0);

namespace {
  struct LocalSortByPt {
    template <class Jet>
    inline bool operator()(const Jet& l, const Jet& r) const {
      return l.pt() > r.pt();
    }
  };
}  // namespace

//
// class declaration
//
class FFTJetCorrectionProducer : public edm::stream::EDProducer<> {
public:
  explicit FFTJetCorrectionProducer(const edm::ParameterSet&);
  ~FFTJetCorrectionProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  template <typename Jet>
  void makeProduces(const std::string& alias, const std::string& tag);

  template <typename Jet>
  void applyCorrections(edm::Event& iEvent, const edm::EventSetup& iSetup);

  template <typename Jet>
  void performPileupSubtraction(Jet&);

  // Label for the input collection
  const edm::InputTag inputLabel;

  // Label for the output objects
  const std::string outputLabel;

  // Jet type to process
  const JetType jetType;

  // The names of the jet correction records
  const std::vector<std::string> records;

  // Are we going to create the uncertainty collection?
  const bool writeUncertainties;

  // What to do about pileup subtraction
  const bool subtractPileup;
  const bool subtractPileupAs4Vec;

  // Print some info about jets
  const bool verbose;

  // Space for level masks
  std::vector<int> sequenceMasks;

  // Event counter
  unsigned long eventCount;

  // tokens for data access
  edm::EDGetTokenT<std::vector<reco::FFTAnyJet<reco::Jet>>> input_jets_token_;
};

template <typename T>
void FFTJetCorrectionProducer::makeProduces(const std::string& alias, const std::string& tag) {
  produces<std::vector<reco::FFTAnyJet<T>>>(tag).setBranchAlias(alias);
}

template <typename Jet>
void FFTJetCorrectionProducer::performPileupSubtraction(Jet& jet) {
  using reco::FFTJet;

  FFTJet<float>& fftJet(const_cast<FFTJet<float>&>(jet.getFFTSpecific()));
  const math::XYZTLorentzVector& new4vec = adjustForPileup(fftJet.f_vec(), fftJet.f_pileup(), subtractPileupAs4Vec);
  fftJet.setFourVec(new4vec);
  int status = fftJet.f_status();
  if (subtractPileupAs4Vec)
    status |= PILEUP_SUBTRACTION_MASK_4VEC;
  else
    status |= PILEUP_SUBTRACTION_MASK_PT;
  fftJet.setStatus(status);
  jet.setP4(new4vec);
}

template <typename Jet>
void FFTJetCorrectionProducer::applyCorrections(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using reco::FFTJet;
  typedef reco::FFTAnyJet<Jet> MyJet;
  typedef std::vector<MyJet> MyCollection;
  typedef typename FFTJetCorrectorSequenceTypemap<MyJet>::loader Loader;
  typedef typename Loader::data_type CorrectorSequence;
  typedef typename CorrectorSequence::result_type CorrectorResult;

  // Load the jet corrector sequences
  const unsigned nRecords = records.size();
  std::vector<edm::ESHandle<CorrectorSequence>> handles(nRecords);
  for (unsigned irec = 0; irec < nRecords; ++irec)
    Loader::instance().load(iSetup, records[irec], handles[irec]);

  // Figure out which correction levels we are applying
  // and create masks which will indicate this
  sequenceMasks.clear();
  sequenceMasks.reserve(nRecords);

  int totalMask = 0;
  for (unsigned irec = 0; irec < nRecords; ++irec) {
    int levelMask = 0;
    const unsigned nLevels = handles[irec]->nLevels();
    for (unsigned i = 0; i < nLevels; ++i) {
      const unsigned lev = (*handles[irec])[i].level();

      // Not tracking "level 0" corrections in the status word.
      // Level 0 is basically reserved for uncertainty calculations.
      if (lev) {
        const int mask = (1 << lev);
        if (totalMask & mask)
          throw cms::Exception("FFTJetBadConfig")
              << "Error in FFTJetCorrectionProducer::applyCorrections:"
              << " jet correction at level " << lev << " is applied more than once\n";
        totalMask |= mask;
        levelMask |= mask;
      }
    }
    sequenceMasks.push_back(levelMask << 12);
  }
  totalMask = (totalMask << 12);

  // Is this data or MC?
  const bool isMC = !iEvent.isRealData();

  // Load the jet collection
  edm::Handle<MyCollection> jets;
  iEvent.getByToken(input_jets_token_, jets);

  // Create the output collection
  const unsigned nJets = jets->size();
  auto coll = std::make_unique<MyCollection>();
  coll->reserve(nJets);

  // Cycle over jets and apply the corrector sequences
  bool sorted = true;
  double previousPt = DBL_MAX;
  for (unsigned ijet = 0; ijet < nJets; ++ijet) {
    const MyJet& j((*jets)[ijet]);

    // Check that this jet has not been corrected yet
    const int initialStatus = j.getFFTSpecific().f_status();
    if (initialStatus & totalMask)
      throw cms::Exception("FFTJetBadConfig") << "Error in FFTJetCorrectionProducer::applyCorrections: "
                                              << "this jet collection is already corrected for some or all "
                                              << "of the specified levels\n";

    MyJet corJ(j);

    if (verbose) {
      const reco::FFTJet<float>& fj = corJ.getFFTSpecific();
      std::cout << "++++ Evt " << eventCount << " jet " << ijet << ": pt = " << corJ.pt()
                << ", eta = " << fj.f_vec().eta() << ", R = " << fj.f_recoScale() << ", s = 0x" << std::hex
                << fj.f_status() << std::dec << std::endl;
    }

    // Check if we need to subtract pileup first.
    // Pileup subtraction is not part of the corrector sequence
    // itself because 4-vector subtraction does not map well
    // into multiplication of 4-vectors by a scale factor.
    if (subtractPileup) {
      if (initialStatus & PILEUP_SUBTRACTION_MASK_ANY)
        throw cms::Exception("FFTJetBadConfig") << "Error in FFTJetCorrectionProducer::applyCorrections: "
                                                << "this jet collection is already pileup-subtracted\n";
      if (!(initialStatus & PILEUP_CALCULATION_MASK))
        throw cms::Exception("FFTJetBadConfig") << "Error in FFTJetCorrectionProducer::applyCorrections: "
                                                << "pileup was not calculated for this jet collection\n";
      performPileupSubtraction(corJ);

      if (verbose) {
        const reco::FFTJet<float>& fj = corJ.getFFTSpecific();
        std::cout << "     Pileup subtract"
                  << ": pt = " << corJ.pt() << ", eta = " << fj.f_vec().eta() << ", R = " << fj.f_recoScale()
                  << ", s = 0x" << std::hex << fj.f_status() << std::dec << std::endl;
      }
    }

    // Apply all jet correction sequences
    double sigmaSquared = 0.0;
    for (unsigned irec = 0; irec < nRecords; ++irec) {
      const CorrectorResult& corr = handles[irec]->correct(corJ, isMC);

      // Update the 4-vector
      FFTJet<float>& fftJet(const_cast<FFTJet<float>&>(corJ.getFFTSpecific()));
      corJ.setP4(corr.vec());
      fftJet.setFourVec(corr.vec());

      // Update the jet correction status
      fftJet.setStatus(fftJet.f_status() | sequenceMasks[irec]);

      // Update the (systematic) uncertainty
      const double s = corr.sigma();
      sigmaSquared += s * s;
    }

    // There is no place for uncertainty in the jet structure.
    // However, there is the unused pileup field (FFTJet maintains
    // the pileup separately as a 4-vector). Use this unused field
    // to store the uncertainty. This hack is needed because
    // subsequent sequence sorting by Pt can change the jet ordering.
    if (writeUncertainties)
      corJ.setPileup(sqrt(sigmaSquared));

    coll->push_back(corJ);

    // Check whether the sequence remains sorted by pt
    const double pt = corJ.pt();
    if (pt > previousPt)
      sorted = false;
    previousPt = pt;

    if (verbose) {
      const reco::FFTJet<float>& fj = corJ.getFFTSpecific();
      std::cout << "     Fully corrected"
                << ": pt = " << corJ.pt() << ", eta = " << fj.f_vec().eta() << ", R = " << fj.f_recoScale()
                << ", s = 0x" << std::hex << fj.f_status() << std::dec << std::endl;
    }
  }

  if (!sorted)
    std::sort(coll->begin(), coll->end(), LocalSortByPt());

  // Create the uncertainty sequence
  if (writeUncertainties) {
    auto unc = std::make_unique<std::vector<float>>();
    unc->reserve(nJets);
    for (unsigned ijet = 0; ijet < nJets; ++ijet) {
      MyJet& j((*coll)[ijet]);
      unc->push_back(j.pileup());
      j.setPileup(0.f);
    }
    iEvent.put(std::move(unc), outputLabel);
  }

  iEvent.put(std::move(coll), outputLabel);
  ++eventCount;
}

//
// constructors and destructor
//
FFTJetCorrectionProducer::FFTJetCorrectionProducer(const edm::ParameterSet& ps)
    : inputLabel(ps.getParameter<edm::InputTag>("src")),
      outputLabel(ps.getParameter<std::string>("outputLabel")),
      jetType(parseJetType(ps.getParameter<std::string>("jetType"))),
      records(ps.getParameter<std::vector<std::string>>("records")),
      writeUncertainties(ps.getParameter<bool>("writeUncertainties")),
      subtractPileup(ps.getParameter<bool>("subtractPileup")),
      subtractPileupAs4Vec(ps.getParameter<bool>("subtractPileupAs4Vec")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)),
      eventCount(0UL) {
  const std::string alias(ps.getUntrackedParameter<std::string>("alias", outputLabel));
  jet_type_switch(makeProduces, alias, outputLabel);

  if (writeUncertainties)
    produces<std::vector<float>>(outputLabel).setBranchAlias(alias);

  input_jets_token_ = consumes<std::vector<reco::FFTAnyJet<reco::Jet>>>(inputLabel);
}

FFTJetCorrectionProducer::~FFTJetCorrectionProducer() {}

// ------------ method called to produce the data  ------------
void FFTJetCorrectionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  jet_type_switch(applyCorrections, iEvent, iSetup);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FFTJetCorrectionProducer);
