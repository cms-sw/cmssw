#include <vector>

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCrossCleaning.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include <algorithm>

namespace reco {
  namespace tau {

    typedef std::vector<reco::PFRecoTauChargedHadron> ChargedHadronList;
    typedef tau::CombinatoricGenerator<ChargedHadronList> ChargedHadronCombo;
    typedef std::vector<RecoTauPiZero> PiZeroList;
    typedef tau::CombinatoricGenerator<PiZeroList> PiZeroCombo;

    class RecoTauBuilderCombinatoricPlugin : public RecoTauBuilderPlugin {
    public:
      explicit RecoTauBuilderCombinatoricPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);
      ~RecoTauBuilderCombinatoricPlugin() override {}

      return_type operator()(const reco::JetBaseRef&,
                             const std::vector<reco::PFRecoTauChargedHadron>&,
                             const std::vector<RecoTauPiZero>&,
                             const std::vector<CandidatePtr>&) const override;

    private:
      std::unique_ptr<RecoTauQualityCuts> qcuts_;

      double isolationConeSize_;

      struct decayModeInfo {
        uint32_t maxPiZeros_;
        uint32_t maxPFCHs_;
        uint32_t nCharged_;
        uint32_t nPiZeros_;
      };
      std::vector<decayModeInfo> decayModesToBuild_;

      StringObjectFunction<reco::PFTau> signalConeSize_;
      double minAbsPhotonSumPt_insideSignalCone_;
      double minRelPhotonSumPt_insideSignalCone_;
      double minAbsPhotonSumPt_outsideSignalCone_;
      double minRelPhotonSumPt_outsideSignalCone_;

      int verbosity_;
    };

    RecoTauBuilderCombinatoricPlugin::RecoTauBuilderCombinatoricPlugin(const edm::ParameterSet& pset,
                                                                       edm::ConsumesCollector&& iC)
        : RecoTauBuilderPlugin(pset, std::move(iC)),
          qcuts_(std::make_unique<RecoTauQualityCuts>(
              pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts"))),
          isolationConeSize_(pset.getParameter<double>("isolationConeSize")),
          signalConeSize_(pset.getParameter<std::string>("signalConeSize")),
          minAbsPhotonSumPt_insideSignalCone_(pset.getParameter<double>("minAbsPhotonSumPt_insideSignalCone")),
          minRelPhotonSumPt_insideSignalCone_(pset.getParameter<double>("minRelPhotonSumPt_insideSignalCone")),
          minAbsPhotonSumPt_outsideSignalCone_(pset.getParameter<double>("minAbsPhotonSumPt_outsideSignalCone")),
          minRelPhotonSumPt_outsideSignalCone_(pset.getParameter<double>("minRelPhotonSumPt_outsideSignalCone")) {
      typedef std::vector<edm::ParameterSet> VPSet;
      const VPSet& decayModes = pset.getParameter<VPSet>("decayModes");
      for (VPSet::const_iterator decayMode = decayModes.begin(); decayMode != decayModes.end(); ++decayMode) {
        decayModeInfo info;
        info.nCharged_ = decayMode->getParameter<uint32_t>("nCharged");
        info.nPiZeros_ = decayMode->getParameter<uint32_t>("nPiZeros");
        info.maxPFCHs_ = decayMode->getParameter<uint32_t>("maxTracks");
        info.maxPiZeros_ = decayMode->getParameter<uint32_t>("maxPiZeros");
        decayModesToBuild_.push_back(info);
      }

      verbosity_ = pset.getParameter<int>("verbosity");
    }

    // define template specialization for cross-cleaning
    namespace xclean {
      template <>
      inline void CrossCleanPiZeros<ChargedHadronCombo::combo_iterator>::initialize(
          const ChargedHadronCombo::combo_iterator& chargedHadronsBegin,
          const ChargedHadronCombo::combo_iterator& chargedHadronsEnd) {
        // Get the list of objects we need to clean
        for (ChargedHadronCombo::combo_iterator chargedHadron = chargedHadronsBegin; chargedHadron != chargedHadronsEnd;
             ++chargedHadron) {
          // CV: Remove PFGammas that are merged into TauChargedHadrons from isolation PiZeros, but not from signal PiZeros.
          //     The overlap between PFGammas contained in signal PiZeros and merged into TauChargedHadrons
          //     is resolved by RecoTauConstructor::addTauChargedHadron,
          //     which gives preference to PFGammas that are within PiZeros and removes those PFGammas from TauChargedHadrons.
          if (mode_ == kRemoveChargedDaughterOverlaps) {
            if (chargedHadron->getChargedPFCandidate().isNonnull())
              toRemove_.insert(reco::CandidatePtr(chargedHadron->getChargedPFCandidate()));
          } else if (mode_ == kRemoveChargedAndNeutralDaughterOverlaps) {
            const reco::CompositePtrCandidate::daughters& daughters = chargedHadron->daughterPtrVector();
            for (reco::CompositePtrCandidate::daughters::const_iterator daughter = daughters.begin();
                 daughter != daughters.end();
                 ++daughter) {
              toRemove_.insert(reco::CandidatePtr(*daughter));
            }
          } else
            assert(0);
        }
      }

      template <>
      inline void CrossCleanPtrs<PiZeroList::const_iterator>::initialize(const PiZeroList::const_iterator& piZerosBegin,
                                                                         const PiZeroList::const_iterator& piZerosEnd) {
        for (auto const& ptr : flattenPiZeros(piZerosBegin, piZerosEnd)) {
          toRemove_.insert(CandidatePtr(ptr));
        }
      }

      template <>
      inline void CrossCleanPtrs<ChargedHadronCombo::combo_iterator>::initialize(
          const ChargedHadronCombo::combo_iterator& chargedHadronsBegin,
          const ChargedHadronCombo::combo_iterator& chargedHadronsEnd) {
        //std::cout << "<CrossCleanPtrs<ChargedHadronCombo>::initialize>:" << std::endl;
        for (ChargedHadronCombo::combo_iterator chargedHadron = chargedHadronsBegin; chargedHadron != chargedHadronsEnd;
             ++chargedHadron) {
          const reco::CompositePtrCandidate::daughters& daughters = chargedHadron->daughterPtrVector();
          for (reco::CompositePtrCandidate::daughters::const_iterator daughter = daughters.begin();
               daughter != daughters.end();
               ++daughter) {
            //std::cout << " adding PFCandidate = " << daughter->id() << ":" << daughter->key() << std::endl;
            toRemove_.insert(reco::CandidatePtr(*daughter));
          }
        }
      }

      template <>
      inline void CrossCleanPtrs<ChargedHadronList::const_iterator>::initialize(
          const ChargedHadronList::const_iterator& chargedHadronsBegin,
          const ChargedHadronList::const_iterator& chargedHadronsEnd) {
        //std::cout << "<CrossCleanPtrs<ChargedHadronList>::initialize>:" << std::endl;
        for (ChargedHadronList::const_iterator chargedHadron = chargedHadronsBegin; chargedHadron != chargedHadronsEnd;
             ++chargedHadron) {
          const reco::CompositePtrCandidate::daughters& daughters = chargedHadron->daughterPtrVector();
          for (reco::CompositePtrCandidate::daughters::const_iterator daughter = daughters.begin();
               daughter != daughters.end();
               ++daughter) {
            //std::cout << " adding PFCandidate = " << daughter->id() << ":" << daughter->key() << std::endl;
            toRemove_.insert(reco::CandidatePtr(*daughter));
          }
        }
      }
    }  // namespace xclean

    namespace {
      // auxiliary class for sorting pizeros by descending transverse momentum
      class SortPi0sDescendingPt {
      public:
        bool operator()(const RecoTauPiZero& a, const RecoTauPiZero& b) const { return a.pt() > b.pt(); }
      };

      double square(double x) { return x * x; }
    }  // namespace

    RecoTauBuilderCombinatoricPlugin::return_type RecoTauBuilderCombinatoricPlugin::operator()(
        const reco::JetBaseRef& jet,
        const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons,
        const std::vector<RecoTauPiZero>& piZeros,
        const std::vector<CandidatePtr>& regionalExtras) const {
      if (verbosity_) {
        std::cout << "<RecoTauBuilderCombinatoricPlugin::operator()>:" << std::endl;
        std::cout << " processing jet: Pt = " << jet->pt() << ", eta = " << jet->eta() << ", phi = " << jet->eta()
                  << ","
                  << " mass = " << jet->mass() << ", area = " << jet->jetArea() << std::endl;
      }

      // Define output.
      output_type output;

      // Update the primary vertex used by the quality cuts.  The PV is supplied by
      // the base class.
      qcuts_->setPV(primaryVertex(jet));

      typedef std::vector<CandidatePtr> CandPtrs;

      if (verbosity_) {
        std::cout << "#chargedHadrons = " << chargedHadrons.size() << std::endl;
        int idx = 0;
        for (ChargedHadronList::const_iterator chargedHadron = chargedHadrons.begin();
             chargedHadron != chargedHadrons.end();
             ++chargedHadron) {
          std::cout << "chargedHadron #" << idx << ":" << std::endl;
          chargedHadron->print(std::cout);
          ++idx;
        }
        std::cout << "#piZeros = " << piZeros.size() << std::endl;
        idx = 0;
        for (PiZeroList::const_iterator piZero = piZeros.begin(); piZero != piZeros.end(); ++piZero) {
          std::cout << "piZero #" << idx << ":" << std::endl;
          piZero->print(std::cout);
          ++idx;
        }
      }

      CandPtrs pfchs = qcuts_->filterCandRefs(pfChargedCands(*jet));
      CandPtrs pfnhs = qcuts_->filterCandRefs(pfCandidatesByPdgId(*jet, 130));
      CandPtrs pfgammas = qcuts_->filterCandRefs(pfCandidatesByPdgId(*jet, 22));

      /// Apply quality cuts to the regional junk around the jet.  Note that the
      /// particle contents of the junk is exclusive to the jet content.
      CandPtrs regionalJunk = qcuts_->filterCandRefs(regionalExtras);

      // Loop over the decay modes we want to build
      for (std::vector<decayModeInfo>::const_iterator decayMode = decayModesToBuild_.begin();
           decayMode != decayModesToBuild_.end();
           ++decayMode) {
        // Find how many piZeros are in this decay mode
        size_t piZerosToBuild = decayMode->nPiZeros_;
        // Find how many tracks are in this decay mode
        size_t tracksToBuild = decayMode->nCharged_;
        if (verbosity_) {
          std::cout << "piZerosToBuild = " << piZerosToBuild << std::endl;
          std::cout << "#piZeros = " << piZeros.size() << std::endl;
          std::cout << "tracksToBuild = " << tracksToBuild << std::endl;
          std::cout << "#chargedHadrons = " << chargedHadrons.size() << std::endl;
        }

        // Skip decay mode if jet doesn't have the multiplicity to support it
        if (chargedHadrons.size() < tracksToBuild)
          continue;

        // Find the start and end of potential signal tracks
        ChargedHadronList::const_iterator chargedHadron_begin = chargedHadrons.begin();
        ChargedHadronList::const_iterator chargedHadron_end = chargedHadrons.end();
        chargedHadron_end = takeNElements(chargedHadron_begin, chargedHadron_end, decayMode->maxPFCHs_);

        // Build our track combo generator
        ChargedHadronCombo trackCombos(chargedHadron_begin, chargedHadron_end, tracksToBuild);

        CandPtrs::iterator pfch_end = pfchs.end();
        pfch_end = takeNElements(pfchs.begin(), pfch_end, decayMode->maxPFCHs_);

        //-------------------------------------------------------
        // Begin combinatoric loop for this decay mode
        //-------------------------------------------------------

        // Loop over the different combinations of tracks
        for (ChargedHadronCombo::iterator trackCombo = trackCombos.begin(); trackCombo != trackCombos.end();
             ++trackCombo) {
          xclean::CrossCleanPiZeros<ChargedHadronCombo::combo_iterator> signalPiZeroXCleaner(
              trackCombo->combo_begin(),
              trackCombo->combo_end(),
              xclean::CrossCleanPiZeros<ChargedHadronCombo::combo_iterator>::kRemoveChargedDaughterOverlaps);

          PiZeroList cleanSignalPiZeros = signalPiZeroXCleaner(piZeros);

          // CV: sort collection of cross-cleaned pi0s by descending Pt
          std::sort(cleanSignalPiZeros.begin(), cleanSignalPiZeros.end(), SortPi0sDescendingPt());

          // Skip decay mode if we don't have enough remaining clean pizeros to
          // build it.
          if (cleanSignalPiZeros.size() < piZerosToBuild)
            continue;

          // Find the start and end of potential signal tracks
          PiZeroList::iterator signalPiZero_begin = cleanSignalPiZeros.begin();
          PiZeroList::iterator signalPiZero_end = cleanSignalPiZeros.end();
          signalPiZero_end = takeNElements(signalPiZero_begin, signalPiZero_end, decayMode->maxPiZeros_);

          // Build our piZero combo generator
          PiZeroCombo piZeroCombos(signalPiZero_begin, signalPiZero_end, piZerosToBuild);
          // Loop over the different combinations of PiZeros
          for (PiZeroCombo::iterator piZeroCombo = piZeroCombos.begin(); piZeroCombo != piZeroCombos.end();
               ++piZeroCombo) {
            // Output tau
            RecoTauConstructor tau(jet,
                                   getPFCands(),
                                   true,
                                   &signalConeSize_,
                                   minAbsPhotonSumPt_insideSignalCone_,
                                   minRelPhotonSumPt_insideSignalCone_,
                                   minAbsPhotonSumPt_outsideSignalCone_,
                                   minRelPhotonSumPt_outsideSignalCone_);
            // Reserve space in our collections
            tau.reserve(RecoTauConstructor::kSignal, RecoTauConstructor::kChargedHadron, tracksToBuild);
            tau.reserve(RecoTauConstructor::kSignal, RecoTauConstructor::kGamma, 2 * piZerosToBuild);  // k-factor = 2
            tau.reservePiZero(RecoTauConstructor::kSignal, piZerosToBuild);

            xclean::CrossCleanPiZeros<ChargedHadronCombo::combo_iterator> isolationPiZeroXCleaner(
                trackCombo->combo_begin(),
                trackCombo->combo_end(),
                xclean::CrossCleanPiZeros<ChargedHadronCombo::combo_iterator>::kRemoveChargedAndNeutralDaughterOverlaps);

            PiZeroList precleanedIsolationPiZeros = isolationPiZeroXCleaner(piZeros);
            std::set<reco::CandidatePtr> toRemove;
            for (PiZeroCombo::combo_iterator signalPiZero = piZeroCombo->combo_begin();
                 signalPiZero != piZeroCombo->combo_end();
                 ++signalPiZero) {
              toRemove.insert(signalPiZero->daughterPtrVector().begin(), signalPiZero->daughterPtrVector().end());
            }
            PiZeroList cleanIsolationPiZeros;
            for (auto const& precleanedPiZero : precleanedIsolationPiZeros) {
              std::set<reco::CandidatePtr> toCheck(precleanedPiZero.daughterPtrVector().begin(),
                                                   precleanedPiZero.daughterPtrVector().end());
              std::vector<reco::CandidatePtr> cleanDaughters;
              std::set_difference(
                  toCheck.begin(), toCheck.end(), toRemove.begin(), toRemove.end(), std::back_inserter(cleanDaughters));
              // CV: piZero is signal piZero if at least one daughter overlaps
              if (cleanDaughters.size() == precleanedPiZero.daughterPtrVector().size()) {
                cleanIsolationPiZeros.push_back(precleanedPiZero);
              }
            }
            if (verbosity_) {
              std::cout << "#cleanIsolationPiZeros = " << cleanIsolationPiZeros.size() << std::endl;
              int idx = 0;
              for (PiZeroList::const_iterator piZero = cleanIsolationPiZeros.begin();
                   piZero != cleanIsolationPiZeros.end();
                   ++piZero) {
                std::cout << "piZero #" << idx << ":" << std::endl;
                piZero->print(std::cout);
                ++idx;
              }
            }

            // FIXME - are all these reserves okay?  will they get propagated to the
            // dataformat size if they are wrong?
            tau.reserve(RecoTauConstructor::kIsolation,
                        RecoTauConstructor::kChargedHadron,
                        chargedHadrons.size() - tracksToBuild);
            tau.reserve(
                RecoTauConstructor::kIsolation, RecoTauConstructor::kGamma, (piZeros.size() - piZerosToBuild) * 2);
            tau.reservePiZero(RecoTauConstructor::kIsolation, (piZeros.size() - piZerosToBuild));

            // Get signal PiZero constituents and add them to the tau.
            // The sub-gammas are automatically added.
            tau.addPiZeros(RecoTauConstructor::kSignal, piZeroCombo->combo_begin(), piZeroCombo->combo_end());

            // Set signal and isolation components for charged hadrons, after
            // converting them to a PFCandidateRefVector
            //
            // NOTE: signal ChargedHadrons need to be added **after** signal PiZeros
            //       to avoid double-counting PFGammas as part of PiZero and merged with ChargedHadron
            //
            tau.addTauChargedHadrons(RecoTauConstructor::kSignal, trackCombo->combo_begin(), trackCombo->combo_end());

            // Now build isolation collections
            // Load our isolation tools
            using namespace reco::tau::cone;
            CandPtrDRFilter isolationConeFilter(tau.p4(), -0.1, isolationConeSize_);

            // Cross cleaning predicate: Remove any PFCandidatePtrs that are contained within existing ChargedHadrons or PiZeros.
            // The predicate will return false for any object that overlaps with chargedHadrons or cleanPiZeros.
            //  1.) to select charged PFCandidates within jet that are not signalPFChargedHadrons
            typedef xclean::CrossCleanPtrs<ChargedHadronCombo::combo_iterator> pfChargedHadronXCleanerType;
            pfChargedHadronXCleanerType pfChargedHadronXCleaner_comboChargedHadrons(trackCombo->combo_begin(),
                                                                                    trackCombo->combo_end());
            // And this cleaning filter predicate with our Iso cone filter
            xclean::PredicateAND<CandPtrDRFilter, pfChargedHadronXCleanerType> pfCandFilter_comboChargedHadrons(
                isolationConeFilter, pfChargedHadronXCleaner_comboChargedHadrons);
            //  2.) to select neutral PFCandidates within jet
            xclean::CrossCleanPtrs<ChargedHadronList::const_iterator> pfChargedHadronXCleaner_allChargedHadrons(
                chargedHadrons.begin(), chargedHadrons.end());
            xclean::CrossCleanPtrs<PiZeroList::const_iterator> piZeroXCleaner(piZeros.begin(), piZeros.end());
            typedef xclean::PredicateAND<xclean::CrossCleanPtrs<ChargedHadronList::const_iterator>,
                                         xclean::CrossCleanPtrs<PiZeroList::const_iterator> >
                pfCandXCleanerType;
            pfCandXCleanerType pfCandXCleaner_allChargedHadrons(pfChargedHadronXCleaner_allChargedHadrons,
                                                                piZeroXCleaner);
            // And this cleaning filter predicate with our Iso cone filter
            xclean::PredicateAND<CandPtrDRFilter, pfCandXCleanerType> pfCandFilter_allChargedHadrons(
                isolationConeFilter, pfCandXCleaner_allChargedHadrons);

            ChargedHadronDRFilter isolationConeFilterChargedHadron(tau.p4(), -0.1, isolationConeSize_);
            PiZeroDRFilter isolationConeFilterPiZero(tau.p4(), -0.1, isolationConeSize_);

            // Additionally make predicates to select the different PF object types
            // of the regional junk objects to add
            typedef xclean::PredicateAND<xclean::FilterCandByAbsPdgId, CandPtrDRFilter> RegionalJunkConeAndIdFilter;

            xclean::FilterCandByAbsPdgId pfchCandSelector(211);
            xclean::FilterCandByAbsPdgId pfgammaCandSelector(22);
            xclean::FilterCandByAbsPdgId pfnhCandSelector(130);

            RegionalJunkConeAndIdFilter pfChargedJunk(pfchCandSelector,      // select charged stuff from junk
                                                      isolationConeFilter);  // only take those in iso cone

            RegionalJunkConeAndIdFilter pfGammaJunk(pfgammaCandSelector,   // select gammas from junk
                                                    isolationConeFilter);  // only take those in iso cone

            RegionalJunkConeAndIdFilter pfNeutralJunk(pfnhCandSelector,      // select neutral stuff from junk
                                                      isolationConeFilter);  // select stuff in iso cone

            tau.addPiZeros(RecoTauConstructor::kIsolation,
                           boost::make_filter_iterator(
                               isolationConeFilterPiZero, cleanIsolationPiZeros.begin(), cleanIsolationPiZeros.end()),
                           boost::make_filter_iterator(
                               isolationConeFilterPiZero, cleanIsolationPiZeros.end(), cleanIsolationPiZeros.end()));

            // Filter the isolation candidates in a DR cone
            //
            // NOTE: isolation ChargedHadrons need to be added **after** signal and isolation PiZeros
            //       to avoid double-counting PFGammas as part of PiZero and merged with ChargedHadron
            //
            if (verbosity_ >= 2) {
              std::cout << "adding isolation PFChargedHadrons from trackCombo:" << std::endl;
            }
            tau.addTauChargedHadrons(
                RecoTauConstructor::kIsolation,
                boost::make_filter_iterator(
                    isolationConeFilterChargedHadron, trackCombo->remainder_begin(), trackCombo->remainder_end()),
                boost::make_filter_iterator(
                    isolationConeFilterChargedHadron, trackCombo->remainder_end(), trackCombo->remainder_end()));

            // Add all the candidates that weren't included in the combinatoric
            // generation
            if (verbosity_ >= 2) {
              std::cout << "adding isolation PFChargedHadrons not considered in trackCombo:" << std::endl;
            }
            tau.addPFCands(RecoTauConstructor::kIsolation,
                           RecoTauConstructor::kChargedHadron,
                           boost::make_filter_iterator(pfCandFilter_comboChargedHadrons, pfch_end, pfchs.end()),
                           boost::make_filter_iterator(pfCandFilter_comboChargedHadrons, pfchs.end(), pfchs.end()));
            // Add all charged candidates that are in the iso cone but weren't in the
            // original PFJet
            if (verbosity_ >= 2) {
              std::cout << "adding isolation PFChargedHadrons from 'regional junk':" << std::endl;
            }
            tau.addPFCands(RecoTauConstructor::kIsolation,
                           RecoTauConstructor::kChargedHadron,
                           boost::make_filter_iterator(pfChargedJunk, regionalJunk.begin(), regionalJunk.end()),
                           boost::make_filter_iterator(pfChargedJunk, regionalJunk.end(), regionalJunk.end()));

            // Add all PFGamma constituents of the jet that are not part of a PiZero
            if (verbosity_ >= 2) {
              std::cout << "adding isolation PFGammas not considered in PiZeros:" << std::endl;
            }
            tau.addPFCands(
                RecoTauConstructor::kIsolation,
                RecoTauConstructor::kGamma,
                boost::make_filter_iterator(pfCandFilter_allChargedHadrons, pfgammas.begin(), pfgammas.end()),
                boost::make_filter_iterator(pfCandFilter_allChargedHadrons, pfgammas.end(), pfgammas.end()));
            // Add all gammas that are in the iso cone but weren't in the
            // orginal PFJet
            tau.addPFCands(RecoTauConstructor::kIsolation,
                           RecoTauConstructor::kGamma,
                           boost::make_filter_iterator(pfGammaJunk, regionalJunk.begin(), regionalJunk.end()),
                           boost::make_filter_iterator(pfGammaJunk, regionalJunk.end(), regionalJunk.end()));

            // Add all the neutral hadron candidates to the isolation collection
            tau.addPFCands(RecoTauConstructor::kIsolation,
                           RecoTauConstructor::kNeutralHadron,
                           boost::make_filter_iterator(pfCandFilter_allChargedHadrons, pfnhs.begin(), pfnhs.end()),
                           boost::make_filter_iterator(pfCandFilter_allChargedHadrons, pfnhs.end(), pfnhs.end()));
            // Add all the neutral hadrons from the region collection that are in
            // the iso cone to the tau
            tau.addPFCands(RecoTauConstructor::kIsolation,
                           RecoTauConstructor::kNeutralHadron,
                           boost::make_filter_iterator(pfNeutralJunk, regionalJunk.begin(), regionalJunk.end()),
                           boost::make_filter_iterator(pfNeutralJunk, regionalJunk.end(), regionalJunk.end()));

            std::unique_ptr<reco::PFTau> tauPtr = tau.get(true);

            // Set event vertex position for tau
            reco::VertexRef primaryVertexRef = primaryVertex(*tauPtr);
            if (primaryVertexRef.isNonnull()) {
              tauPtr->setVertex(primaryVertexRef->position());
            }

            double tauEn = tauPtr->energy();
            double tauPz = tauPtr->pz();
            const double chargedPionMass = 0.13957;  // GeV
            double tauMass = std::max(tauPtr->mass(), chargedPionMass);
            double bendCorrMass2 = 0.;
            const std::vector<RecoTauPiZero>& piZeros = tauPtr->signalPiZeroCandidates();
            for (auto const& piZero : piZeros) {
              double piZeroEn = piZero.energy();
              double piZeroPx = piZero.px();
              double piZeroPy = piZero.py();
              double piZeroPz = piZero.pz();
              double tau_wo_piZeroPx = tauPtr->px() - piZeroPx;
              double tau_wo_piZeroPy = tauPtr->py() - piZeroPy;
              // CV: Compute effect of varying strip four-vector by eta and phi correction on tau mass
              //    (derrivative of tau mass by strip eta, phi has been computed using Mathematica)
              bendCorrMass2 += square(((piZeroPz * tauEn - piZeroEn * tauPz) / tauMass) * piZero.bendCorrEta());
              bendCorrMass2 +=
                  square(((piZeroPy * tau_wo_piZeroPx - piZeroPx * tau_wo_piZeroPy) / tauMass) * piZero.bendCorrPhi());
            }
            //edm::LogPrint("RecoTauBuilderCombinatoricPlugin") << "bendCorrMass2 = " << sqrt(bendCorrMass2) << std::endl;
            tauPtr->setBendCorrMass(sqrt(bendCorrMass2));

            output.push_back(std::move(tauPtr));
          }
        }
      }

      return output;
    }

  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauBuilderPluginFactory,
                  reco::tau::RecoTauBuilderCombinatoricPlugin,
                  "RecoTauBuilderCombinatoricPlugin");
