/*
 * RecoTauBuilderConePlugin
 *
 * Build a PFTau using cones defined in DeltaR.
 *
 * Original Authors: Ludovic Houchu, Simone Gennai
 * Modifications: Evan K. Friis
 *
 * $Id $
 */

#include <vector>
#include <algorithm>

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include "RecoTauTag/RecoTau/interface/ConeTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCrossCleaning.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"

namespace reco { namespace tau {

class RecoTauBuilderConePlugin : public RecoTauBuilderPlugin {
  public:
    explicit RecoTauBuilderConePlugin(const edm::ParameterSet& pset);
    ~RecoTauBuilderConePlugin() {}
    // Build a tau from a jet
    return_type operator()(const reco::PFJetRef& jet,
        const std::vector<RecoTauPiZero>& piZeros) const;
  private:
    RecoTauQualityCuts qcuts_;
    bool usePFLeptonsAsChargedHadrons_;
    double leadObjecPtThreshold_;
    /* String function to extract values from PFJets */
    typedef StringObjectFunction<reco::PFJet> JetFunc;
    // Cone defintions
    JetFunc matchingCone_;
    JetFunc signalConeChargedHadrons_;
    JetFunc isoConeChargedHadrons_;
    JetFunc signalConePiZeros_;
    JetFunc isoConePiZeros_;
    JetFunc signalConeNeutralHadrons_;
    JetFunc isoConeNeutralHadrons_;
};

// ctor - initialize all of our variables
RecoTauBuilderConePlugin::RecoTauBuilderConePlugin(
    const edm::ParameterSet& pset):RecoTauBuilderPlugin(pset),
    qcuts_(pset.getParameter<edm::ParameterSet>("qualityCuts")),
    usePFLeptonsAsChargedHadrons_(pset.getParameter<bool>("usePFLeptons")),
    leadObjecPtThreshold_(pset.getParameter<double>("leadObjectPt")),
    matchingCone_(pset.getParameter<std::string>("matchingCone")),
    signalConeChargedHadrons_(pset.getParameter<std::string>(
            "signalConeChargedHadrons")),
    isoConeChargedHadrons_(
        pset.getParameter<std::string>("isoConeChargedHadrons")),
    signalConePiZeros_(
        pset.getParameter<std::string>("signalConePiZeros")),
    isoConePiZeros_(
        pset.getParameter<std::string>("isoConePiZeros")),
    signalConeNeutralHadrons_(
        pset.getParameter<std::string>("signalConeNeutralHadrons")),
    isoConeNeutralHadrons_(
        pset.getParameter<std::string>("isoConeNeutralHadrons"))
{}

RecoTauBuilderConePlugin::return_type RecoTauBuilderConePlugin::operator()(
    const reco::PFJetRef& jet,
    const std::vector<RecoTauPiZero>& piZeros) const {
  // Get access to our cone tools
  using namespace cone;
  // Define output.  We only produce one tau per jet for the cone algo.
  output_type output;

  // Our tau builder - the true indicates to automatically copy gamma candidates
  // from the pizeros.
  RecoTauConstructor tau(jet, getPFCands(), true);
  // Setup our quality cuts to use the current vertex (supplied by base class)
  qcuts_.setPV(primaryVertex());

  typedef std::vector<PFCandidatePtr> PFCandPtrs;

  // Get the PF Charged hadrons + quality cuts
  PFCandPtrs pfchs;
  if (!usePFLeptonsAsChargedHadrons_) {
    pfchs = qcuts_.filterRefs(pfCandidates(*jet, reco::PFCandidate::h));
  } else {
    // Check if we want to include electrons in muons in "charged hadron"
    // collection.  This is the preferred behavior, as the PF lepton selections
    // are very loose.
    pfchs = qcuts_.filterRefs(pfChargedCands(*jet));
  }

  // Get the PF gammas
  PFCandPtrs pfGammas = qcuts_.filterRefs(
      pfCandidates(*jet, reco::PFCandidate::gamma));
  // Neutral hadrons
  PFCandPtrs pfnhs = qcuts_.filterRefs(
      pfCandidates(*jet, reco::PFCandidate::h0));

  /***********************************************
   ******     Lead Candidate Finding    **********
   ***********************************************/

  // Define our matching cone and filters
  double matchingCone = matchingCone_(*jet);
  PFCandPtrDRFilter matchingConeFilter(jet->p4(), 0, matchingCone);

  // Find the maximum PFCharged hadron in the matching cone.  The call to
  // PFCandidates always a sorted list, so we can just take the first if it
  // if it exists.
  PFCandidatePtr leadPFCH;
  PFCandPtrs::iterator leadPFCH_iter =
      std::find_if(pfchs.begin(), pfchs.end(), matchingConeFilter);

  if (leadPFCH_iter != pfchs.end()) {
    leadPFCH = *leadPFCH_iter;
    // Set leading candidate
    tau.setleadPFChargedHadrCand(leadPFCH);
  } else {
    // If there is no leading charged candidate at all, return nothing - the
    // producer class that owns the plugin will build a null tau if desired.
    return output.release();
  }

  // Find the leading neutral candidate
  PFCandidatePtr leadPFGamma;
  PFCandPtrs::iterator leadPFGamma_iter =
      std::find_if(pfGammas.begin(), pfGammas.end(), matchingConeFilter);

  if (leadPFGamma_iter != pfGammas.end()) {
    leadPFGamma = *leadPFGamma_iter;
    // Set leading neutral candidate
    tau.setleadPFNeutralCand(leadPFGamma);
  }

  PFCandidatePtr leadPFCand;
  // Always use the leadPFCH if it is above our threshold
  if (leadPFCH.isNonnull() && leadPFCH->pt() > leadObjecPtThreshold_) {
    leadPFCand = leadPFCH;
  } else if (leadPFGamma.isNonnull() &&
             leadPFGamma->pt() > leadObjecPtThreshold_) {
    // Otherwise use the lead Gamma if it is above threshold
    leadPFCand = leadPFGamma;
  } else {
    // If both are too low PT, just take the charged one
    leadPFCand = leadPFCH;
  }

  tau.setleadPFCand(leadPFCand);

  // Our cone axis is defined about the lead charged hadron
  reco::Candidate::LorentzVector coneAxis = leadPFCH->p4();

  /***********************************************
   ******     Cone Construction         **********
   ***********************************************/

  // Define the signal and isolation cone sizes for this jet and build filters
  // to select elements in the given DeltaR regions

  PFCandPtrDRFilter signalConePFCHFilter(
      coneAxis, -0.1, signalConeChargedHadrons_(*jet));
  PFCandPtrDRFilter signalConePFNHFilter(
      coneAxis, -0.1, signalConeNeutralHadrons_(*jet));
  PiZeroDRFilter signalConePiZeroFilter(
      coneAxis, -0.1, signalConePiZeros_(*jet));

  PFCandPtrDRFilter isoConePFCHFilter(
      coneAxis, signalConeChargedHadrons_(*jet), isoConeChargedHadrons_(*jet));
  PFCandPtrDRFilter isoConePFNHFilter(
      coneAxis, signalConeNeutralHadrons_(*jet), isoConeNeutralHadrons_(*jet));
  PiZeroDRFilter isoConePiZeroFilter(
      coneAxis, signalConePiZeros_(*jet), isoConePiZeros_(*jet));

  // Build filter iterators select the signal charged stuff.
  PFCandPtrDRFilterIter signalPFCHs_begin(signalConePFCHFilter, pfchs.begin(), pfchs.end());
  PFCandPtrDRFilterIter signalPFCHs_end(signalConePFCHFilter, pfchs.end(), pfchs.end());

  // Cross clean pi zero content using signal cone charged hadron constituents.
  xclean::CrossCleanPiZeros<PFCandPtrDRFilterIter> xCleaner(
      signalPFCHs_begin, signalPFCHs_end);
  std::vector<reco::RecoTauPiZero> cleanPiZeros = xCleaner(piZeros);

  // For the rest of the constituents, we need to filter any constituents that
  // are already contained in the pizeros (i.e. electrons)
  xclean::CrossCleanPtrs pfCandXCleaner(cleanPiZeros);

  // Build signal charged hadrons
  tau.addPFCands(RecoTauConstructor::kSignal,
                 RecoTauConstructor::kChargedHadron,
                 signalPFCHs_begin, signalPFCHs_end);

  tau.addPFCands(RecoTauConstructor::kSignal,
                 RecoTauConstructor::kNeutralHadron,
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(signalConePFNHFilter, pfCandXCleaner),
                   pfnhs.begin(), pfnhs.end()),
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(signalConePFNHFilter, pfCandXCleaner),
                   pfnhs.end(), pfnhs.end()));

  // Build signal PiZeros
  tau.addPiZeros(RecoTauConstructor::kSignal,
                 PiZeroDRFilterIter(signalConePiZeroFilter,
                                    cleanPiZeros.begin(), cleanPiZeros.end()),
                 PiZeroDRFilterIter(signalConePiZeroFilter,
                                    cleanPiZeros.end(), cleanPiZeros.end()));

  // Build isolation charged hadrons
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kChargedHadron,
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFCHFilter, pfCandXCleaner),
                   pfchs.begin(), pfchs.end()),
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFCHFilter, pfCandXCleaner),
                   pfchs.end(), pfchs.end()));

  // Build isolation neutral hadrons
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kNeutralHadron,
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFNHFilter, pfCandXCleaner),
                   pfnhs.begin(), pfnhs.end()),
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFNHFilter, pfCandXCleaner),
                   pfnhs.end(), pfnhs.end()));

  // Build isolation PiZeros
  tau.addPiZeros(RecoTauConstructor::kIsolation,
                 PiZeroDRFilterIter(isoConePiZeroFilter, cleanPiZeros.begin(),
                                    cleanPiZeros.end()),
                 PiZeroDRFilterIter(isoConePiZeroFilter, cleanPiZeros.end(),
                                    cleanPiZeros.end()));

  // Put our built tau in the output - 'false' indicates don't build the
  // leading candidtes, we already did that explicitly above.

  output.push_back(tau.get(false));
  return output.release();
}
}}  // end namespace reco::tauk

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauBuilderPluginFactory,
                  reco::tau::RecoTauBuilderConePlugin,
                  "RecoTauBuilderConePlugin");
