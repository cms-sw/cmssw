#include <vector>

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCrossCleaning.h"
#include "RecoTauTag/RecoTau/interface/ConeTools.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

namespace reco { namespace tau {

class RecoTauBuilderCombinatoricPlugin : public RecoTauBuilderPlugin {
  public:
    explicit RecoTauBuilderCombinatoricPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauBuilderCombinatoricPlugin() {}
    virtual return_type operator() (const reco::PFJetRef& jet,
         const std::vector<RecoTauPiZero>& piZeros,
         const std::vector<PFCandidatePtr>& regionalExtras) const;
  private:
    RecoTauQualityCuts qcuts_;
    bool usePFLeptonsAsChargedHadrons_;
    double isolationConeSize_;
    struct decayModeInfo {
      uint32_t maxPiZeros_;
      uint32_t maxPFCHs_;
      uint32_t nCharged_;
      uint32_t nPiZeros_;
    };
    std::vector<decayModeInfo> decayModesToBuild_;
};

RecoTauBuilderCombinatoricPlugin::RecoTauBuilderCombinatoricPlugin(
    const edm::ParameterSet& pset): RecoTauBuilderPlugin(pset),
  qcuts_(pset.getParameterSet(
        "qualityCuts").getParameterSet("signalQualityCuts")),
  usePFLeptonsAsChargedHadrons_(pset.getParameter<bool>("usePFLeptons")),
  isolationConeSize_(pset.getParameter<double>("isolationConeSize")) {
  typedef std::vector<edm::ParameterSet> VPSet;
  const VPSet& decayModes = pset.getParameter<VPSet>("decayModes");
  for (VPSet::const_iterator dm = decayModes.begin();
      dm != decayModes.end(); ++dm) {
    decayModeInfo info;
    info.nCharged_ = dm->getParameter<uint32_t>("nCharged");
    info.nPiZeros_ = dm->getParameter<uint32_t>("nPiZeros");
    info.maxPFCHs_ = dm->getParameter<uint32_t>("maxTracks");
    info.maxPiZeros_ = dm->getParameter<uint32_t>("maxPiZeros");
    decayModesToBuild_.push_back(info);
  }
}

namespace
{
  class SortPi0sDescendingPt {
    public:
      bool operator()(const RecoTauPiZero& a, const RecoTauPiZero& b) const {
        return a.pt() > b.pt();
      }  
  };
}

RecoTauBuilderCombinatoricPlugin::return_type
RecoTauBuilderCombinatoricPlugin::operator()(
    const reco::PFJetRef& jet,
    const std::vector<RecoTauPiZero>& piZeros,
    const std::vector<PFCandidatePtr>& regionalExtras) const 
{
  //std::cout << "<RecoTauBuilderCombinatoricPlugin::operator()>:" << std::endl;

  typedef std::vector<PFCandidatePtr> PFCandPtrs;
  typedef std::vector<RecoTauPiZero> PiZeroList;

  output_type output;
  reco::VertexRef primaryVertexRef = primaryVertex(jet);
 
  // Update the primary vertex used by the quality cuts.  The PV is supplied by
  // the base class.
  qcuts_.setPV(primaryVertexRef);

  // Get PFCHs from this jet.  They are already sorted by descending Pt
  PFCandPtrs pfchs;
  if (!usePFLeptonsAsChargedHadrons_) {
    pfchs = qcuts_.filterRefs(pfCandidates(*jet, reco::PFCandidate::h));
  } else {
    // Check if we want to include electrons in muons in "charged hadron"
    // collection.  This is the preferred behavior, as the PF lepton selections
    // are very loose.
    pfchs = qcuts_.filterRefs(pfChargedCands(*jet));
  }
  //std::cout << "#pfchs = " << pfchs.size() << std::endl;
  //int idx = 0;
  //for ( PFCandPtrs::const_iterator pfch = pfchs.begin();
  //	  pfch != pfchs.end(); ++pfch ) {
  //  std::cout << "pfch #" << idx << ": Pt = " << (*pfch)->pt() << ", eta = " << (*pfch)->eta() << ", phi = " << (*pfch)->phi() << std::endl;
  //  ++idx;
  //}
  //std::cout << "#piZeros = " << piZeros.size() << std::endl;
  //idx = 0;
  //for ( std::vector<RecoTauPiZero>::const_iterator piZero = piZeros.begin();
  //	  piZero != piZeros.end(); ++piZero ) {
  //  std::cout << "piZero #" << idx << ": Pt = " << piZero->pt() << ", eta = " << piZero->eta() << ", phi = " << piZero->phi() << std::endl;
  //  ++idx;
  //}

  PFCandPtrs pfnhs = qcuts_.filterRefs(
      pfCandidates(*jet, reco::PFCandidate::h0));

  /// Apply quality cuts to the regional junk around the jet.  Note that the
  /// particle contents of the junk is exclusive to the jet content.
  PFCandPtrs regionalJunk = qcuts_.filterRefs(regionalExtras);

  // Loop over the decay modes we want to build
  for (std::vector<decayModeInfo>::const_iterator
       decayMode = decayModesToBuild_.begin();
       decayMode != decayModesToBuild_.end(); ++decayMode) {
    // Find how many piZeros are in this decay mode
    size_t piZerosToBuild = decayMode->nPiZeros_;
    //std::cout << "piZerosToBuild = " << piZerosToBuild << std::endl;
    // Find how many tracks are in this decay mode
    size_t tracksToBuild = decayMode->nCharged_;
    //std::cout << "tracksToBuild = " << tracksToBuild << std::endl;

    // Skip decay mode if jet doesn't have the multiplicity to support it
    if (pfchs.size() < tracksToBuild)
      continue;

    // Find the start and end of potential signal tracks
    PFCandPtrs::iterator pfch_begin = pfchs.begin();
    PFCandPtrs::iterator pfch_end =  pfchs.end();
    pfch_end = takeNElements(pfch_begin, pfch_end, decayMode->maxPFCHs_);

    // Build our track combo generator
    typedef tau::CombinatoricGenerator<PFCandPtrs> PFCombo;
    PFCombo trackCombos(pfch_begin, pfch_end, tracksToBuild);

    /*
     * Begin combinatoric loop for this decay mode
     */
    // Loop over the different combinations of tracks
    for (PFCombo::iterator trackCombo = trackCombos.begin();
         trackCombo != trackCombos.end(); ++trackCombo) {
      xclean::CrossCleanPiZeros<PFCombo::combo_iterator>
        xCleaner(trackCombo->combo_begin(), trackCombo->combo_end());

      PiZeroList cleanPiZeros = xCleaner(piZeros);

      // CV: sort collection of cross-cleaned pi0s by descending Pt
      std::sort(cleanPiZeros.begin(), cleanPiZeros.end(), SortPi0sDescendingPt());

      // Skip decay mode if we don't have enough remaining clean pizeros to
      // build it.
      if (cleanPiZeros.size() < piZerosToBuild)
        continue;

      // Find the start and end of potential signal tracks
      PiZeroList::iterator piZero_begin = cleanPiZeros.begin();
      PiZeroList::iterator piZero_end = cleanPiZeros.end();

      piZero_end = takeNElements(piZero_begin, piZero_end,
                                 decayMode->maxPiZeros_);

      // Build our piZero combo generator
      typedef tau::CombinatoricGenerator<PiZeroList> PiZeroCombo;
      PiZeroCombo piZeroCombos(piZero_begin, piZero_end, piZerosToBuild);
      // Loop over the different combinations of PiZeros
      for (PiZeroCombo::iterator piZeroCombo = piZeroCombos.begin();
           piZeroCombo != piZeroCombos.end(); ++piZeroCombo) {
        // Output tau
        RecoTauConstructor tau(jet, getPFCands(), true);
        // Reserve space in our collections
        tau.reserve(RecoTauConstructor::kSignal,
                    RecoTauConstructor::kChargedHadron, tracksToBuild);
        tau.reserve(
            RecoTauConstructor::kSignal,
            RecoTauConstructor::kGamma, 2*piZerosToBuild); // k-factor = 2
        tau.reservePiZero(RecoTauConstructor::kSignal, piZerosToBuild);

        // FIXME - are all these reserves okay?  will they get propagated to the
        // dataformat size if they are wrong?
        tau.reserve(
            RecoTauConstructor::kIsolation,
            RecoTauConstructor::kChargedHadron, pfchs.size() - tracksToBuild);
        tau.reserve(RecoTauConstructor::kIsolation,
                    RecoTauConstructor::kGamma,
                    (cleanPiZeros.size() - piZerosToBuild)*2);
        tau.reservePiZero(RecoTauConstructor::kIsolation,
                          (cleanPiZeros.size() - piZerosToBuild));

        // Set signal and isolation components for charged hadrons, after
        // converting them to a PFCandidateRefVector
        tau.addPFCands(
            RecoTauConstructor::kSignal, RecoTauConstructor::kChargedHadron,
            trackCombo->combo_begin(), trackCombo->combo_end()
            );

        // Get signal PiZero constituents and add them to the tau.
        // The sub-gammas are automatically added.
        tau.addPiZeros(
            RecoTauConstructor::kSignal,
            piZeroCombo->combo_begin(), piZeroCombo->combo_end()
            );

        // Now build isolation collections
        // Load our isolation tools
        using namespace reco::tau::cone;
        PFCandPtrDRFilter isolationConeFilter(tau.p4(), 0, isolationConeSize_);

        // Cross cleaning predicate.  Remove any PFCandidatePtrs that are
        // contained within existing PiZeros.  This predicate will return false
        // for any object that overlaps with cleanPiZeros.
        xclean::CrossCleanPtrs pfCandXCleaner(cleanPiZeros);
        // And this cleaning filter predicate with our Iso cone filter
        xclean::PredicateAND<PFCandPtrDRFilter, xclean::CrossCleanPtrs>
          pfCandFilter(isolationConeFilter, pfCandXCleaner);

        PiZeroDRFilter isolationConeFilterPiZero(
            tau.p4(), 0, isolationConeSize_);

        // Additionally make predicates to select the different PF object types
        // of the regional junk objects to add
        typedef xclean::PredicateAND<xclean::FilterPFCandByParticleId,
                PFCandPtrDRFilter> RegionalJunkConeAndIdFilter;

        xclean::FilterPFCandByParticleId
          pfchCandSelector(reco::PFCandidate::h);
        xclean::FilterPFCandByParticleId
          pfgammaCandSelector(reco::PFCandidate::gamma);
        xclean::FilterPFCandByParticleId
          pfnhCandSelector(reco::PFCandidate::h0);

        RegionalJunkConeAndIdFilter pfChargedJunk(
            pfchCandSelector, // select charged stuff from junk
            isolationConeFilter // only take those in iso cone
            );

        RegionalJunkConeAndIdFilter pfGammaJunk(
            pfgammaCandSelector, // select gammas from junk
            isolationConeFilter // only take those in iso cone
            );

        RegionalJunkConeAndIdFilter pfNeutralJunk(
            pfnhCandSelector, // select neutral stuff from junk
            isolationConeFilter // select stuff in iso cone
            );

        // Filter the isolation candidates in a DR cone
        tau.addPFCands(
            RecoTauConstructor::kIsolation, RecoTauConstructor::kChargedHadron,
            boost::make_filter_iterator(
                pfCandFilter,
                trackCombo->remainder_begin(), trackCombo->remainder_end()),
            boost::make_filter_iterator(
                pfCandFilter,
                trackCombo->remainder_end(), trackCombo->remainder_end())
            );

        // Add all the candidates that weren't included in the combinatoric
        // generation
        tau.addPFCands(
            RecoTauConstructor::kIsolation, RecoTauConstructor::kChargedHadron,
            boost::make_filter_iterator(
                pfCandFilter,
                pfch_end, pfchs.end()),
            boost::make_filter_iterator(
                pfCandFilter,
                pfchs.end(), pfchs.end())
            );
        // Add all charged candidates that are in the iso cone but weren't in the
        // original PFJet
        tau.addPFCands(
            RecoTauConstructor::kIsolation, RecoTauConstructor::kChargedHadron,
            boost::make_filter_iterator(
                pfChargedJunk, regionalJunk.begin(), regionalJunk.end()),
            boost::make_filter_iterator(
                pfChargedJunk, regionalJunk.end(), regionalJunk.end())
            );

        // Add all gammas that are in the iso cone but weren't in the
        // orginal PFJet
        tau.addPFCands(
            RecoTauConstructor::kIsolation, RecoTauConstructor::kGamma,
            boost::make_filter_iterator(
                pfGammaJunk, regionalJunk.begin(), regionalJunk.end()),
            boost::make_filter_iterator(
                pfGammaJunk, regionalJunk.end(), regionalJunk.end())
            );

        // Add all the netural hadron candidates to the isolation collection
        tau.addPFCands(
            RecoTauConstructor::kIsolation, RecoTauConstructor::kNeutralHadron,
            boost::make_filter_iterator(
                pfCandFilter,
                pfnhs.begin(), pfnhs.end()),
            boost::make_filter_iterator(
                pfCandFilter,
                pfnhs.end(), pfnhs.end())
            );
        // Add all the netural hadrons from the region collection that are in
        // the iso cone to the tau
        tau.addPFCands(
            RecoTauConstructor::kIsolation,  RecoTauConstructor::kNeutralHadron,
            boost::make_filter_iterator(
              pfNeutralJunk, regionalJunk.begin(), regionalJunk.end()),
            boost::make_filter_iterator(
              pfNeutralJunk, regionalJunk.end(), regionalJunk.end())
            );

        tau.addPiZeros(
            RecoTauConstructor::kIsolation,
            boost::make_filter_iterator(
                isolationConeFilterPiZero,
                piZeroCombo->remainder_begin(), piZeroCombo->remainder_end()),
            boost::make_filter_iterator(
                isolationConeFilterPiZero,
                piZeroCombo->remainder_end(), piZeroCombo->remainder_end())
            );

        tau.addPiZeros(
            RecoTauConstructor::kIsolation,
            boost::make_filter_iterator(
                isolationConeFilterPiZero,
                piZero_end, cleanPiZeros.end()),
            boost::make_filter_iterator(
                isolationConeFilterPiZero,
                cleanPiZeros.end(), cleanPiZeros.end())
            );

        std::auto_ptr<reco::PFTau> tauPtr = tau.get(true);

	if ( primaryVertexRef.isNonnull() )
	  tauPtr->setVertex(primaryVertexRef->position());

        output.push_back(tauPtr);
      }
    }
  }
  return output.release();
}
}}  // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauBuilderPluginFactory,
                  reco::tau::RecoTauBuilderCombinatoricPlugin,
                  "RecoTauBuilderCombinatoricPlugin");
