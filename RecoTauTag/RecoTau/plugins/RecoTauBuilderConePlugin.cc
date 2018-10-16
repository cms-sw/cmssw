/*
 * RecoTauBuilderConePlugin
 *
 * Build a PFTau using cones defined in DeltaR.
 *
 * Original Authors: Ludovic Houchu, Simone Gennai
 * Modifications: Evan K. Friis
 *
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
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"

namespace reco { namespace tau {

typedef std::vector<RecoTauPiZero> PiZeroList;
  
class RecoTauBuilderConePlugin : public RecoTauBuilderPlugin {
  public:
  explicit RecoTauBuilderConePlugin(const edm::ParameterSet& pset,edm::ConsumesCollector &&iC);
    ~RecoTauBuilderConePlugin() override {}
    // Build a tau from a jet
    return_type operator()(const reco::PFJetRef& jet,
	const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons,
        const std::vector<RecoTauPiZero>& piZeros,
        const std::vector<PFCandidatePtr>& regionalExtras) const override;
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
    StringObjectFunction<reco::PFTau> signalConeSizeToStore_;
    JetFunc isoConePiZeros_;
    JetFunc signalConeNeutralHadrons_;
    JetFunc isoConeNeutralHadrons_;

    int maxSignalConeChargedHadrons_;
    double minAbsPhotonSumPt_insideSignalCone_;
    double minRelPhotonSumPt_insideSignalCone_;

    void setTauQuantities(reco::PFTau& aTau,
			  double minAbsPhotonSumPt_insideSignalCone = 2.5,
			  double minRelPhotonSumPt_insideSignalCone = 0.,
			  double minAbsPhotonSumPt_outsideSignalCone = 1.e+9,
			  double minRelPhotonSumPt_outsideSignalCone = 1.e+9) const;

};

// ctor - initialize all of our variables
RecoTauBuilderConePlugin::RecoTauBuilderConePlugin(
						   const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauBuilderPlugin(pset,std::move(iC)),
    qcuts_(pset.getParameterSet(
          "qualityCuts").getParameterSet("signalQualityCuts")),
    usePFLeptonsAsChargedHadrons_(pset.getParameter<bool>("usePFLeptons")),
    leadObjecPtThreshold_(pset.getParameter<double>("leadObjectPt")),
    matchingCone_(pset.getParameter<std::string>("matchingCone")),
    signalConeChargedHadrons_(pset.getParameter<std::string>(
            "signalConeChargedHadrons")),
    isoConeChargedHadrons_(
        pset.getParameter<std::string>("isoConeChargedHadrons")),
    signalConePiZeros_(
        pset.getParameter<std::string>("signalConePiZeros")),
    signalConeSizeToStore_(//MB: stored inside PFTau and used to compte photon-pt-out-of-signal-cone and DM hypothsis
	pset.getParameter<std::string>("signalConePiZeros")),
    isoConePiZeros_(
        pset.getParameter<std::string>("isoConePiZeros")),
    signalConeNeutralHadrons_(
        pset.getParameter<std::string>("signalConeNeutralHadrons")),
    isoConeNeutralHadrons_(
        pset.getParameter<std::string>("isoConeNeutralHadrons")), 
    maxSignalConeChargedHadrons_(
        pset.getParameter<int>("maxSignalConeChargedHadrons")),
    minAbsPhotonSumPt_insideSignalCone_(
        pset.getParameter<double>("minAbsPhotonSumPt_insideSignalCone")),
    minRelPhotonSumPt_insideSignalCone_(
	pset.getParameter<double>("minRelPhotonSumPt_insideSignalCone"))

{}

namespace xclean 
{ 
  // define template specialization for cross-cleaning
  template<>
  inline void CrossCleanPiZeros<cone::PFCandPtrDRFilterIter>::initialize(const cone::PFCandPtrDRFilterIter& signalTracksBegin, const cone::PFCandPtrDRFilterIter& signalTracksEnd) 
  {
    // Get the list of objects we need to clean
    for ( cone::PFCandPtrDRFilterIter signalTrack = signalTracksBegin; signalTrack != signalTracksEnd; ++signalTrack ) {
      toRemove_.insert(reco::CandidatePtr(*signalTrack));
    }
  }
  
  template<>
  inline void CrossCleanPtrs<PiZeroList::const_iterator>::initialize(const PiZeroList::const_iterator& piZerosBegin, const PiZeroList::const_iterator& piZerosEnd) 
  {
    for(auto const& ptr : flattenPiZeros(piZerosBegin, piZerosEnd) ) {
      toRemove_.insert(CandidatePtr(ptr));
    }
  }
}

void RecoTauBuilderConePlugin::setTauQuantities(reco::PFTau& aTau,
						double minAbsPhotonSumPt_insideSignalCone,
						double minRelPhotonSumPt_insideSignalCone,
						double minAbsPhotonSumPt_outsideSignalCone,
						double minRelPhotonSumPt_outsideSignalCone) const {
  //MB: Set tau quantities which are computed by RecoTauConstructor::get() 
  //    method with PFRecoTauChargedHadrons not used here
  
  // Set charge
  int charge = 0;
  int leadCharge = aTau.leadPFChargedHadrCand().isNonnull() ? aTau.leadPFChargedHadrCand()->charge() : 0;
  const std::vector<reco::PFCandidatePtr>& pfChs = aTau.signalPFChargedHadrCands();
  for(const reco::PFCandidatePtr& pfCh : pfChs){
    charge += pfCh->charge();
  }
  charge = charge==0 ? leadCharge : charge;
  aTau.setCharge(charge);
  
  // Set PDG id
  aTau.setPdgId(aTau.charge() < 0 ? 15 : -15); 
  
  // Set Decay Mode
  PFTau::hadronicDecayMode dm = PFTau::kNull; 
  unsigned int nPiZeros = 0;
  const std::vector<RecoTauPiZero>& piZeros = aTau.signalPiZeroCandidates();
  for(const RecoTauPiZero& piZero : piZeros) {
    double photonSumPt_insideSignalCone = 0.;
    double photonSumPt_outsideSignalCone = 0.;
    int numPhotons = piZero.numberOfDaughters();
    for(int idxPhoton = 0; idxPhoton < numPhotons; ++idxPhoton) {
      const reco::Candidate* photon = piZero.daughter(idxPhoton);
      double dR = deltaR(photon->p4(), aTau.p4());
      if(dR < aTau.signalConeSize() ){
	photonSumPt_insideSignalCone += photon->pt();
      } else {
	photonSumPt_outsideSignalCone += photon->pt();
      }
    }
    if( photonSumPt_insideSignalCone  > minAbsPhotonSumPt_insideSignalCone  || photonSumPt_insideSignalCone  > (minRelPhotonSumPt_insideSignalCone*aTau.pt())  ||
	photonSumPt_outsideSignalCone > minAbsPhotonSumPt_outsideSignalCone || photonSumPt_outsideSignalCone > (minRelPhotonSumPt_outsideSignalCone*aTau.pt()) ) ++nPiZeros;
  }
  // Find the maximum number of PiZeros our parameterization can hold
  const unsigned int maxPiZeros = PFTau::kOneProngNPiZero;
  // Determine our track index
  unsigned int nCharged = pfChs.size();
  if(nCharged>0){
    unsigned int trackIndex = (nCharged - 1)*(maxPiZeros + 1);
    // Check if we handle the given number of tracks
    if(trackIndex >= PFTau::kRareDecayMode) 
      dm = PFTau::kRareDecayMode;    
    else
      dm = static_cast<PFTau::hadronicDecayMode>(trackIndex + std::min(nPiZeros,maxPiZeros) );
  }
  else{
    dm = PFTau::kNull;
  }
  aTau.setDecayMode(dm);
  return;
}

RecoTauBuilderConePlugin::return_type RecoTauBuilderConePlugin::operator()(
    const reco::PFJetRef& jet,
    const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons, 
    const std::vector<RecoTauPiZero>& piZeros,
    const std::vector<PFCandidatePtr>& regionalExtras) const {
  //std::cout << "<RecoTauBuilderConePlugin::operator()>:" << std::endl;
  //std::cout << " jet: Pt = " << jet->pt() << ", eta = " << jet->eta() << ", phi = " << jet->phi() << std::endl;

  // Get access to our cone tools
  using namespace cone;
  // Define output.  We only produce one tau per jet for the cone algo.
  output_type output;

  // Our tau builder - the true indicates to automatically copy gamma candidates
  // from the pizeros.
  RecoTauConstructor tau(jet, getPFCands(), true, 
			 &signalConeSizeToStore_, 
			 minAbsPhotonSumPt_insideSignalCone_, 
			 minRelPhotonSumPt_insideSignalCone_);
  // Setup our quality cuts to use the current vertex (supplied by base class)
  qcuts_.setPV(primaryVertex(jet));

  typedef std::vector<PFCandidatePtr> PFCandPtrs;

  // Get the PF Charged hadrons + quality cuts
  PFCandPtrs pfchs;
  if (!usePFLeptonsAsChargedHadrons_) {
    pfchs = qcuts_.filterCandRefs(pfCandidates(*jet, reco::PFCandidate::h));
  } else {
    // Check if we want to include electrons in muons in "charged hadron"
    // collection.  This is the preferred behavior, as the PF lepton selections
    // are very loose.
    pfchs = qcuts_.filterCandRefs(pfChargedCands(*jet));
  }

  // CV: sort collection of PF Charged hadrons by descending Pt
  std::sort(pfchs.begin(), pfchs.end(), SortPFCandsDescendingPt());

  // Get the PF gammas
  PFCandPtrs pfGammas = qcuts_.filterCandRefs(
      pfCandidates(*jet, reco::PFCandidate::gamma));
  // Neutral hadrons
  PFCandPtrs pfnhs = qcuts_.filterCandRefs(
      pfCandidates(*jet, reco::PFCandidate::h0));

  // All the extra junk
  PFCandPtrs regionalJunk = qcuts_.filterCandRefs(regionalExtras);

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
  PFCandPtrDRFilter isoConePFGammaFilter(
      coneAxis, signalConePiZeros_(*jet), isoConePiZeros_(*jet));
  PFCandPtrDRFilter isoConePFNHFilter(
      coneAxis, signalConeNeutralHadrons_(*jet), isoConeNeutralHadrons_(*jet));
  PiZeroDRFilter isoConePiZeroFilter(
      coneAxis, signalConePiZeros_(*jet), isoConePiZeros_(*jet));

  // Additionally make predicates to select the different PF object types
  // of the regional junk objects to add to the iso cone.
  typedef xclean::PredicateAND<xclean::FilterPFCandByParticleId, PFCandPtrDRFilter> RegionalJunkConeAndIdFilter;

  xclean::FilterPFCandByParticleId pfchCandSelector(reco::PFCandidate::h);
  xclean::FilterPFCandByParticleId pfgammaCandSelector(reco::PFCandidate::gamma);
  xclean::FilterPFCandByParticleId pfnhCandSelector(reco::PFCandidate::h0);

  // Predicate to select the regional junk in the iso cone by PF id
  RegionalJunkConeAndIdFilter pfChargedJunk(
      pfchCandSelector, // select charged stuff from junk
      isoConePFCHFilter // only take those in iso cone
      );

  RegionalJunkConeAndIdFilter pfGammaJunk(
      pfgammaCandSelector, // select gammas from junk
      isoConePFGammaFilter // only take those in iso cone
      );

  RegionalJunkConeAndIdFilter pfNeutralJunk(
      pfnhCandSelector, // select neutral stuff from junk
      isoConePFNHFilter // select stuff in iso cone
      );

  // Build filter iterators select the signal charged stuff.
  PFCandPtrDRFilterIter signalPFCHCands_begin(
      signalConePFCHFilter, pfchs.begin(), pfchs.end());
  PFCandPtrDRFilterIter signalPFCHCands_end(
      signalConePFCHFilter, pfchs.end(), pfchs.end());
  PFCandPtrs signalPFCHs;
  int numSignalPFCHs = 0;
  PFCandPtrs isolationPFCHs;
  int numIsolationPFCHs = 0;
  for ( PFCandPtrDRFilterIter iter = signalPFCHCands_begin; iter != signalPFCHCands_end; ++iter ) {
    if ( numSignalPFCHs < maxSignalConeChargedHadrons_ || maxSignalConeChargedHadrons_ == -1 ) {
      //std::cout << "adding signalPFCH #" << numSignalPFCHs << ": Pt = " << (*iter)->pt() << ", eta = " << (*iter)->eta() << ", phi = " << (*iter)->phi() << std::endl;
      signalPFCHs.push_back(*iter);
      ++numSignalPFCHs;
    } else {
      //std::cout << "maxSignalConeChargedHadrons reached" 
      //	  << " --> adding isolationPFCH #" << numIsolationPFCHs << ": Pt = " << (*iter)->pt() << ", eta = " << (*iter)->eta() << ", phi = " << (*iter)->phi() << std::endl;
      isolationPFCHs.push_back(*iter);
      ++numIsolationPFCHs;
    }
  }
  PFCandPtrs::const_iterator signalPFCHs_begin = signalPFCHs.begin();
  PFCandPtrs::const_iterator signalPFCHs_end = signalPFCHs.end();

  // Cross clean pi zero content using signal cone charged hadron constituents.
  xclean::CrossCleanPiZeros<PFCandPtrDRFilterIter> piZeroXCleaner(
      signalPFCHCands_begin, signalPFCHCands_end);
  std::vector<reco::RecoTauPiZero> cleanPiZeros = piZeroXCleaner(piZeros);

  // For the rest of the constituents, we need to filter any constituents that
  // are already contained in the pizeros (i.e. electrons)
  xclean::CrossCleanPtrs<PiZeroList::const_iterator> pfCandXCleaner(cleanPiZeros.begin(), cleanPiZeros.end());

  auto isolationPFCHCands_begin(
      boost::make_filter_iterator(
        xclean::makePredicateAND(isoConePFCHFilter, pfCandXCleaner),
	pfchs.begin(), pfchs.end()));
  auto isolationPFCHCands_end(
      boost::make_filter_iterator(
	xclean::makePredicateAND(isoConePFCHFilter, pfCandXCleaner),
	pfchs.end(), pfchs.end()));
  for ( auto iter = isolationPFCHCands_begin; iter != isolationPFCHCands_end; ++iter ) {
    //std::cout << "adding isolationPFCH #" << numIsolationPFCHs << ": Pt = " << (*iter)->pt() << ", eta = " << (*iter)->eta() << ", phi = " << (*iter)->phi() << std::endl;
    isolationPFCHs.push_back(*iter);
    ++numIsolationPFCHs;
  }
  PFCandPtrs::const_iterator isolationPFCHs_begin = isolationPFCHs.begin();
  PFCandPtrs::const_iterator isolationPFCHs_end = isolationPFCHs.end();

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
		 isolationPFCHs_begin, isolationPFCHs_end);

  // Add all the stuff in the isolation cone that wasn't in the jet constituents
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kChargedHadron,
                 boost::make_filter_iterator(
                   pfChargedJunk, regionalJunk.begin(), regionalJunk.end()),
                 boost::make_filter_iterator(
                   pfChargedJunk, regionalJunk.end(), regionalJunk.end())
      );

  // Build isolation neutral hadrons
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kNeutralHadron,
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFNHFilter, pfCandXCleaner),
                   pfnhs.begin(), pfnhs.end()),
                 boost::make_filter_iterator(
                   xclean::makePredicateAND(isoConePFNHFilter, pfCandXCleaner),
                   pfnhs.end(), pfnhs.end()));

  // Add regional stuff not in jet
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kNeutralHadron,
                 boost::make_filter_iterator(
                   pfNeutralJunk, regionalJunk.begin(), regionalJunk.end()),
                 boost::make_filter_iterator(
                   pfNeutralJunk, regionalJunk.end(), regionalJunk.end())
      );

  // Build isolation PiZeros
  tau.addPiZeros(RecoTauConstructor::kIsolation,
                 PiZeroDRFilterIter(isoConePiZeroFilter, cleanPiZeros.begin(),
                                    cleanPiZeros.end()),
                 PiZeroDRFilterIter(isoConePiZeroFilter, cleanPiZeros.end(),
                                    cleanPiZeros.end()));

  // Add regional stuff not in jet
  tau.addPFCands(RecoTauConstructor::kIsolation,
                 RecoTauConstructor::kGamma,
                 boost::make_filter_iterator(
                   pfGammaJunk, regionalJunk.begin(), regionalJunk.end()),
                 boost::make_filter_iterator(
                   pfGammaJunk, regionalJunk.end(), regionalJunk.end())
      );

  // Put our built tau in the output - 'false' indicates don't build the
  // leading candidates, we already did that explicitly above.

  std::auto_ptr<reco::PFTau> tauPtr = tau.get(false);

  // Set event vertex position for tau
  reco::VertexRef primaryVertexRef = primaryVertex(*tauPtr);
  if ( primaryVertexRef.isNonnull() )
    tauPtr->setVertex(primaryVertexRef->position());

  // Set missing tau quantities
  setTauQuantities(*tauPtr,
  		   minAbsPhotonSumPt_insideSignalCone_,
  		   minRelPhotonSumPt_insideSignalCone_
		   );

  output.push_back(tauPtr);
  return output.release();
}
}}  // end namespace reco::tauk

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauBuilderPluginFactory,
                  reco::tau::RecoTauBuilderConePlugin,
                  "RecoTauBuilderConePlugin");
