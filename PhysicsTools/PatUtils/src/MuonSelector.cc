#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "PhysicsTools/PatUtils/interface/MuonSelector.h"

using pat::MuonSelector;
using namespace reco;


//______________________________________________________________________________
const pat::ParticleStatus
MuonSelector::filter( const unsigned int&    index, 
                      const edm::View<Muon>& muons ) const
{

  // List of possible selections
  if      ( config_.selectionType == "none"  ) 
    {
      return GOOD;
    }
  else if ( config_.selectionType == "globalMuons" )
    {
      if ( muons[index].isGlobalMuon() ) return GOOD;
      else return BAD;
    }
  else if ( config_.selectionType == "muonPOG"  )
    {
      return muIdSelection_( index, muons );
    }
  else if ( config_.selectionType == "custom"     ) 
    {
      return customSelection_( index, muons );
    }

  // Throw! unknown configuration
  throw edm::Exception(edm::errors::Configuration) 
    << "Unknown electron ID selection " << config_.selectionType;

}

//______________________________________________________________________________
const pat::ParticleStatus
MuonSelector::customSelection_( const unsigned int&    index, 
                                const edm::View<Muon>& muons ) const
{

  // Custom muon selection from SusyAnalyzer (TQAF has a subset of these cuts)

  // Use global muon if possible
  TrackRef muontrack;
  if ( muons[index].isGlobalMuon() )
    muontrack = muons[index].track();
  else
    muontrack = muons[index].combinedMuon();

  float pt_track  = muontrack->pt();
  float dpt_track = muontrack->error(0)/muontrack->qoverp()*muontrack->pt();
  float chisq     = muontrack->normalizedChi2();
  int nHitsValid  = muontrack->numberOfValidHits();

  if ( dpt_track >= config_.dPbyPmax * pt_track ) return BAD;
  
  if ( chisq > config_.chi2max ) return BAD;

  if ( nHitsValid < config_.nHitsMin ) return BAD;

  return GOOD;

}


//______________________________________________________________________________
const pat::ParticleStatus
MuonSelector::muIdSelection_( const unsigned int&    index, 
                              const edm::View<Muon>& muons ) const
{
  // MuonID algorithm
  if ( muon::isGoodMuon((muons[index]),config_.flag) )
    {
      return BAD;
    }

  // Direct cuts on compatibility
  if (  muons[index].caloCompatibility()    <= config_.minCaloCompatibility
     || muon::segmentCompatibility(muons[index]) <= config_.minSegmentCompatibility )
    {
      return BAD;
    }

  return GOOD;
}
