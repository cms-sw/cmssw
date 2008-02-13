#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/MuonIdentification/interface/IdGlobalFunctions.h"

#include "PhysicsTools/PatUtils/interface/MuonSelector.h"

using pat::MuonSelector;
using namespace reco;


//______________________________________________________________________________
MuonSelector::MuonSelector( const edm::ParameterSet& config ) :
  selectionCfg_(config),
  selectionType_( config.getParameter<std::string>("type"))
{

  if ( selectionType_ == "custom" )
    {
      dPbyPmax_ = config.getParameter<double>("dPbyPmax");
      chi2max_  = config.getParameter<double>("chi2max");
      nHitsMin_ = config.getParameter<int>("nHitsMin");
    }

}


//______________________________________________________________________________
const pat::ParticleStatus
MuonSelector::filter( const unsigned int&    index, 
                      const edm::View<Muon>& muons ) const
{

  // List of possible selections
  if      ( selectionType_ == "none"  ) 
    {
      return GOOD;
    }
  else if ( selectionType_ == "globalMuons" )
    {
      if ( muons[index].isGlobalMuon() ) return GOOD;
      return BAD;
    }
  else if ( selectionType_ == "muId"  )
    {
      throw edm::Exception(edm::errors::UnimplementedFeature)
        << "Sorry: muId-based selection is not implemented yet";
    }
  else if ( selectionType_ == "custom"     ) 
    {
      return customSelection_( index, muons );
    }

  // Throw! unknown configuration
  throw edm::Exception(edm::errors::Configuration) 
    << "Unknown electron ID selection " << selectionType_;

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

  if ( dpt_track >= dPbyPmax_ * pt_track ) return BAD;
  
  if ( chisq > chi2max_ ) return BAD;

  if ( nHitsValid < nHitsMin_ ) return BAD;

  return GOOD;

}
