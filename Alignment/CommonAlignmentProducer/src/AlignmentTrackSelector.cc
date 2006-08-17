#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentProducer/interface/AlignmentTrackSelector.h"

// constructor ----------------------------------------------------------------

AlignmentTrackSelector::AlignmentTrackSelector(const edm::ParameterSet & cfg) :
  applyBasicCuts( cfg.getParameter<bool>( "applyBasicCuts" ) ),
  applyNHighestPt( cfg.getParameter<bool>( "applyNHighestPt" ) ),
  applyMultiplicityFilter( cfg.getParameter<bool>( "applyMultiplicityFilter" ) ),
  nHighestPt( cfg.getParameter<int>( "nHighestPt" ) ),
  minMultiplicity ( cfg.getParameter<int>( "minMultiplicity" ) ),
  ptMin( cfg.getParameter<double>( "ptMin" ) ),
  ptMax( cfg.getParameter<double>( "ptMax" ) ),
  etaMin( cfg.getParameter<double>( "etaMin" ) ),
  etaMax( cfg.getParameter<double>( "etaMax" ) ),
  phiMin( cfg.getParameter<double>( "phiMin" ) ),
  phiMax( cfg.getParameter<double>( "phiMax" ) ),
  nHitMin( cfg.getParameter<double>( "nHitMin" ) ),
  nHitMax( cfg.getParameter<double>( "nHitMax" ) ),
  chi2nMax( cfg.getParameter<double>( "chi2nMax" ) )
{

  if (applyBasicCuts)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "applying basic track cuts ..."
	  << "\nptmin,ptmax:     " << ptMin   << "," << ptMax 
	  << "\netamin,etamax:   " << etaMin  << "," << etaMax
	  << "\nphimin,phimax:   " << phiMin  << "," << phiMax
	  << "\nnhitmin,nhitmax: " << nHitMin << "," << nHitMax
	  << "\nchi2nmax:        " << chi2nMax;

  if (applyNHighestPt)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "filter N tracks with highest Pt N=" << nHighestPt;

  if (applyMultiplicityFilter)
	edm::LogInfo("AlignmentTrackSelector") 
	  << "apply multiplicity filter N>=" << minMultiplicity;

}

// destructor -----------------------------------------------------------------

AlignmentTrackSelector::~AlignmentTrackSelector()
{}


// do selection ---------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::select(const Tracks& tracks, const edm::Event& evt) const 
{
  Tracks result=tracks;

  // apply basic track cuts (if selected)
  if (applyBasicCuts)  result= this->basicCuts(result);

  // filter N tracks with highest Pt (if selected)
  if (applyNHighestPt) result= this->theNHighestPtTracks(result);

  // apply minimum multiplicity requirement (if selected)
  if (applyMultiplicityFilter) {
    if (result.size()<(unsigned int)minMultiplicity) result.clear();
  }

  //edm::LogDebug("AlignmentTrackSelector") << "tracks all,kept: " << tracks.size() << "," << result.size();

  return result;

}

// make basic cuts ------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::basicCuts(const Tracks& tracks) const 
{
  Tracks result;

  for(Tracks::const_iterator it=tracks.begin();
      it!=tracks.end();it++) {
    const reco::Track* trackp=*it;
    float pt=trackp->pt();
    float eta=trackp->eta();
    float phi=trackp->phi();
    int nhit = trackp->recHitsSize(); 
    float chi2n = trackp->normalizedChi2();

    //edm::LogDebug("AlignmentTrackSelector") << " pt,eta,phi,nhit: "
    //  <<pt<<","<<eta<<","<<phi<<","<<nhit;

    if (pt>ptMin && pt<ptMax 
       && eta>etaMin && eta<etaMax 
       && phi>phiMin && phi<phiMax 
       && nhit>=nHitMin && nhit<=nHitMax
       && chi2n<chi2nMax) {
      result.push_back(trackp);
    }
  }

  return result;
}

//-----------------------------------------------------------------------------

AlignmentTrackSelector::Tracks 
AlignmentTrackSelector::theNHighestPtTracks(const Tracks& tracks) const
{
  Tracks sortedTracks=tracks;
  Tracks result;

  // sort in pt
  std::sort(sortedTracks.begin(),sortedTracks.end(),ptComparator);

  // copy theTrackMult highest pt tracks to result vector
  int n=0;
  for (Tracks::const_iterator it=sortedTracks.begin();
	   it!=sortedTracks.end(); it++) {
	if (n<nHighestPt) { result.push_back(*it); n++; }
  }

  return result;
}

