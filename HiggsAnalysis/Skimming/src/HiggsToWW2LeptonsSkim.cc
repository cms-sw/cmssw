/** \class HiggsToWW2LeptonsSkim
 *
 *  
 *  This class is an EDFilter for HWW events
 *
 *  $Date: 2007/05/31 $
 *  $Revision: 1.3 $
 *
 *  \author Ezio Torassa  -  INFN Padova
 *
 */

#include "HiggsAnalysis/Skimming/interface/HiggsToWW2LeptonsSkim.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Candidate/interface/Candidate.h"


#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;
using namespace reco;

HiggsToWW2LeptonsSkim::HiggsToWW2LeptonsSkim(const edm::ParameterSet& iConfig) :
  nEvents_(0), nAccepted_(0)
{
  trackLabel_ = iConfig.getParameter<InputTag>("TrackLabel");
  singleTrackPtMin_ = iConfig.getUntrackedParameter<double>("SingleTrackPtMin",20.);
  diTrackPtMin_ = iConfig.getUntrackedParameter<double>("DiTrackPtMin",10.);
  etaMin_ = iConfig.getUntrackedParameter<double>("etaMin",-2.4);
  etaMax_ = iConfig.getUntrackedParameter<double>("etaMax",2.4);
}


HiggsToWW2LeptonsSkim::~HiggsToWW2LeptonsSkim()
{
}

void HiggsToWW2LeptonsSkim::endJob() 
{
  edm::LogVerbatim("HiggsToWW2LeptonsSkim") 
	    << "Events read " << nEvents_ 
            << " Events accepted " << nAccepted_ 
            << "\nEfficiency " << ((double)nAccepted_)/((double)nEvents_) 
	    << std::endl;
}

// ------------ method called to skim the data  ------------
bool HiggsToWW2LeptonsSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  nEvents_++;
  bool accepted = false;
  using namespace edm;

  //  Handle<reco::TrackCollection> tracks;
  Handle<CandidateCollection> tracks;

  try
  {
    iEvent.getByLabel(trackLabel_, tracks);
  }

  catch (...) 
  {	
    edm::LogError("HiggsToWW2LeptonsSkim") << "FAILED to get Track Collection. ";
    return false;
  }

  if ( tracks->empty() ) {
    return false;
  }

  // at least one track above a pt threshold singleTrackPtMin 
  // or at least 2 tracks above a pt threshold diTrackPtMin
  int nTrackOver2ndCut = 0;
  for( size_t c = 0; c != tracks->size(); ++ c ) {
    CandidateRef cref( tracks, c );
    if ( cref->pt() > singleTrackPtMin_ && cref->eta() > etaMin_ && cref->eta() < etaMax_ ) accepted = true;
    if ( cref->pt() > diTrackPtMin_     && cref->eta() > etaMin_ && cref->eta() < etaMax_ )  nTrackOver2ndCut++;
  }
  if ( nTrackOver2ndCut >= 2 ) accepted = true;

  if ( accepted ) nAccepted_++;

  return accepted;

}
