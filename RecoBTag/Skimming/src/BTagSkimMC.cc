#include "RecoBTag/Skimming/interface/BTagSkimMC.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
using namespace reco;

BTagSkimMC::BTagSkimMC( const ParameterSet & p ) :
  nEvents_(0), nAccepted_(0)
{
  verbose = p.getUntrackedParameter<bool> ("verbose", false);
  pthatMin = p.getParameter<double> ("pthat_min");
  pthatMax = p.getParameter<double> ("pthat_max");
  process_ = p.getParameter<string> ("mcProcess");
  if (verbose) cout << " Requested:  " << process_<<endl;

}


bool BTagSkimMC::filter( Event& evt, const EventSetup& es )
{
  nEvents_++;

  Handle<int> genProcessID;
  evt.getByLabel( "genEventProcID", genProcessID );
  double processID = *genProcessID;

  Handle<double> genEventScale;
  evt.getByLabel( "genEventScale", genEventScale );
  double pthat = *genEventScale;

  if (verbose) cout << "processID: "<< processID << " - pthat: " << pthat;
  
  if  ((processID != 4) && (process_=="QCD")){  // the Pythia events (for ALPGEN see below)

    Handle<double> genFilterEff;
    evt.getByLabel( "genEventRunInfo", "FilterEfficiency", genFilterEff);
    double filter_eff = *genFilterEff;
    if (verbose) cout << " Is QCD ";
    // qcd (including min bias HS)
    if ((filter_eff == 1. || filter_eff == 0.964) && (processID == 11 || processID == 12 || processID == 13 || processID == 28 || processID == 68 || processID == 53)) {

      if (pthat > pthatMin && pthat < pthatMax) {
      if (verbose) cout << " ACCEPTED "<<endl;
	nAccepted_++;
	return true;
      }
    }


  }  // ALPGEN
  else if(processID == 4) { // this is the number for external ALPGEN events

    Handle<GenParticleCollection> genParticles;
    evt.getByLabel( "genParticles", genParticles );

    for( size_t i = 0; i < genParticles->size(); ++ i ) {
      const Candidate & p = (*genParticles)[ i ];
      int id = p.pdgId();
      int st = p.status();

      // tt+jets
      if(st == 3 && (id == 6 || id == -6) ) {
	if (verbose) cout << "We have a ttbar event"<<endl;
	nAccepted_++;
	return true;
      }
    }
  }
  if (verbose) cout << " REJECTED "<<endl;

  return false;
}

void BTagSkimMC::endJob()
{
  edm::LogVerbatim( "BTagSkimMC" ) 
    << "=============================================================================\n"
	<< " Events read: " << nEvents_
    << "\n Events accepted by BTagSkimMC: " << nAccepted_
    << "\n Efficiency: " << (double)(nAccepted_)/(double)(nEvents_)
	<< "\n==========================================================================="
    << endl;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( BTagSkimMC );
