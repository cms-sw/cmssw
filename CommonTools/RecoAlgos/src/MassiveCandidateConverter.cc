#include "CommonTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepPDT/ParticleData.hh"
#include "FWCore/Framework/interface/ESHandle.h"
#include <algorithm>
using namespace edm;
using namespace std;
using namespace converter;

MassiveCandidateConverter::MassiveCandidateConverter( const edm::ParameterSet & cfg ) :
  massSqr_(0), particle_( cfg.getParameter<PdtEntry>( "particleType" ) ) {
}

void MassiveCandidateConverter::beginFirstRun( const EventSetup & es ) {
  particle_.setup(es);
  massSqr_ = particle_.data().mass(); 
  massSqr_ *= massSqr_;
}

