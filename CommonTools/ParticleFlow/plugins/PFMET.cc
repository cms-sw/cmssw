#include "CommonTools/ParticleFlow/plugins/PFMET.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace math;

PFMET::PFMET(const edm::ParameterSet& iConfig) : pfMETAlgo_(iConfig) {



  inputTagPFCandidates_
    = iConfig.getParameter<InputTag>("PFCandidates");
  tokenPFCandidates_ = consumes<PFCandidateCollection>(inputTagPFCandidates_);

  produces<METCollection>();

  LogDebug("PFMET")
    <<" input collection : "<<inputTagPFCandidates_ ;

}



PFMET::~PFMET() { }



void PFMET::beginJob() { }


void PFMET::produce(Event& iEvent,
			  const EventSetup& iSetup) {

  LogDebug("PFMET")<<"START event: "<<iEvent.id().event()
		   <<" in run "<<iEvent.id().run()<<endl;



  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  iEvent.getByToken( tokenPFCandidates_, pfCandidates);

  auto_ptr< METCollection >
    pOutput( new METCollection() );



  pOutput->push_back( pfMETAlgo_.produce( *pfCandidates ) );
  iEvent.put( pOutput );

  LogDebug("PFMET")<<"STOP event: "<<iEvent.id().event()
		   <<" in run "<<iEvent.id().run()<<endl;
}


