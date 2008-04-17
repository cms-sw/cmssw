#include "PhysicsTools/PFCandProducer/interface/PFMET.h"
#include "PhysicsTools/PFCandProducer/interface/FetchCollection.h"

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

PFMET::PFMET(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<METCollection>();
  

  LogDebug("PFMET")
    <<" input collection : "<<inputTagPFCandidates_ ;
   
}



PFMET::~PFMET() { }



void PFMET::beginJob(const edm::EventSetup & es) { }


void PFMET::produce(Event& iEvent, 
			  const EventSetup& iSetup) {
  
  LogDebug("PFMET")<<"START event: "<<iEvent.id().event()
		   <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );
  
  auto_ptr< METCollection > 
    pOutput( new METCollection() ); 
  
  double sumEx = 0;
  double sumEy = 0;
  double sumEt = 0;

  for( unsigned i=0; i<pfCandidates->size(); i++ ) {

    const reco::PFCandidate& cand = (*pfCandidates)[i];
    
    double E = cand.energy();

    double phi = cand.phi();
    double cosphi = cos(phi);
    double sinphi = sin(phi);

    double theta = cand.theta();
    double sintheta = sin(theta);
    
    double et = E*sintheta;
    double ex = et*cosphi;
    double ey = et*sinphi;
    
    sumEx += ex;
    sumEy += ey;
    sumEt += et;
  }
  
  double Et = sqrt( sumEx*sumEx + sumEy*sumEy);
  XYZTLorentzVector missingEt( -sumEx, -sumEy, 0, Et);
  
  if(verbose_) {
    cout<<"PFMET: mEx, mEy, mEt = "
	<< missingEt.X() <<", "
	<< missingEt.Y() <<", "
	<< missingEt.T() <<endl;
  }

  XYZPoint vertex; // dummy vertex
  pOutput->push_back( MET(sumEt, missingEt, vertex) );

  iEvent.put( pOutput );
  
  LogDebug("PFMET")<<"STOP event: "<<iEvent.id().event()
		   <<" in run "<<iEvent.id().run()<<endl;
}


