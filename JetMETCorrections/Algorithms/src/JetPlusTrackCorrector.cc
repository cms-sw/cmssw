#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Algorithms/interface/SingleParticleJetResponse.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"



using namespace std;

JetPlusTrackCorrector::JetPlusTrackCorrector(const edm::ParameterSet& iConfig)
{
			  m_JetTracksAtVertex = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtVertex");
			  m_JetTracksAtCalo = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtCalo");
			  theResponseAlgo = iConfig.getParameter<int>("respalgo");
			  theSingle = new SingleParticleJetResponse;
}

JetPlusTrackCorrector::~JetPlusTrackCorrector()
{
//    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackCorrector::setParameters(double aCalo, double aVert, int theResp )
{ 
     theResponseAlgo = theResp;
}

double JetPlusTrackCorrector::correction( const LorentzVector& fJet) const 
{
         float mScale = 1.;
	 	cout<<" JetPlusTrack fake correction "<<endl;	
     return mScale;
}
double JetPlusTrackCorrector::correction(const reco::Jet& fJet,
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& theEventSetup) const 
{

// Get Jet-track association at Vertex
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtVertex;
   iEvent.getByLabel(m_JetTracksAtVertex,jetTracksAtVertex);
   const reco::JetTracksAssociation::Container jtV = *(jetTracksAtVertex.product());
   const reco::TrackRefVector trAtVertex = reco::JetTracksAssociation::getValue(jtV,fJet);

// Get Jet-track association at Calo
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtCalo;
   iEvent.getByLabel(m_JetTracksAtCalo,jetTracksAtCalo);
   const reco::JetTracksAssociation::Container jtC = *(jetTracksAtCalo.product());
   const reco::TrackRefVector trAtCalo = reco::JetTracksAssociation::getValue(jtC,fJet);
//
   double NewResponse = fJet.energy();

   if(fabs(fJet.eta())>2.1) {return NewResponse/fJet.energy();}

// Look if jet is associated with track

     if( trAtVertex.size() == 0 ) {
      return NewResponse/fJet.energy();    
     }
	 
     double echar = 0.; 
     for( reco::TrackRefVector::iterator itV = trAtVertex.begin(); itV != trAtVertex.end(); itV++)
     {
        echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);
	NewResponse = NewResponse + echar;
     }

     if( trAtCalo.size() == 0 ) {
      return NewResponse/fJet.energy();    
     }

      
     for( reco::TrackRefVector::iterator itC = trAtCalo.begin(); itC != trAtCalo.end(); itC++)
     {
       echar=sqrt((**itC).px()*(**itC).px()+(**itC).py()*(**itC).py()+(**itC).pz()*(**itC).pz()+0.14*0.14);
       double x = 0.;
       vector<double> resp=theSingle->response(echar,x,theResponseAlgo);
       NewResponse =  NewResponse - resp.front() - resp.back();
     } 
     
      
         float mScale = NewResponse/fJet.energy();
	 		
     return mScale;
}
