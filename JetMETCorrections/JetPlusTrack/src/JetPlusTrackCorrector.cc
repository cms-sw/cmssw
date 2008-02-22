#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackCorrector.h"

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "JetMETCorrections/JetPlusTrack/interface/SingleParticleJetResponseTmp.h"




using namespace std;

JetPlusTrackCorrector::JetPlusTrackCorrector(const edm::ParameterSet& iConfig)
{
                          mInputCaloTower = iConfig.getParameter<edm::InputTag>("src2");
			  mInputPVfCTF = iConfig.getParameter<edm::InputTag>("src3");
			  
			  
			  theRcalo = iConfig.getParameter<double>("rcalo");
			  theRvert = iConfig.getParameter<double>("rvert");
			  theResponseAlgo = iConfig.getParameter<int>("respalgo");
       m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");

       edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
       parameters_.loadParameters( parameters );
                          theSingle = new SingleParticleJetResponseTmp;
			  setParameters(theRcalo,theRvert,theResponseAlgo);
       trackAssociator_ =  new TrackDetectorAssociator();
       trackAssociator_->useDefaultPropagator();
       
//    cout<<" JetPlusTrack constructor "<<endl;			  
}

JetPlusTrackCorrector::~JetPlusTrackCorrector()
{
//    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackCorrector::setParameters(double aCalo, double aVert, int theResp )
{ 
     theRcalo = aCalo;
     theRvert = aVert;
     theResponseAlgo = theResp;
        // Fill data labels for trackassociator

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
//   cout<<" JetPlusTrackCorrector::correction::starts "<<endl;
         if(fabs(fJet.eta())>2.1) return 1.;
// Get Tracker information
// Take Vertex collection   
   edm::Handle<reco::VertexCollection> primary_vertices;                 //Define Inputs (vertices)
   iEvent.getByLabel(mInputPVfCTF, primary_vertices);                  //Get Inputs    (vertices)
// Take Track Collection =================================================================
   edm::Handle<reco::TrackCollection> trackCollection;
   iEvent.getByLabel(m_inputTrackLabel,trackCollection);
   const reco::TrackCollection tC = *(trackCollection.product());
   reco::VertexCollection::const_iterator pv = primary_vertices->begin();

   if( primary_vertices->size() == 0 )
   {
// No PV correction for this event, try track collection
     if( tC.size() == 0 ) {
      return 1.;    
     }
   }	 
	 
   reco::VertexCollection::const_iterator pvmax = pv;
   
   double ptmax = -1000.;
   vector<reco::Track> theTrack;


   if( primary_vertices->size() > 0 )
   {
   for (; pv != primary_vertices->end(); pv++ )
   {
      double pto = 0.;
      
      vector<reco::Track>  tmp;
      for (reco::track_iterator track = (*pv).tracks_begin();
                track != (*pv).tracks_end(); track++)
		{
		   pto = pto + (*track)->pt();
		   tmp.push_back((**track));
		}
       if ( ptmax < pto )
       {  
           ptmax = pto;
	   pvmax = pv;
	   theTrack = tmp;
       }		
   } 
   reco::Vertex theRecVertex = *pvmax;
    
//    cout<<" Vertex with pt= "<<ptmax<<endl;
   }// Primary vertex exists
    else // No primary vetices but tracks are found
   {
     for(reco::TrackCollection::const_iterator it = tC.begin(); it != tC.end(); it++)
     {
       theTrack.push_back(*it);
     }
//     cout<<" The number of tracks included in correction "<<theTrack.size()<<endl;
   } // Track collection was taken
    
//==================================================================================      

      double NewResponse = fJet.energy(); double echar = 0.; double echarsum = 0.;
    
  
      for (vector<reco::Track>::const_iterator track = theTrack.begin();
                track != theTrack.end(); track++)
      {
             double deta = (*track).eta() - fJet.eta();
	     double dphi = fabs((*track).phi() - fJet.phi());
	     if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	     double dr = sqrt(dphi*dphi+deta*deta);
//             cout<<" Momentum of track= "<<(*track).pt() <<" "<<(*track).eta()<<endl;
//	     cout<<" Vertex level track eta,phi "<<(*track).eta()<<" "<<(*track).phi()<<endl;
//	     cout<<" Vertex level jet eta,phi "<<fJet.eta()<<" "<<fJet.phi()<<" dr "<<dr<<endl;
	     
	     if (dr > theRvert) continue;

//             cout<<" Track inside jet cone at vertex"<<endl;

//
// Add energy of charged particles
//
            echar=sqrt((*track).px()*(*track).px()+(*track).py()*(*track).py()+(*track).pz()*(*track).pz()+0.14*0.14);
            NewResponse = NewResponse + echar; 
	    echarsum = echarsum + echar;	     
//
// extrapolate track to ECAL surface
//
   
      const FreeTrajectoryState fts = trackAssociator_->getFreeTrajectoryState(theEventSetup, *track);
//      std::cout << "Details:\n" <<std::endl;
      TrackDetMatchInfo info = trackAssociator_->associate(iEvent, theEventSetup,
							  fts,
							  parameters_);
							  
//      std::cout << "ECAL, track reach ECAL: "<<info.isGoodEcal<<std::endl;
//      std::cout << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() << std::endl;
//      std::cout << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" << std::endl;
//      std::cout << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() << std::endl;
//      std::cout << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" << std::endl;
//      std::cout << "ECAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
//	<< info.trkGlobPosAtEcal.R() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
//	<< info.trkGlobPosAtEcal.phi()<< std::endl;
      
//      std::cout << "HCAL, number of crossed towers: " << info.crossedTowers.size() << std::endl;
//      std::cout << "HCAL, energy of crossed towers: " << info.hcalEnergy() << " GeV" << std::endl;
//      std::cout << "HCAL, number of towers in the cone: " << info.towers.size() << std::endl;
//      std::cout << "HCAL, energy in the cone: " << info.hcalConeEnergy() << " GeV" << std::endl;
//      std::cout << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHcal.z() << "  , "
//	<< info.trkGlobPosAtHcal.R() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
//	<< info.trkGlobPosAtHcal.phi()<< std::endl;

       if( info.isGoodEcal == 0 ) continue;

       deta = info.trkGlobPosAtEcal.eta() - fJet.eta();
       dphi = fabs( info.trkGlobPosAtEcal.phi() - fJet.phi());
       if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
       dr = sqrt(dphi*dphi+deta*deta);
 //	     cout<<" Calo level track eta,phi "<<info.trkGlobPosAtEcal.eta()<<" "<<info.trkGlobPosAtEcal.phi()<<endl;
//	     cout<<" Calo level jet eta,phi "<<fJet.eta()<<" "<<fJet.phi()<<" dr "<<dr<<endl;
      
       
       if (dr > theRcalo)
       {
               continue; 
       }

         vector<double> resp=theSingle->response(echar,info.ecalConeEnergy(),theResponseAlgo);
	 
  //    cout<<" Single particle response= "<< resp.front()<<" "<<resp.back()<<endl;

         NewResponse =  NewResponse - resp.front() - resp.back();
   
      } 
//       cout<<"Old ET"<<fJet.et()<<" eta "<<fJet.eta()<<"old energy of jet "<<fJet.energy()<<" new energy of jet  "<<NewResponse<<
//       " sum of charged energy "<<echarsum<<" correction factor "<<NewResponse/fJet.energy()<<endl;
	 
         float mScale = NewResponse/fJet.energy();
	 		
     return mScale;
}
