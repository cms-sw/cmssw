#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackCorrector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <boost/regex.hpp>
using namespace std;

JetPlusTrackCorrector::JetPlusTrackCorrector(const edm::ParameterSet& iConfig)
{
                          mInputCaloTower = iConfig.getParameter<edm::InputTag>("src2");
			  mInputPVfCTF = iConfig.getParameter<edm::InputTag>("src3");
			  
			  m_inputTrackLabel = iConfig.getUntrackedParameter<std::string>("inputTrackLabel","ctfWithMaterialTracks");
			  std::vector<std::string> theLabels = iConfig.getParameter<std::vector<std::string> >("labels");
			  
			  theRcalo = iConfig.getParameter<double>("rcalo");
			  theRvert = iConfig.getParameter<double>("rvert");
			  theResponseAlgo = iConfig.getParameter<int>("respalgo");
			  
                          trackAssociator_.useDefaultPropagator();
                          theSingle = new SingleParticleJetResponseTmp;
			  setParameters(theRcalo,theRvert,theResponseAlgo,theLabels);
			  
}

JetPlusTrackCorrector::~JetPlusTrackCorrector()
{
}

void JetPlusTrackCorrector::setParameters(double aCalo, double aVert, int theResp, std::vector<std::string> labels )
{ 
     theRcalo = aCalo;
     theRvert = aVert;
     theResponseAlgo = theResp;
        // Fill data labels


   boost::regex regExp1 ("([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::regex regExp2 ("([^\\s,]+)[\\s,]+([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::smatch matches;
	

   for(std::vector<std::string>::const_iterator label = labels.begin(); label != labels.end(); label++) {
      if (boost::regex_match(*label,matches,regExp1))
	trackAssociator_.addDataLabels(matches[1],matches[2]);
      else if (boost::regex_match(*label,matches,regExp2))
	trackAssociator_.addDataLabels(matches[1],matches[2],matches[3]);
      else
	edm::LogError("ConfigurationError") << "Failed to parse label:\n" << *label << "Skipped.\n";
   }
}

double JetPlusTrackCorrector::correction( const LorentzVector& fJet) const 
{
         float mScale = 1.;
	 		
     return mScale;
}
double JetPlusTrackCorrector::correction( const LorentzVector& fJet, 
                                                edm::Event& iEvent, 
			                  const edm::EventSetup& theEventSetup) 
{

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
// No correction for this event
      return 1.;    
   }	 
	 
   reco::VertexCollection::const_iterator pvmax = pv;
   
   double ptmax = -1000.;
   vector<reco::Track> theTrack;
   
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
    
    cout<<" Vertex with pt= "<<ptmax<<endl;
    
//      setPrimaryVertex(const_cast(*pvmax));
       
//      setTracksFromPrimaryVertex(trPV);
//==================================================================================      
         TrackAssociator::AssociatorParameters parameters;
         parameters.useEcal = true ;
         parameters.useHcal = false ;
         parameters.useMuon = false ;
         parameters.dREcal = 0.03;
//         parameters.dRHcal = 0.07;
//        parameters.dRMuon = 0.1;
	 
//	 std::vector<GlobalPoint> AllTracks;
//         std::vector<GlobalPoint> AllTracks1;
//         cout<<" JetPlusTrackCorrector::The position of the primary vertex "<<theRecVertex.position()<<endl;

      double NewResponse = fJet.energy(); double echar = 0.; double echarsum = 0.;
      
      for (vector<reco::Track>::const_iterator track = theTrack.begin();
                track != theTrack.end(); track++)
      {
             double deta = (*track).eta() - fJet.eta();
	     double dphi = fabs((*track).phi() - fJet.phi());
	     if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	     double dr = sqrt(dphi*dphi+deta*deta);
             cout<<" Momentum of track= "<<(*track).pt() <<" "<<(*track).eta()<<endl;
	     cout<<" Vertex level track eta,phi "<<(*track).eta()<<" "<<(*track).phi()<<endl;
	     cout<<" Vertex level jet eta,phi "<<fJet.eta()<<" "<<fJet.phi()<<" dr "<<dr<<endl;
	     
	     if (dr > theRvert) continue;

             cout<<" Track inside jet cone at vertex"<<endl;

//
// Add energy of charged particles
//
            echar=sqrt((*track).px()*(*track).px()+(*track).py()*(*track).py()+(*track).pz()*(*track).pz()+0.14*0.14);
            NewResponse = NewResponse + echar; 
	    echarsum = echarsum + echar;	     
//
// extrapolate track to ECAL surface
//
   
      const FreeTrajectoryState fts = trackAssociator_.getFreeTrajectoryState(theEventSetup, *track);
      const TrackAssociator::AssociatorParameters myparameters = parameters;   
      std::cout << "Details:\n" <<std::endl;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, theEventSetup,
							  fts,
							  myparameters);
							  
      std::cout << "ECAL, track reach ECAL: "<<info.isGoodEcal<<std::endl;
      std::cout << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() << std::endl;
      std::cout << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" << std::endl;
      std::cout << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() << std::endl;
      std::cout << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" << std::endl;
      std::cout << "ECAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
	<< info.trkGlobPosAtEcal.R() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
	<< info.trkGlobPosAtEcal.phi()<< std::endl;
      
      std::cout << "HCAL, number of crossed towers: " << info.crossedTowers.size() << std::endl;
      std::cout << "HCAL, energy of crossed towers: " << info.hcalEnergy() << " GeV" << std::endl;
      std::cout << "HCAL, number of towers in the cone: " << info.towers.size() << std::endl;
      std::cout << "HCAL, energy in the cone: " << info.hcalConeEnergy() << " GeV" << std::endl;
      std::cout << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHcal.z() << "  , "
	<< info.trkGlobPosAtHcal.R() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
	<< info.trkGlobPosAtHcal.phi()<< std::endl;

       if( info.isGoodEcal == 0 ) continue;

       deta = info.trkGlobPosAtEcal.eta() - fJet.eta();
       dphi = fabs( info.trkGlobPosAtEcal.phi() - fJet.phi());
       if (dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
       dr = sqrt(dphi*dphi+deta*deta);
 	     cout<<" Calo level track eta,phi "<<info.trkGlobPosAtEcal.eta()<<" "<<info.trkGlobPosAtEcal.phi()<<endl;
	     cout<<" Calo level jet eta,phi "<<fJet.eta()<<" "<<fJet.phi()<<" dr "<<dr<<endl;
      
       
       if (dr > theRcalo)
       {
               continue; 
       }

         vector<double> resp=theSingle->response(echar,info.ecalConeEnergy(),theResponseAlgo);
	 
      cout<<" Single particle response= "<< resp.front()<<" "<<resp.back()<<endl;

         NewResponse =  NewResponse - resp.front() - resp.back();
    
      } 
       cout<<" Energy of charged= "<<echar<<" energy of jet "<<fJet.energy()<<" "<<NewResponse<<
       " "<<echarsum<<endl;
	 
         float mScale = NewResponse/fJet.energy();
	 		
     return mScale;
}
