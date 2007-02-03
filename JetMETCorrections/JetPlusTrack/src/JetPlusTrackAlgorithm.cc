#include "JetMETCorrections/JetPlusTrack/interface/JetPlusTrackAlgorithm.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <boost/regex.hpp>
using namespace std;
using namespace reco;


JetPlusTrackAlgorithm::~JetPlusTrackAlgorithm()
{
}

void JetPlusTrackAlgorithm::setParameters(double aCalo, double aVert, int theResp, std::vector<std::string> labels )
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

reco::CaloJet JetPlusTrackAlgorithm::applyCorrection( const reco::CaloJet& fJet) 
{
         float mScale = 1.;
         Jet::LorentzVector common (fJet.px()*mScale, fJet.py()*mScale,
                           fJet.pz()*mScale, fJet.energy()*mScale);

         reco::CaloJet theJet (common, fJet.getSpecific (), fJet.getJetConstituents());

	 cout<<" The new jet is created "<<endl;
	 		
     return theJet;
}
reco::CaloJet JetPlusTrackAlgorithm::applyCorrection( const reco::CaloJet& fJet, 
                                                      edm::Event& theEvent, 
						      const edm::EventSetup& theEventSetup) 
{

         if(fabs(fJet.eta())>2.1) return fJet;
	 
         TrackDetectorAssociator::AssociatorParameters parameters;
         parameters.useEcal = true ;
         parameters.useHcal = false ;
         parameters.useMuon = false ;
         parameters.dREcal = 0.03;
//         parameters.dRHcal = 0.07;
//        parameters.dRMuon = 0.1;
	 
//	 std::vector<GlobalPoint> AllTracks;
//         std::vector<GlobalPoint> AllTracks1;
//         cout<<" JetPlusTrackAlgorithm::The position of the primary vertex "<<theRecVertex.position()<<endl;

      double NewResponse = fJet.energy(); double echar = 0.; double echarsum = 0.;
      
      for (vector<Track>::const_iterator track = theTrack.begin();
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

      std::cout << "Details:\n" <<std::endl;
      TrackDetMatchInfo info = trackAssociator_.associate(theEvent, theEventSetup,
							  trackAssociator_.getFreeTrajectoryState(theEventSetup, *track),
							  parameters);
							  
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
         Jet::LorentzVector common (fJet.px()*mScale, fJet.py()*mScale,
                           fJet.pz()*mScale, fJet.energy()*mScale);

         reco::CaloJet theJet (common, fJet.getSpecific (), fJet.getJetConstituents());

	 cout<<" The new jet is created "<<endl;
	 		
     return theJet;
}
