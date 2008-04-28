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
  //           std::cout<<" JetPlusTrackCorrector::JetPlusTrackCorrector::constructor start "<<std::endl;
			  m_JetTracksAtVertex = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtVertex");
    //         std::cout<<" Point 1 "<<std::endl;
			  m_JetTracksAtCalo = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtCalo");
			  theResponseAlgo = iConfig.getParameter<int>("respalgo");
      //       std::cout<<" Read the first set of parameters "<<std::endl;
//  Efficiency and separation of out/in cone tracks
                          theNonEfficiencyFile = iConfig.getParameter<std::string>("NonEfficiencyFile");
                          theNonEfficiencyFileResp = iConfig.getParameter<std::string>("NonEfficiencyFileResp");
        //     std::cout<<" Read the second set of parameters "<<std::endl;
//  Out of cone tracks are added 
			  theAddOutOfConeTracks = iConfig.getParameter<bool>("AddOutOfConeTracks");
             std::cout<<" JetPlusTrackCorrector::JetPlusTrackCorrector::response algo "<<theResponseAlgo<<std::endl;			  
             std::string file1="JetMETCorrections/Configuration/data/"+theNonEfficiencyFile+".txt";
             std::string file2="JetMETCorrections/Configuration/data/"+theNonEfficiencyFileResp+".txt";

             std::cout<< " Try to open files "<<std::endl;

             edm::FileInPath f1(file1);
             edm::FileInPath f2(file2);
          //   std::cout<< " Before the set of parameters "<<std::endl;			  
			  setParameters(f1.fullPath(),f2.fullPath());
			  theSingle = new SingleParticleJetResponse();
			  
}

JetPlusTrackCorrector::~JetPlusTrackCorrector()
{
//    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackCorrector::setParameters(std::string fDataFile1,std::string fDataFile2)
{ 
  bool debug=false;
  if(debug) std::cout<<" JetPlusTrackCorrector::setParameters "<<std::endl;
  netabin = 0;
  nptbin = 0;
    
  // Read nonefficiency map
  std::ifstream in( fDataFile1.c_str() );
  string line;
  int ietaold = -1; 
  while( std::getline( in, line)){
    if(!line.size() || line[0]=='#') continue;
    istringstream linestream(line);
    double eta, pt, eff;
    int ieta, ipt;
    linestream>>ieta>>ipt>>eta>>pt>>eff;
    
    if(debug) cout <<" ieta = " << ieta <<" ipt = " << ipt <<" eta = " << eta <<" pt = " << pt <<" eff = " << eff << endl;
    if(ieta != ietaold)
      {
	etabin.push_back(eta);
	ietaold = ieta;
	netabin = ieta+1;
	if(debug) cout <<"   netabin = " << netabin <<" eta = " << eta << endl; 
      }
    
    if(ietaold == 0) 
      {
	ptbin.push_back(pt); 
	nptbin = ipt+1;
	if(debug) cout <<"   nptbin = " << nptbin <<" pt = " << pt << endl; 
      }

    trkeff.push_back(eff);
  }

  if(debug) cout <<" ====> netabin = " << netabin <<" nptbin = " << nptbin << " Efficiency vector size "<<trkeff.size()<< endl;
  // Read response suppression for the track interacted in Tracker
  std::ifstream in2( fDataFile2.c_str() );
  int ii=0;
  while( std::getline( in2, line)){
   istringstream linestream(line);
   double eta,eff;
   int ieta;
   ii++;
   linestream>>ieta>>eta>>eff;
   if(debug) cout <<" ieta = " <<ii<<" "<< ieta <<" "<<eta<<" "<<eff<<endl;
   trkeff_resp.push_back(eff);  
  }
  
  // ==========================================    
   
}

double JetPlusTrackCorrector::correction( const LorentzVector& fJet) const 
{
  throw cms::Exception("Invalid correction use") << "JetPlusTrackCorrector can be run on entire event only";
}

double JetPlusTrackCorrector::correction( const reco::Jet& fJet) const 
{
  throw cms::Exception("Invalid correction use") << "JetPlusTrackCorrector can be run on entire event only";
}

double JetPlusTrackCorrector::correction(const reco::Jet& fJet,
                                         const edm::Event& iEvent,
                                         const edm::EventSetup& theEventSetup) const 
{

   bool debug = false;

   double NewResponse = fJet.energy();

// Jet with |eta|>2.1 is not corrected

   if(fabs(fJet.eta())>2.1) {return NewResponse/fJet.energy();}

// Get Jet-track association at Vertex
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtVertex;
   iEvent.getByLabel(m_JetTracksAtVertex,jetTracksAtVertex);
   // std::cout<<" Get collection of tracks at vertex "<<m_JetTracksAtVertex<<std::endl;
   
   const reco::JetTracksAssociation::Container jtV = *(jetTracksAtVertex.product());
   //std::cout<<" Get collection of tracks at vertex :: Point 0 "<<jtV.size()<<std::endl;
     // std::cout<<" E, eta, phi "<<fJet.energy()<<" "<<fJet.eta()<<" "<<fJet.phi()<<std::endl;  

  vector<double> emean_incone;
  vector<double> netracks_incone;
  vector<double> emean_outcone;
  vector<double> netracks_outcone;

  // cout<< " netabin = "<<netabin<<" nptbin= "<<nptbin<<endl;

  for(int i=0; i < netabin; i++)
    {
      for(int j=0; j < nptbin; j++)
        {
          emean_incone.push_back(0.);
          netracks_incone.push_back(0.);
          emean_outcone.push_back(0.);
          netracks_outcone.push_back(0.);
        } // ptbin
    } // etabin

   const reco::TrackRefVector trAtVertex = reco::JetTracksAssociation::getValue(jtV,fJet);
   // std::cout<<" Get collection of tracks at vertex :: Point 1 "<<std::endl;

// Look if jet is associated with tracks. If not, return thr response of jet.

     if( trAtVertex.size() == 0 ) {
      return NewResponse/fJet.energy();
     }

// Get Jet-track association at Calo
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtCalo;
   iEvent.getByLabel(m_JetTracksAtCalo,jetTracksAtCalo);
   // std::cout<<" Get collection of tracks at Calo "<<std::endl;
   
   const reco::JetTracksAssociation::Container jtC = *(jetTracksAtCalo.product());
   const reco::TrackRefVector trAtCalo = reco::JetTracksAssociation::getValue(jtC,fJet);

// Get the collection of out-of-cone tracks
 
     reco::TrackRefVector trInCaloOutOfVertex;
     reco::TrackRefVector trInCaloInVertex;
     reco::TrackRefVector trOutOfCaloInVertex; 

     // cout<<" Number of tracks at vertex "<<trAtVertex.size()<<" Number of tracks at Calo "<<trAtCalo.size()<<endl;

     for( reco::TrackRefVector::iterator itV = trAtVertex.begin(); itV != trAtVertex.end(); itV++)
     {
       double ptV = sqrt((**itV).momentum().x()*(**itV).momentum().x()+(**itV).momentum().y()*(**itV).momentum().y());
       reco::TrackRefVector::iterator it = find(trAtCalo.begin(),trAtCalo.end(),(*itV));
       if(it != trAtCalo.end()) {
              if(debug)  cout<<" Track in cone at Calo in Vertex "<<
                ptV << " " <<(**itV).momentum().eta()<<
                " "<<(**itV).momentum().phi()<<endl;
                trInCaloInVertex.push_back(*it);
       } else
       {
       trOutOfCaloInVertex.push_back(*itV);
       if(debug) cout<<" Track out of cone at Calo in Vertex "<<ptV << " " <<(**itV).momentum().eta()<<
          " "<<(**itV).momentum().phi()<<endl;

       }
       } // in Vertex


       for ( reco::TrackRefVector::iterator itC = trAtCalo.begin(); itC != trAtCalo.end(); itC++)
       {
        if( trInCaloInVertex.size() > 0 )
        {
        reco::TrackRefVector::iterator it = find(trInCaloInVertex.begin(),trInCaloInVertex.end(),(*itC));
        if( it == trInCaloInVertex.end())
        {
          trInCaloOutOfVertex.push_back(*itC); 
        }
        } 
       }

     if(debug) {
     std::cout<<" Size of theRecoTracksOutOfConeInVertex " << trOutOfCaloInVertex.size()<<std::endl;
     std::cout<<" Size of theRecoTracksInConeInVertex " << trInCaloInVertex.size()<<std::endl;  
     std::cout<<" Size of theRecoTracksInConeOutOfVertex " << trInCaloOutOfVertex.size()<<std::endl;   
     }
//
// If track is out of cone at Vertex level but come in at Calo- subtract response but do not add
// The energy of track
//
     if(trInCaloOutOfVertex.size() > 0)
     {
        for( reco::TrackRefVector::iterator itV = trInCaloOutOfVertex.begin(); itV != trInCaloOutOfVertex.end(); itV++)
        {
          
          double echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);
          double x = 0.;
          vector<double> resp=theSingle->response(echar,x,theResponseAlgo);
          NewResponse =  NewResponse - resp.front() - resp.back();
        }   
     } 


//
// If this is the track out of cone at vertex but in cone at Calo surface - no actions are performed
//   

// Track is in cone at the Calo and in cone at the vertex

     double echar = 0.;
     if( trInCaloInVertex.size() > 0 ){ 
     for( reco::TrackRefVector::iterator itV = trInCaloInVertex.begin(); itV != trInCaloInVertex.end(); itV++)
     {
        echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);
	NewResponse = NewResponse + echar;
        double x = 0.;
        vector<double> resp=theSingle->response(echar,x,theResponseAlgo);
        NewResponse =  NewResponse - resp.front() - resp.back();

//
// Calculate the number of in cone tracks
//

      for(int i=0; i < netabin-1; i++)
        {
          for(int j=0; j < nptbin-1; j++)
            {
              if(fabs((**itV).eta())>etabin[i] && fabs((**itV).eta())<etabin[i+1])
                {
                  if(fabs((**itV).pt())>ptbin[j] && fabs((**itV).pt())<ptbin[j+1])
                    {
                      int k = i*nptbin+j;
                      netracks_incone[k]++;
                      emean_incone[k] = emean_incone[k] + echar;
                     if(debug) cout <<"        k eta/pT index = " << k
                           <<" netracks_incone[k] = " << netracks_incone[k]
                           <<" emean_incone[k] = " << emean_incone[k] << endl;
                    }
                }
            }
        }
     } // in cone, in vertex tracks

 // cout <<"         Correct for tracker inefficiency for in cone tracks" << endl;

  double corrinef = 0.;
  // for in cone tracks
  for (int etatr = 0; etatr < netabin-1; etatr++ )
    {
      for (int pttr = 0; pttr < nptbin-1; pttr++ )
        {
          int k = nptbin*etatr + pttr;
          if(netracks_incone[k]>0.) {
            emean_incone[k] = emean_incone[k]/netracks_incone[k];
            // ?? assumption. to be OK for ECAL+HCAL response
            double ee = 0.;
            vector<double> resp=theSingle->response(emean_incone[k],ee,theResponseAlgo);
            corrinef = corrinef + netracks_incone[k]*((1.-trkeff[k])/trkeff[k])*(emean_incone[k] - trkeff_resp[etatr]*(resp.front() + resp.back()));
             cout <<" k eta/pt index = " << k
                 <<" trkeff[k] = " << trkeff[k]
                 <<" netracks_incone[k] = " <<  netracks_incone[k]
                 <<" emean_incone[k] = " << emean_incone[k]
                 <<" resp = " << resp.front() + resp.back()
                 <<" resp_suppr = "<<trkeff_resp[etatr]  
                 <<" corrinef = " << corrinef << endl;
          }
        }
    }

     NewResponse = NewResponse + corrinef;

    } // theRecoTracksInConeInVertex.size() > 0

// Track is out of cone at the Calo and in cone at the vertex
     if (theAddOutOfConeTracks) {
     echar = 0.;
     double corrinef = 0.; 
     if( trOutOfCaloInVertex.size() > 0 ){
     for( reco::TrackRefVector::iterator itV = trOutOfCaloInVertex.begin(); itV != trOutOfCaloInVertex.end(); itV++)
     {
        echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);
        NewResponse = NewResponse + echar;
          for(int i=0; i < netabin-1; i++)
            {
             for(int j=0; j < nptbin-1; j++)
                {
                  if(fabs((**itV).eta())>etabin[i] && fabs((**itV).eta())<etabin[i+1])
                    {
                      if(fabs((**itV).pt())>ptbin[j] && fabs((**itV).pt())<ptbin[j+1])
                        {
                          int k = i*nptbin+j;
                          netracks_outcone[k]++;
                          emean_outcone[k] = emean_outcone[k] + echar;
                          
                          if(debug) cout <<"         k eta/pT index = " << k
                               <<" netracks_outcone[k] = " << netracks_outcone[k]
                               <<" emean_outcone[k] = " << emean_outcone[k] << endl;
                         
                        } // pt
                    } // eta
                } // pt
            } // eta
     } // out-of-cone tracks

        for (int etatr = 0; etatr < netabin-1; etatr++ )
         {
           for (int pttr = 0; pttr < nptbin-1; pttr++ )
             {
                int k = nptbin*etatr + pttr;
                if(netracks_outcone[k]>0.) {
                emean_outcone[k] = emean_outcone[k]/netracks_outcone[k];
                corrinef = corrinef + netracks_outcone[k]*((1.-trkeff[k])/trkeff[k])*emean_outcone[k];
                if(debug) cout <<" k eta/pt index = " << k
                   <<" trkeff[k] = " << trkeff[k]
                   <<" netracks_outcone[k] = " <<  netracks_outcone[k]
                   <<" emean_outcone[k] = " << emean_outcone[k]
                   <<" corrinef = " << corrinef << endl;
                }
            } // pt
         } // eta


       NewResponse = NewResponse + corrinef; 
     } // if we have out-of-cone tracks
     } // if we need out-of-cone tracks

      
     float mScale = NewResponse/fJet.energy();

     if(debug) std::cout<<" mScale= "<<mScale<<" NewRespnse "<<NewResponse<<" Jet energy "<<fJet.energy()<<std::endl;
	 		
     return mScale;
}
