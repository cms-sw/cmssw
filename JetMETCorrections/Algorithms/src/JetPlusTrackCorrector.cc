#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
//
#include <vector>

//#include "FWCore/Framework/interface/Event.h"
//#include "DataFormats/Common/interface/Handle.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "JetMETCorrections/Algorithms/interface/SingleParticleJetResponse.h"
//#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
//#include "DataFormats/MuonReco/interface/Muon.h"
//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/MuonReco/interface/MuonFwd.h"
//

using namespace std;

JetPlusTrackCorrector::JetPlusTrackCorrector(const edm::ParameterSet& iConfig)
{
  m_JetTracksAtVertex = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtVertex");
  m_JetTracksAtCalo = iConfig.getParameter<edm::InputTag>("JetTrackCollectionAtCalo");
  m_muonsSrc = iConfig.getParameter<edm::InputTag>("muonSrc");
  theResponseAlgo = iConfig.getParameter<int>("respalgo");
  theAddOutOfConeTracks = iConfig.getParameter<bool>("AddOutOfConeTracks");
  theNonEfficiencyFile = iConfig.getParameter<std::string>("NonEfficiencyFile");
  theNonEfficiencyFileResp = iConfig.getParameter<std::string>("NonEfficiencyFileResp");
  theResponseFile = iConfig.getParameter<std::string>("ResponseFile"); 			  
  theAddOutOfConeTracks = iConfig.getParameter<bool>("AddOutOfConeTracks");
  theUseQuality = iConfig.getParameter<bool>("UseQuality");
  theTrackQuality = iConfig.getParameter<std::string>("TrackQuality");

  trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

  std::cout<<" BugFix JetPlusTrackCorrector::JetPlusTrackCorrector::response algo "<< theResponseAlgo
	   <<" TheAddOutOfConeTracks " << theAddOutOfConeTracks
	   <<" TheNonEfficiencyFile "<< theNonEfficiencyFile
	   <<" TheNonEfficiencyFileResp "<< theNonEfficiencyFileResp
	   <<" TheResponseFile "<< theResponseFile <<std::endl;
  
             std::string file1="JetMETCorrections/Configuration/data/"+theNonEfficiencyFile+".txt";
             std::string file2="JetMETCorrections/Configuration/data/"+theNonEfficiencyFileResp+".txt";
	     std::string file3="JetMETCorrections/Configuration/data/"+theResponseFile+".txt";

             std::cout<< " Try to open files "<<std::endl;

             edm::FileInPath f1(file1);
             edm::FileInPath f2(file2);
	     edm::FileInPath f3(file3);
          //   std::cout<< " Before the set of parameters "<<std::endl;			  
	     setParameters(f1.fullPath(),f2.fullPath(),f3.fullPath());
	     theSingle = new SingleParticleJetResponse();
			  
}

JetPlusTrackCorrector::~JetPlusTrackCorrector()
{
//    cout<<" JetPlusTrack destructor "<<endl;
}

void JetPlusTrackCorrector::setParameters(std::string fDataFile1,std::string fDataFile2, std::string fDataFile3)
{ 
  //bool debug = true;
  bool debug = false;
  
  if(debug) std::cout<<" JetPlusTrackCorrector::setParameters "<<std::endl;
  // Read efficiency map
  netabin1 = 0;
  nptbin1  = 0;
  if(debug) std::cout <<" Read efficiency map " << std::endl;
  if(debug) std::cout <<" =================== " << std::endl;
  std::ifstream in1( fDataFile1.c_str() );
  string line1;
  int ietaold = -1; 
  while( std::getline( in1, line1)){
    if(!line1.size() || line1[0]=='#') continue;
    istringstream linestream(line1);
    double eta, pt, eff;
    int ieta, ipt;
    linestream>>ieta>>ipt>>eta>>pt>>eff;
    
    if(debug) std::cout <<" ieta = " << ieta <<" ipt = " << ipt <<" eta = " << eta <<" pt = " << pt <<" eff = " << eff << std::endl;
    if(ieta != ietaold)
      {
	etabin1.push_back(eta);
	ietaold = ieta;
	netabin1 = ieta+1;
	if(debug) std::cout <<"   netabin1 = " << netabin1 <<" eta = " << eta << std::endl; 
      }
    
    if(ietaold == 0) 
      {
	ptbin1.push_back(pt); 
	nptbin1 = ipt+1;
	if(debug) std::cout <<"   nptbin1 = " << nptbin1 <<" pt = " << pt << std::endl; 
      }

    trkeff.push_back(eff);
  }
  if(debug) std::cout <<" ====> netabin1 = " << netabin1 <<" nptbin1 = " << nptbin1 << std::endl;
  // ==========================================    

  // Read leakage map
  netabin2 = 0;
  nptbin2  = 0;
  if(debug) std::cout <<" Read leakage map " << std::endl;
  if(debug) std::cout <<" ================ " << std::endl;
  std::ifstream in2( fDataFile2.c_str() );
  string line2;
  ietaold = -1; 
  while( std::getline( in2, line2)){
    if(!line2.size() || line2[0]=='#') continue;
    istringstream linestream(line2);
    double eta, pt, eleak;
    int ieta, ipt;
    linestream>>ieta>>ipt>>eta>>pt>>eleak;
    
    if(debug) std::cout <<" ieta = " << ieta <<" ipt = " << ipt <<" eta = " << eta <<" pt = " << pt <<" eff = " << eleak << std::endl;
    if(ieta != ietaold)
      {
	etabin2.push_back(eta);
	ietaold = ieta;
	netabin2 = ieta+1;
	if(debug) std::cout <<"   netabin2 = " << netabin2 <<" eta = " << eta << std::endl; 
      }
    
    if(ietaold == 0) 
      {
	ptbin2.push_back(pt); 
	nptbin2 = ipt+1;
	if(debug) std::cout <<"   nptbin2 = " << nptbin2 <<" pt = " << pt << std::endl; 
      }

    eleakage.push_back(eleak);
  }
  if(debug) std::cout <<" ====> netabin2 = " << netabin1 <<" nptbin2 = " << nptbin2 << std::endl;
  // ==========================================    
  // Read efficiency map
  netabin3 = 0;
  nptbin3  = 0;
  if(debug) std::cout <<" Read response map " << std::endl;
  if(debug) std::cout <<" =================== " << std::endl;
  std::ifstream in3( fDataFile3.c_str() );
  string line3;
  ietaold = -1; 
  while( std::getline( in3, line3)){
    if(!line3.size() || line3[0]=='#') continue;
    istringstream linestream(line3);
    double eta, pt, resp;
    int ieta, ipt;
    linestream>>ieta>>ipt>>eta>>pt>>resp;
    
    if(debug) std::cout <<" ieta = " << ieta <<" ipt = " << ipt <<" eta = " << eta <<" pt = " << pt <<" resp = " << resp << std::endl;
    if(ieta != ietaold)
      {
	etabin3.push_back(eta);
	ietaold = ieta;
	netabin3 = ieta+1;
	if(debug) std::cout <<"   netabin3 = " << netabin3 <<" eta = " << eta << std::endl; 
      }
    
    if(ietaold == 0) 
      {
	ptbin3.push_back(pt); 
	nptbin3 = ipt+1;
	if(debug) std::cout <<"   nptbin3 = " << nptbin3 <<" pt = " << pt << std::endl; 
      }

    response.push_back(resp);
  }
  if(debug) std::cout <<" ====> netabin3 = " << netabin3 <<" nptbin3 = " << nptbin3 << std::endl;
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
   //bool debug = true;

   double NewResponse = fJet.energy();

// Jet with |eta|>2.1 is not corrected

   if(fabs(fJet.eta())>2.1) {return NewResponse/fJet.energy();}

   // Get muons
   edm::Handle<reco::MuonCollection> muons;
   reco::MuonCollection::const_iterator muon;
   iEvent.getByLabel(m_muonsSrc, muons);
   
   if(debug){
   cout <<" muon collection size = " << muons->size() << endl;   

   if(muons->size() != 0) {
     for (muon = muons->begin(); muon != muons->end(); ++muon) {
       //reco::TrackRef glbTrack = muon->combinedMuon();
       cout <<" muon pT = " << muon->pt() 
	    <<" muon eta = " << muon->eta()
	    <<" muon phi = " << muon->phi() << endl;  
     }
   }
   }

   // Get Jet-track association at Vertex
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtVertex;
   iEvent.getByLabel(m_JetTracksAtVertex,jetTracksAtVertex);
   
//
// No tracks at Vertex - no Corrections
// 
  
   if(!jetTracksAtVertex.isValid()) {return NewResponse/fJet.energy();}
    
   // std::cout<<" Get collection of tracks at vertex "<<m_JetTracksAtVertex<<std::endl;
   
   const reco::JetTracksAssociation::Container jtV = *(jetTracksAtVertex.product());
   
  //std::cout<<" Get collection of tracks at vertex :: Point 0 "<<jtV.size()<<std::endl;
  // std::cout<<" E, eta, phi "<<fJet.energy()<<" "<<fJet.eta()<<" "<<fJet.phi()<<std::endl;  

  vector<double> emean_incone;
  vector<double> netracks_incone;
  vector<double> emean_outcone;
  vector<double> netracks_outcone;

  // cout<< " netabin = "<<netabin<<" nptbin= "<<nptbin<<endl;

  for(int i=0; i < netabin1; i++)
    {
      for(int j=0; j < nptbin1; j++)
        {
          emean_incone.push_back(0.);
          netracks_incone.push_back(0.);
          emean_outcone.push_back(0.);
          netracks_outcone.push_back(0.);
        } // ptbin
    } // etabin

   const reco::TrackRefVector trAtVertex = reco::JetTracksAssociation::getValue(jtV,fJet);

// std::cout<<" Get collection of tracks at vertex :: Point 1 "<<std::endl;
// Look if jet is associated with tracks. If not, return the response of jet.

   if( trAtVertex.size() == 0 ) {return NewResponse/fJet.energy();}

// Get Jet-track association at Calo
   edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtCalo;
   iEvent.getByLabel(m_JetTracksAtCalo,jetTracksAtCalo);
   
   // std::cout<<" Get collection of tracks at Calo "<<std::endl;
   
   reco::TrackRefVector trAtCalo;
   
   if(jetTracksAtCalo.isValid()) { 
     const reco::JetTracksAssociation::Container jtC = *(jetTracksAtCalo.product());
     trAtCalo = reco::JetTracksAssociation::getValue(jtC,fJet);
   }

//   const reco::TrackRefVector trAtCalo = reco::JetTracksAssociation::getValue(jtC,fJet);


   // tracks in vertex cone and in calo cone 
   reco::TrackRefVector trInCaloInVertex;
   // tracks in calo cone, but out of vertex cone 
   reco::TrackRefVector trInCaloOutOfVertex;
   // tracks in vertex cone but out of calo cone
   reco::TrackRefVector trOutOfCaloInVertex; 

   // muon in vertex cone and in calo cone 
   reco::TrackRefVector muInCaloInVertex;
   // muon in calo cone, but out of vertex cone 
   reco::TrackRefVector muInCaloOutOfVertex;
   // muon in vertex cone but out of calo cone
   reco::TrackRefVector muOutOfCaloInVertex; 


   // cout<<" Number of tracks at vertex "<<trAtVertex.size()<<" Number of tracks at Calo "<<trAtCalo.size()<<endl;

   for( reco::TrackRefVector::iterator itV = trAtVertex.begin(); itV != trAtVertex.end(); itV++)
     {
       if(theUseQuality && (!(**itV).quality(trackQuality_))) continue;
       double ptV = (*itV)->pt();
       // check if it is muon
       bool ismuon(false);
       for (muon = muons->begin(); muon != muons->end(); ++muon) {
	 // muon id here
	 // track quality requirements are general and should be done elsewhere
	 if (! muon->isGood(reco::Muon::TMLastStationTight) || 
	     muon->innerTrack()->pt()<3.0 ) continue;
	 if (itV->id() != muon->innerTrack().id())
	     throw cms::Exception("FatalError") 
	       << "Product id of the tracks associated to the jet " << itV->id() 
	       <<" is different from the product id of the inner track used for muons " << muon->innerTrack().id()
	       << "\nCannot compare tracks from different collection. Configuration Error\n";
	   if (*itV == muon->innerTrack()) {
	     ismuon = true;
	     break;
	   }
       }

       reco::TrackRefVector::iterator it = find(trAtCalo.begin(),trAtCalo.end(),(*itV));
       if(it != trAtCalo.end()) {
	 if(debug)  cout<<" Track in cone at Calo in Vertex "<<
		      ptV << " " <<(**itV).momentum().eta()<<
		      " "<<(**itV).momentum().phi()<<endl;
	 if (ismuon) 
	   muInCaloInVertex.push_back(*it);
	 else 
	   trInCaloInVertex.push_back(*it);
       } else { // trAtCalo.end()
	 if (ismuon)
	   muOutOfCaloInVertex.push_back(*itV);
	 else 
	   trOutOfCaloInVertex.push_back(*itV);
	 
	 if(debug) cout<<" Track out of cone at Calo in Vertex "<<ptV << " " <<(**itV).momentum().eta()<<
		     " "<<(**itV).momentum().phi()<<endl;
       } // trAtCalo.end()
     } // in Vertex
   
   
   for ( reco::TrackRefVector::iterator itC = trAtCalo.begin(); itC != trAtCalo.end(); itC++)
     {
       if(theUseQuality && (!(**itC).quality(trackQuality_))) continue;
       if( trInCaloInVertex.size() > 0 )
	 {

	   reco::TrackRefVector::iterator it = find(trInCaloInVertex.begin(),trInCaloInVertex.end(),(*itC));
	   reco::TrackRefVector::iterator im = find(muInCaloInVertex.begin(),muInCaloInVertex.end(),(*itC));

	   if( it == trInCaloInVertex.end() && im == muInCaloInVertex.end())
	     {
	       // check if it is muon
	       bool ismuon(false);
	       for (muon = muons->begin(); muon != muons->end(); ++muon) {
		 // muon id here
		 // track quality requirements are general and should be done elsewhere
		 if (! muon->isGood(reco::Muon::TMLastStationTight) || 
		     muon->innerTrack()->pt()<3.0 ) continue;
		 if (itC->id() != muon->innerTrack().id())
		   throw cms::Exception("FatalError") 
		     << "Product id of the tracks associated to the jet " << itC->id() 
		     <<" is different from the product id of the inner track used for muons " << muon->innerTrack().id()
		     << "\nCannot compare tracks from different collection. Configuration Error\n";
		 if (*itC == muon->innerTrack()) {
		   ismuon = true;
		   break;
		 }
	       }
	     
	       if (ismuon)
		 muInCaloOutOfVertex.push_back(*itC);
	       else 
		 trInCaloOutOfVertex.push_back(*itC);
	     }
	 }
     } 
   
   if(debug) {
     std::cout<<" Size of theRecoTracksInConeInVertex " << trInCaloInVertex.size()<<std::endl;  
     std::cout<<" Size of theRecoTracksOutOfConeInVertex " << trOutOfCaloInVertex.size()<<std::endl;
     std::cout<<" Size of theRecoTracksInConeOutOfVertex " << trInCaloOutOfVertex.size()<<std::endl; 
     std::cout<<" Size of theRecoMuonTracksInConeInVertex " << muInCaloInVertex.size()<<std::endl;  
     std::cout<<" Size of theRecoMuonTracksOutOfConeInVertex " << muOutOfCaloInVertex.size()<<std::endl;
     std::cout<<" Size of theRecoMuonTracksInConeOutOfVertex " << muInCaloOutOfVertex.size()<<std::endl; 
     std::cout <<" muon collection size = " << muons->size() << std::endl;  
   }
   //
   // If track is out of cone at Vertex level but come in at Calo cone subtract response but do not add
   // The energy of track
   //
   if(trInCaloOutOfVertex.size() > 0)
     {
       for( reco::TrackRefVector::iterator itV = trInCaloOutOfVertex.begin(); itV != trInCaloOutOfVertex.end(); itV++)
	 {
	   double echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);
	   /*
	     double x = 0.;
	     vector<double> resp=theSingle->response(echar,x,theResponseAlgo);
	     NewResponse =  NewResponse - resp.front() - resp.back();
	   */
	   for(int i=0; i < netabin1-1; i++)
	     {
	       for(int j=0; j < nptbin1-1; j++)
		 {
		   if(fabs((**itV).eta())>etabin1[i] && fabs((**itV).eta())<etabin1[i+1])
		     {
		       if(fabs((**itV).pt())>ptbin1[j] && fabs((**itV).pt())<ptbin1[j+1])
			 {
			   int k = i*nptbin1+j;
			   NewResponse =  NewResponse - response[k]*echar;
			   if(debug) cout <<"        k eta/pT index = " << k
					  <<" netracks_incone[k] = " << NewResponse << endl;
			 }
		     }
		 }
	     }   
	 }
     }

   // Track is in cone at the Calo and in cone at the vertex
   
   double echar = 0.;
   if( trInCaloInVertex.size() > 0 ){ 
     for( reco::TrackRefVector::iterator itV = trInCaloInVertex.begin(); itV != trInCaloInVertex.end(); itV++)
       {
	 
// Temporary solution>>>>>> Remove tracks with pt>50 GeV
         if( (**itV).pt() >= 50. ) continue;
// >>>>>>>>>>>>>>

	 echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.14*0.14);

	 NewResponse = NewResponse + echar;
	 
         if(debug) cout<<" New response in Calo in Vertex sum "<<NewResponse<<" "<<echar<<endl;

	 /*
	 double x = 0.;
	 vector<double> resp=theSingle->response(echar,x,theResponseAlgo);
	 NewResponse =  NewResponse - resp.front() - resp.back();
	 */
	 //
	 // Calculate the number of in cone tracks and response subtruction
	 //
	 
	 for(int i=0; i < netabin1-1; i++)
	   {
	     for(int j=0; j < nptbin1-1; j++)
	       {
		 if(fabs((**itV).eta())>etabin1[i] && fabs((**itV).eta())<etabin1[i+1])
		   {
		     if(fabs((**itV).pt())>ptbin1[j] && fabs((**itV).pt())<ptbin1[j+1])
		       {
			 int k = i*nptbin1+j;
			 netracks_incone[k]++;
			 emean_incone[k] = emean_incone[k] + echar;
			 
                         if( debug ) cout<<" Before subtraction "<<NewResponse<<" echar "<<echar<<" "<<
                         response[k]*echar<<endl;
			 
			 
			 NewResponse =  NewResponse - response[k]*echar;
			 if(debug) cout <<"        k eta/pT index = " << k
					<<" netracks_incone[k] = " << netracks_incone[k]
					<<" emean_incone[k] = " << emean_incone[k]
                                        <<" i,j "<<i<<" "<<j<<" "<<response[k] 
                                        <<" echar "<<echar<<" "<<response[k]*echar
                                        <<" New Response "<< NewResponse 
					<< endl;
		       }
		   }
	       }
	   }
       } // in cone, in vertex tracks
     
     // cout <<"         Correct for tracker inefficiency for in cone tracks" << endl;
     
     double corrinef = 0.;
     // for in cone tracks
     for (int etatr = 0; etatr < netabin1-1; etatr++ )
       {
	 for (int pttr = 0; pttr < nptbin1-1; pttr++ )
	   {
	     int k = nptbin1*etatr + pttr;
	     if(netracks_incone[k]>0.) {
	       emean_incone[k] = emean_incone[k]/netracks_incone[k];
	       /*
	       double ee = 0.;
	       vector<double> resp=theSingle->response(emean_incone[k],ee,theResponseAlgo);
	       corrinef = corrinef + netracks_incone[k]*((1.-trkeff[k])/trkeff[k])*(emean_incone[k] - eleakage[etatr]*(resp.front() + resp.back()));
	       */
	       corrinef = corrinef + netracks_incone[k] * ((1.-trkeff[k])/trkeff[k]) * emean_incone[k] * (1.-eleakage[k]*response[k]);
	       if(debug) cout <<" k eta/pt index = " << k
			      <<" trkeff[k] = " << trkeff[k]
			      <<" netracks_incone[k] = " <<  netracks_incone[k]
			      <<" emean_incone[k] = " << emean_incone[k]
			      <<" resp = " << response[k]
			      <<" resp_suppr = "<<eleakage[etatr]  
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
	   for(int i=0; i < netabin1-1; i++)
	     {
	       for(int j=0; j < nptbin1-1; j++)
		 {
		   if(fabs((**itV).eta())>etabin1[i] && fabs((**itV).eta())<etabin1[i+1])
		     {
		       if(fabs((**itV).pt())>ptbin1[j] && fabs((**itV).pt())<ptbin1[j+1])
			 {
			   int k = i*nptbin1+j;
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
       
       // correct of inefficiency of out of cone tracks
       for (int etatr = 0; etatr < netabin1-1; etatr++ )
         {
           for (int pttr = 0; pttr < nptbin1-1; pttr++ )
             {
	       int k = nptbin1*etatr + pttr;
	       if(netracks_outcone[k]>0.) {
		 emean_outcone[k] = emean_outcone[k]/netracks_outcone[k];
		 corrinef = corrinef + netracks_outcone[k] * ((1.-trkeff[k])/trkeff[k]) * emean_outcone[k];
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
   
   // muon treatment:
   //    for in vertex , in calo cone muons add muon p and subtract 2 GeV
     for( reco::TrackRefVector::iterator itV = muInCaloInVertex.begin(); itV != muInCaloInVertex.end(); itV++)
       {
	 echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.105*0.105);
	 NewResponse = NewResponse + echar - 2.0;
       }
   //    for in vertex, out of calo cone muons add muon pt
     if( trOutOfCaloInVertex.size() > 0 ){
       for( reco::TrackRefVector::iterator itV = muOutOfCaloInVertex.begin(); itV != muOutOfCaloInVertex.end(); itV++)
	 {
	   echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.105*0.105);
	   NewResponse = NewResponse + echar;
	 }
     }
   //    for out of vertex, in cone muons suntract 2 GeV  
       for( reco::TrackRefVector::iterator itV = muInCaloOutOfVertex.begin(); itV != muInCaloOutOfVertex.end(); itV++)
	 {
	   //double echar=sqrt((**itV).px()*(**itV).px()+(**itV).py()*(**itV).py()+(**itV).pz()*(**itV).pz()+0.105*0.105);
	   NewResponse = NewResponse - 2.0;
	 }
   float mScale = NewResponse/fJet.energy();
   
   if(debug) std::cout<<" mScale= "<<mScale<<" NewResponse "<<NewResponse<<" Jet energy "<<fJet.energy()<<std::endl;
   
   return mScale;
}
