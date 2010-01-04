//
// Original Author:  Olga Kodolova, September 2007
//
// ZSP Jet Corrector
//
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrectorBG.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;

JetPlusTrackCorrectorBG::JetPlusTrackCorrectorBG (const edm::ParameterSet& fConfig) {
    mJets = fConfig.getParameter<edm::InputTag> ("jets");
    mTracks = fConfig.getParameter<edm::InputTag> ("tracks");
    mConeSize = fConfig.getParameter<double> ("coneSize");
    theUseQuality = fConfig.getParameter<bool>("UseTrackQuality");
    theTrackQuality = fConfig.getParameter<std::string>("TrackQuality");
    theNonEfficiencyFile = fConfig.getParameter<std::string>("EfficiencyMap");
    theNonEfficiencyFileResp = fConfig.getParameter<std::string>("LeakageMap");
    theResponseFile = fConfig.getParameter<std::string>("ResponseMap");           

    trackQuality_=reco::TrackBase::qualityByName(theTrackQuality);

    std::string file1=theNonEfficiencyFile+".txt";
    std::string file2=theNonEfficiencyFileResp+".txt";
    std::string file3=theResponseFile+".txt";



    std::cout<< " Try to open files "<<std::endl;

    edm::FileInPath f1(file1);
    edm::FileInPath f2(file2);
    edm::FileInPath f3(file3);
          //   std::cout<< " Before the set of parameters "<<std::endl;                   
    setParameters(f1.fullPath(),f2.fullPath(),f3.fullPath());
    theSingle = new SingleParticleJetResponse();

}


JetPlusTrackCorrectorBG::~JetPlusTrackCorrectorBG () {
} 




void JetPlusTrackCorrectorBG::setParameters(std::string fDataFile1,std::string fDataFile2, std::string fDataFile3)
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

double JetPlusTrackCorrectorBG::correction( const reco::Jet& jet ) const {
  edm::LogError("JetPlusTrackCorrectorBG")
    << "JetPlusTrackCorrectorBG can be run on entire event only";
  return 1.;
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrectorBG::correction( const reco::Particle::LorentzVector& jet ) const {
  edm::LogError("JetPlusTrackCorrectorBG")
    << "JetPlusTrackCorrectorBG can be run on entire event only";
  return 1.;
}





double JetPlusTrackCorrectorBG::correction( const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& iSetup) const
{

  double mScale = 1.;
  double NewResponse = fJet.energy();
  bool debug=false;

//  typedef std::vector<reco::JetBaseRef>::iterator JetBaseRefIterator;
//  const reco::JetTracksAssociation::Container jtV = *( jetTracksAtVertex.product() );
//  std::vector<reco::JetBaseRef> fJets = reco::JetTracksAssociation::allJets(jtV);

  edm::Handle <edm::View <reco::Jet> > jets_h;
  fEvent.getByLabel (mJets, jets_h);
  edm::Handle <reco::TrackCollection> tracks_h;
  fEvent.getByLabel (mTracks, tracks_h);

  std::vector <edm::RefToBase<reco::Jet> > fJets;
  fJets.reserve (jets_h->size());
  for (unsigned i = 0; i < jets_h->size(); ++i) fJets.push_back (jets_h->refAt(i));


  std::vector <reco::TrackRef> fTracks;
  fTracks.reserve (tracks_h->size());
  for (unsigned i = 0; i < tracks_h->size(); ++i) {
             fTracks.push_back (reco::TrackRef (tracks_h, i));
  } 


  double myjetEta = fJet.eta();
  reco::TrackRefVector trBgOutOfVertex;

  // loop on tracks
  for (unsigned t = 0; t < fTracks.size(); ++t) {

     int track_bg = 0;

     if(theUseQuality && (!(*fTracks[t]).quality(trackQuality_)))
     {
       cout<<"BG, BAD trackQuality, ptBgV="<<fTracks[t]->pt()<<" etaBgV = "<<fTracks[t]->eta()<<" phiBgV = "<<fTracks[t]->phi()<<endl;
       continue;
     }

    const reco::Track* track = &*(fTracks[t]);
    double trackEta = track->eta();
    double trackPhi = track->phi();

    std::cout<<"++++++++++++++++>  track="<<t<<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
             <<" coneSize="<<mConeSize<<std::endl;
   
   //loop on jets
    for (unsigned j = 0; j < fJets.size(); ++j) {

     const reco::Jet* jet = &*(fJets[j]);
     double jetEta = jet->eta();
     double jetPhi = jet->phi();

      std::cout<<"-jet="<<j<<" jetEt ="<<jet->pt()
      <<" jetE="<<jet->energy()<<" jetEta="<<jetEta<<" jetPhi="<<jetPhi<<std::endl;

      if(fabs(jetEta - trackEta) < mConeSize) {
       double dphiTrackJet = fabs(trackPhi - jetPhi);
       if(dphiTrackJet > M_PI) dphiTrackJet = 2*M_PI - dphiTrackJet;

//        std::cout<<"     detaJetTrack="<<fabs(jetEta - trackEta)<<" dphiTrackJet="<<dphiTrackJet
//                 <<" trackPhi="<<trackPhi<<" jetPhi="<<jetPhi<<std::endl;

       if(dphiTrackJet < mConeSize) 
        {
         track_bg = 1;
         
         std::cout<<"===>>>> Track inside jet at vertex, track_bg="<< track_bg <<" track="<<t<<" jet="<<j
                 <<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
                 <<" jetEta="<<jetEta<<" jetPhi="<<jetPhi<<std::endl;
        }
      }
//=>
//       if(track_bg == 1) {
//        std::cout<<" Track inside jet at vertex, track_bg="<< track_bg <<" track="<<t<<" jet="<<j
//                 <<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
//                 <<" jetEta="<<jetEta<<" jetPhi="<<jetPhi<<std::endl;
//      }
//=>
      
    } //jets

    if( track_bg == 1 ) continue;
// Check if track is within eta ring

    if(fabs(myjetEta - trackEta) > mConeSize ) continue;

       trBgOutOfVertex.push_back (fTracks[t]);
        std::cout<<"------Track outside jet at vertex, track_bg="<< track_bg<<" track="<<t
                 <<" trackEta="<<trackEta<<" trackPhi="<<trackPhi
                 <<std::endl;    

  } //tracks    

     std::cout<<"+++> JetBackgroundTracksDRVertex::Ntracks in region eta_ring: backgroundTracksDETA.size()=  "
              << trBgOutOfVertex.size()<<std::endl;

   double echarBg = 0.;
   double respBg  = 0.;
   double EnergyOfBackgroundCharged   = 0.;
   double ResponseOfBackgroundCharged = 0.;


   if( trBgOutOfVertex.size() == 0 ) return 1.; 


     for( reco::TrackRefVector::iterator iBgtV = trBgOutOfVertex.begin(); iBgtV != trBgOutOfVertex.end(); iBgtV++)
       {

// Temporary solution>>>>>> Remove tracks with pt>50 GeV
         if( (**iBgtV).pt() >= 50. ) continue;
// >>>>>>>>>>>>>>


           echarBg=sqrt((**iBgtV).px()*(**iBgtV).px()+(**iBgtV).py()*(**iBgtV).py()+(**iBgtV).pz()*(**iBgtV).pz()+0.14*0.14);

// calculate response of background charged outside all found jets at vertex
// in region eta_ring=2*mCone at calorimeter

           for(int i=0; i < netabin1-1; i++)
             {
               for(int j=0; j < nptbin1-1; j++)
                 {
                   if(fabs((**iBgtV).eta())>etabin1[i] && fabs((**iBgtV).eta())<etabin1[i+1])
                     {
                       if(fabs((**iBgtV).pt())>ptbin1[j] && fabs((**iBgtV).pt())<ptbin1[j+1])
                         {
                           int k = i*nptbin1+j;
                           respBg =  response[k]*echarBg;
//                           cout <<"        k eta/pT index = " << k
//                                          <<" netracks_incone[k] = " <<NewResponse << endl;
                         }// ptbin1
                     } // etabin1
                 } // cycle j
             } // cycle i

              cout<<"===BG TRACKS, echarBg ="<<echarBg<<" respBg  ="<<respBg<<endl;

      EnergyOfBackgroundCharged = EnergyOfBackgroundCharged + echarBg;
      ResponseOfBackgroundCharged = ResponseOfBackgroundCharged + respBg;

       } // BG tracks

      cout<<"+++BG TRACKS, EnergyOfBackgroundCharged="<<EnergyOfBackgroundCharged
          <<" ResponseOfBackgroundCharged="<<ResponseOfBackgroundCharged<<endl;

    double SquareEtaRingWithoutJets = 4*(M_PI*mConeSize - mConeSize*mConeSize);

    EnergyOfBackgroundCharged = EnergyOfBackgroundCharged/SquareEtaRingWithoutJets;
    ResponseOfBackgroundCharged = ResponseOfBackgroundCharged/SquareEtaRingWithoutJets;

      cout<<"+++BG TRACKS, DENSITY: EnergyOfBackgroundCharged/5.28="<<EnergyOfBackgroundCharged
          <<" ResponseOfBackgroundCharged/5.28="<<ResponseOfBackgroundCharged<<endl;

// calculate the mean charged energy and response in jet area (M_PI*mCone*mCone)

//      EnergyOfBackgroundCharged   = 0.785*EnergyOfBackgroundCharged;
//      ResponseOfBackgroundCharged = 0.785*ResponseOfBackgroundCharged;

      EnergyOfBackgroundCharged   = M_PI*mConeSize*mConeSize*EnergyOfBackgroundCharged;
      ResponseOfBackgroundCharged = M_PI*mConeSize*mConeSize*ResponseOfBackgroundCharged;

      cout<<"===BG TRACKS, MEAN: 0.785*EnergyOfBackgroundCharged="<<EnergyOfBackgroundCharged
          <<" 0.785*ResponseOfBackgroundCharged="<<ResponseOfBackgroundCharged<<endl;
//===>

//===> BG tracks: correction on energy and response of background tracks
           cout<<"---BEFORE correction on Bg tracks, NewResponse="<<NewResponse<<endl;

           NewResponse = NewResponse - EnergyOfBackgroundCharged + ResponseOfBackgroundCharged;

           cout<<"---AFTER correction on Bg tracks, NewResponse="<<NewResponse<<endl;
//===>

///======================= END CORRECTION ON BACKGROUND TRACKS ===============================   

   
   mScale = NewResponse/fJet.energy();
// Do nothing if mScale<0.
   if(mScale <0.) mScale=1;
   
   if(debug) std::cout<<" mScale= "<<mScale<<" NewResponse "<<NewResponse<<" Jet energy "<<fJet.energy()<<" event "<<fEvent.id().event()<<std::endl;

 return mScale;

} // correction

