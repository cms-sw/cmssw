#include <iostream>
//
#include "RecoEgamma/Examples/plugins/MCPhotonAnalyzer.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruthFinder.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"
// 
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
//
#include "DataFormats/Common/interface/Handle.h"
//
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// 
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
// 

 

using namespace std;

 
MCPhotonAnalyzer::MCPhotonAnalyzer( const edm::ParameterSet& pset )
   : fOutputFileName_( pset.getUntrackedParameter<string>("HistOutFile",std::string("TestConversions.root")) ),
     fOutputFile_(0)
{

  
}



MCPhotonAnalyzer::~MCPhotonAnalyzer() {


  delete thePhotonMCTruthFinder_;

}


void MCPhotonAnalyzer::beginJob( const edm::EventSetup& setup)
{


  nEvt_=0;
  
  thePhotonMCTruthFinder_ = new PhotonMCTruthFinder();

  fOutputFile_   = new TFile( fOutputFileName_.c_str(), "RECREATE" ) ;

 //// All MC photons  
  h_MCPhoE_ = new TH1F("MCPhoE","MC photon energy",100,0.,100.);
  h_MCPhoPhi_ = new TH1F("MCPhoPhi","MC photon phi",40,-3.14, 3.14);
  h_MCPhoEta_ = new TH1F("MCPhoEta","MC photon eta",40,-3., 3.);
  h_MCPhoEta1_ = new TH1F("MCPhoEta1","MC photon eta",40,-3., 3.);
  h_MCPhoEta2_ = new TH1F("MCPhoEta2","MC photon eta",40,-3., 3.);
  h_MCPhoEta3_ = new TH1F("MCPhoEta3","MC photon eta",40,-3., 3.);
  /// conversions
  h_MCConvPhoE_ = new TH1F("MCConvPhoE","MC converted photon energy",100,0.,100.);
  h_MCConvPhoPhi_ = new TH1F("MCConvPhoPhi","MC converted photon phi",40,-3.14, 3.14);
  h_MCConvPhoEta_ = new TH1F("MCConvPhoEta","MC converted photon eta",40,-3., 3.);
  h_MCConvPhoR_ = new TH1F("MCConvPhoR","MC converted photon R",120,0.,120.);

  h_MCConvPhoREta1_ = new TH1F("MCConvPhoREta1","MC converted photon R",120,0.,120.);
  h_MCConvPhoREta2_ = new TH1F("MCConvPhoREta2","MC converted photon R",120,0.,120.);
  h_MCConvPhoREta3_ = new TH1F("MCConvPhoREta3","MC converted photon R",120,0.,120.);

  h_convFracEta1_ = new TH1F("convFracEta1","Integrated(R) fraction of conversion |eta|=0.2",120,0.,120.);
  h_convFracEta2_ = new TH1F("convFracEta2","Integrated(R) fraction of conversion |eta|=0.9",120,0.,120.);
  h_convFracEta3_ = new TH1F("convFracEta3","Integrated(R) fraction of conversion |eta|=1.5",120,0.,120.);
  /// conversions with two tracks
  h_MCConvPhoTwoTracksE_ = new TH1F("MCConvPhoTwoTracksE","MC converted photon with 2 tracks  energy",100,0.,100.);
  h_MCConvPhoTwoTracksPhi_ = new TH1F("MCConvPhoTwoTracksPhi","MC converted photon 2 tracks  phi",40,-3.14, 3.14);
  h_MCConvPhoTwoTracksEta_ = new TH1F("MCConvPhoTwoTracksEta","MC converted photon 2 tracks eta",40,-3., 3.);
  h_MCConvPhoTwoTracksR_ = new TH1F("MCConvPhoTwoTracksR","MC converted photon 2 tracks eta",48,0.,120.);
  // conversions with one track
  h_MCConvPhoOneTrackE_ = new TH1F("MCConvPhoOneTrackE","MC converted photon with 1 track  energy",100,0.,100.);
  h_MCConvPhoOneTrackPhi_ = new TH1F("MCConvPhoOneTrackPhi","MC converted photon 1 track  phi",40,-3.14, 3.14);
  h_MCConvPhoOneTrackEta_ = new TH1F("MCConvPhoOneTrackEta","MC converted photon 1 track eta",40,-3., 3.);
  h_MCConvPhoOneTrackR_ = new TH1F("MCConvPhoOneTrackR","MC converted photon 1 track eta",48,0.,120.);

  /// electrons from conversions
  h_MCEleE_ = new TH1F("MCEleE","MC ele energy",100,0.,200.);
  h_MCElePhi_ = new TH1F("MCElePhi","MC ele phi",40,-3.14, 3.14);
  h_MCEleEta_ = new TH1F("MCEleEta","MC ele eta",40,-3., 3.);
  h_BremFrac_ = new TH1F("bremFrac","brem frac ", 100, 0., 1.);
  h_BremEnergy_ = new TH1F("BremE","Brem energy",100,0.,200.);
  h_EleEvsPhoE_ = new TH2F ("eleEvsPhoE","eleEvsPhoE",100,0.,200.,100,0.,200.);  
  h_bremEvsEleE_ = new TH2F ("bremEvsEleE","bremEvsEleE",100,0.,200.,100,0.,200.);  

  p_BremVsR_ = new TProfile("BremVsR", " Mean Brem Energy vs R ", 48, 0., 120.);
  p_BremVsEta_ = new TProfile("BremVsEta", " Mean Brem Energy vs Eta ", 50, -2.5, 2.5);

  p_BremVsConvR_ = new TProfile("BremVsConvR", " Mean Brem Fraction vs conversion R ", 48, 0., 120.);
  p_BremVsConvEta_ = new TProfile("BremVsConvEta", " Mean Brem Fraction vs converion Eta ", 50, -2.5, 2.5);

  h_bremFracVsConvR_ = new TH2F ("bremFracVsConvR","brem Fraction vs conversion R",60,0.,120.,100,0.,1.);  
  
  return ;
}


float MCPhotonAnalyzer::etaTransformation(  float EtaParticle , float Zvertex)  {

//---Definitions
	const float PI    = 3.1415927;
	const float TWOPI = 2.0*PI;

//---Definitions for ECAL
	const float R_ECAL           = 136.5;
	const float Z_Endcap         = 328.0;
	const float etaBarrelEndcap  = 1.479; 
   
//---ETA correction

	float Theta = 0.0  ; 
        float ZEcal = R_ECAL*sinh(EtaParticle)+Zvertex;

	if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
	if(Theta<0.0) Theta = Theta+PI ;
	float ETA = - log(tan(0.5*Theta));
         
	if( fabs(ETA) > etaBarrelEndcap )
	  {
	   float Zend = Z_Endcap ;
	   if(EtaParticle<0.0 )  Zend = -Zend ;
	   float Zlen = Zend - Zvertex ;
	   float RR = Zlen/sinh(EtaParticle); 
	   Theta = atan(RR/Zend);
	   if(Theta<0.0) Theta = Theta+PI ;
 	   ETA = - log(tan(0.5*Theta));		      
	  } 
//---Return the result
        return ETA;
//---end
}

float MCPhotonAnalyzer::phiNormalization(float & phi)
{
//---Definitions
 const float PI    = 3.1415927;
 const float TWOPI = 2.0*PI;


 if(phi >  PI) {phi = phi - TWOPI;}
 if(phi < -PI) {phi = phi + TWOPI;}

 //  cout << " Float_t PHInormalization out " << PHI << endl;
 return phi;

}





void MCPhotonAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& )
{
  
  
  using namespace edm;
  const float etaPhiDistance=0.01;
  // Fiducial region
  const float TRK_BARL =0.9;
  const float BARL = 1.4442; // DAQ TDR p.290
  const float END_LO = 1.566;
  const float END_HI = 2.5;
 // Electron mass
  const Float_t mElec= 0.000511;


  nEvt_++;  
  LogInfo("mcEleAnalyzer") << "MCPhotonAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";
  //  LogDebug("MCPhotonAnalyzer") << "MCPhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";
  std::cout << "MCPhotonAnalyzer Analyzing event number: "  << e.id() << " Global Counter " << nEvt_ <<"\n";



  //////////////////// Get the MC truth: SimTracks   
  std::cout  << " MCPhotonAnalyzer Looking for MC truth " << "\n";
  
  //get simtrack info
  std::vector<SimTrack> theSimTracks;
  std::vector<SimVertex> theSimVertices;
  
  edm::Handle<SimTrackContainer> SimTk;
  edm::Handle<SimVertexContainer> SimVtx;
  e.getByLabel("g4SimHits",SimTk);
  e.getByLabel("g4SimHits",SimVtx);
  
  theSimTracks.insert(theSimTracks.end(),SimTk->begin(),SimTk->end());
  theSimVertices.insert(theSimVertices.end(),SimVtx->begin(),SimVtx->end());
  std::cout << " MCPhotonAnalyzer This Event has " <<  theSimTracks.size() << " sim tracks " << std::endl;
  std::cout << " MCPhotonAnalyzer This Event has " <<  theSimVertices.size() << " sim vertices " << std::endl;
  if (  ! theSimTracks.size() ) std::cout << " Event number " << e.id() << " has NO sim tracks " << std::endl;
  
  
  std::vector<PhotonMCTruth> mcPhotons=thePhotonMCTruthFinder_->find (theSimTracks,  theSimVertices);  
  std::cout << " MCPhotonAnalyzer mcPhotons size " <<  mcPhotons.size() << std::endl;
 

 
  for ( std::vector<PhotonMCTruth>::const_iterator iPho=mcPhotons.begin(); iPho !=mcPhotons.end(); ++iPho ){
    std::vector<ElectronMCTruth> mcElectrons=(*iPho).electrons();
    std::cout << " mcEleAnalyzer mcElectrons size " <<  mcElectrons.size() << std::endl;

    if ( (*iPho).fourMomentum().e() < 35 ) continue;

    h_MCPhoE_->Fill  ( (*iPho).fourMomentum().e() );
    h_MCPhoEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    h_MCPhoPhi_->Fill  ( (*iPho).fourMomentum().phi() );

    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.25 &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.15  ) 
      h_MCPhoEta1_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.95  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.85  ) 
      h_MCPhoEta2_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 1.55  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.45  ) 
      h_MCPhoEta3_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
    
    
    
    //    if ( (*iPho).isAConversion()  && (*iPho).vertex().perp()< 10 ) {
        if ( (*iPho).isAConversion() ) {

      
      for ( std::vector<ElectronMCTruth>::const_iterator iEl=mcElectrons.begin(); iEl !=mcElectrons.end(); ++iEl ){

	//	if (  (*iEl).fourMomentum().e() < 35  ) continue;

	h_MCEleE_->Fill  ( (*iEl).fourMomentum().e() );
	h_MCEleEta_->Fill  ( (*iEl).fourMomentum().pseudoRapidity() );
	h_MCElePhi_->Fill  ( (*iEl).fourMomentum().phi() );

	h_EleEvsPhoE_->Fill ( (*iPho).fourMomentum().e(), (*iEl).fourMomentum().e() ); 
	  
	float totBrem=0;
	for ( int iBrem=0; iBrem < (*iEl).bremVertices().size(); ++iBrem ) {

	  float rBrem=  (*iEl).bremVertices()[iBrem].perp();
          float etaBrem=(*iEl).bremVertices()[iBrem].eta();
	  if ( rBrem < 120 ) {
	    totBrem +=  (*iEl).bremMomentum()[iBrem].e();
	    p_BremVsR_ ->Fill ( rBrem, (*iEl).bremMomentum()[iBrem].e() );   
	    p_BremVsEta_ ->Fill ( etaBrem, (*iEl).bremMomentum()[iBrem].e() );   

	  }

	}


	h_BremFrac_->Fill( totBrem/(*iEl).fourMomentum().e() );
	h_BremEnergy_->Fill (  totBrem  );
	h_bremEvsEleE_->Fill ( (*iEl).fourMomentum().e(),  totBrem );
       
        p_BremVsConvR_ ->Fill ( (*iPho).vertex().perp(), totBrem/(*iEl).fourMomentum().e());
        p_BremVsConvEta_ ->Fill ( (*iPho).vertex().eta(), totBrem/(*iEl).fourMomentum().e());

	h_bremFracVsConvR_ -> Fill ((*iPho).vertex().perp(), totBrem/(*iEl).fourMomentum().e());

      }





      h_MCConvPhoE_->Fill  ( (*iPho).fourMomentum().e() );
      h_MCConvPhoEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
      h_MCConvPhoPhi_->Fill  ( (*iPho).fourMomentum().phi() );
      h_MCConvPhoR_->Fill  ( (*iPho).vertex().perp() );
      
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.25 &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.15  )       
	h_MCConvPhoREta1_->Fill  ( (*iPho).vertex().perp() );
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 0.95  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=0.85  ) 
	h_MCConvPhoREta2_->Fill  ( (*iPho).vertex().perp() );
      if ( fabs((*iPho).fourMomentum().pseudoRapidity() ) <= 1.55  &&  fabs((*iPho).fourMomentum().pseudoRapidity() ) >=1.45  )  
	h_MCConvPhoREta3_->Fill  ( (*iPho).vertex().perp() );
      
      if ( (*iPho).electrons().size() == 2 ) {
	h_MCConvPhoTwoTracksE_->Fill  ( (*iPho).fourMomentum().e() );
	h_MCConvPhoTwoTracksEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
	h_MCConvPhoTwoTracksPhi_->Fill  ( (*iPho).fourMomentum().phi() );
	h_MCConvPhoTwoTracksR_->Fill  ( (*iPho).vertex().perp() );
      } else if ( (*iPho).electrons().size() == 1 ) {
	h_MCConvPhoOneTrackE_->Fill  ( (*iPho).fourMomentum().e() );
	h_MCConvPhoOneTrackEta_->Fill  ( (*iPho).fourMomentum().pseudoRapidity() );
	h_MCConvPhoOneTrackPhi_->Fill  ( (*iPho).fourMomentum().phi() );
	h_MCConvPhoOneTrackR_->Fill  ( (*iPho).vertex().perp() );
      }      
    } // end conversions





  }   /// Loop over all MC photons in the event


  

}




void MCPhotonAnalyzer::endJob()
{


  int s1, s2, s3;
  s1=s2=s3=0;
  int e1, e2, e3;
  e1=e2=e3=0;

  int nTotEta1 = h_MCPhoEta1_->GetEntries();
  int nTotEta2 = h_MCPhoEta2_->GetEntries();
  int nTotEta3 = h_MCPhoEta3_->GetEntries();

  for ( int i=1; i<=120; ++i) {
  e1 = h_MCConvPhoREta1_->GetBinContent(i);
  e2 = h_MCConvPhoREta2_->GetBinContent(i);
  e3 = h_MCConvPhoREta3_->GetBinContent(i);
  s1+=e1;
  s2+=e2;
  s3+=e3;
  h_convFracEta1_->SetBinContent(i,float(s1)*100/float(nTotEta1));
  h_convFracEta2_->SetBinContent(i,float(s2)*100/float(nTotEta2));
  h_convFracEta3_->SetBinContent(i,float(s3)*100/float(nTotEta3));




  
}


       
   fOutputFile_->Write() ;
   fOutputFile_->Close() ;
  
   edm::LogInfo("MCPhotonAnalyzer") << "Analyzed " << nEvt_  << "\n";
   std::cout  << "MCPhotonAnalyzer::endJob Analyzed " << nEvt_ << " events " << "\n";

   return ;
}
 
