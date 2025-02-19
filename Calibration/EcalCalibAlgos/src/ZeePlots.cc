
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/CalibElectron.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalCalibAlgos/interface/ZeePlots.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeKinematicTools.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"


#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

ZeePlots::ZeePlots( const char* fileName )
{

  fileName_ = fileName;
  file_ = new TFile(fileName_, "RECREATE");
}


ZeePlots::~ZeePlots()
{
  
  file_->Close();
  
  delete file_;

}

//========================================================================

void ZeePlots::openFile(){


  file_ -> cd();

}
//========================================================================

void ZeePlots::bookZMCHistograms(){

  file_ -> cd();

  h1_gen_ZMass_ = new TH1F("gen_ZMass","Generated Z mass",200,0.,150.);
  h1_gen_ZMass_->SetXTitle("gen_ZMass (GeV)");
  h1_gen_ZMass_->SetYTitle("events");

  h1_gen_ZEta_ = new TH1F("gen_ZEta","Eta of gen Z",200,-6.,6.);
  h1_gen_ZEta_->SetXTitle("#eta");
  h1_gen_ZEta_->SetYTitle("events");

  h1_gen_ZPhi_ = new TH1F("gen_ZPhi","Phi of gen Z",200,-4.,4.);
  h1_gen_ZPhi_->SetXTitle("#phi");
  h1_gen_ZPhi_->SetYTitle("events");

  h1_gen_ZRapidity_ = new TH1F("gen_ZRapidity","Rapidity of gen Z",200,-6.,6.);
  h1_gen_ZRapidity_->SetXTitle("Y");
  h1_gen_ZRapidity_->SetYTitle("events");

  h1_gen_ZPt_ = new TH1F("gen_ZPt","Pt of gen Z",200, 0.,100.);
  h1_gen_ZPt_->SetXTitle("p_{T} (GeV/c)");
  h1_gen_ZPt_->SetYTitle("events");


}

void ZeePlots::bookZHistograms(){

  file_ -> cd();

  h1_reco_ZEta_ = new TH1F("reco_ZEta","Eta of reco Z",200,-6.,6.);
  h1_reco_ZEta_->SetXTitle("#eta");
  h1_reco_ZEta_->SetYTitle("events");
  
  h1_reco_ZTheta_ = new TH1F("reco_ZTheta","Theta of reco Z",200, 0., 4.);
  h1_reco_ZTheta_->SetXTitle("#theta");
  h1_reco_ZTheta_->SetYTitle("events");
  
  h1_reco_ZRapidity_ = new TH1F("reco_ZRapidity","Rapidity of reco Z",200,-6.,6.);
  h1_reco_ZRapidity_->SetXTitle("Y");
  h1_reco_ZRapidity_->SetYTitle("events");
  
  h1_reco_ZPhi_ = new TH1F("reco_ZPhi","Phi of reco Z",100,-4.,4.);
  h1_reco_ZPhi_->SetXTitle("#phi");
  h1_reco_ZPhi_->SetYTitle("events");
  
  h1_reco_ZPt_ = new TH1F("reco_ZPt","Pt of reco Z",200,0.,100.);
  h1_reco_ZPt_->SetXTitle("p_{T} (GeV/c)");
  h1_reco_ZPt_->SetYTitle("events");
  
  
}

//========================================================================

void ZeePlots::fillZInfo( std::pair<calib::CalibElectron*,calib::CalibElectron*> myZeeCandidate ) {

  h1_reco_ZEta_->Fill( ZeeKinematicTools::calculateZEta(myZeeCandidate) );
  h1_reco_ZTheta_->Fill( ZeeKinematicTools::calculateZTheta(myZeeCandidate) );
  h1_reco_ZRapidity_->Fill( ZeeKinematicTools::calculateZRapidity(myZeeCandidate) );
  h1_reco_ZPhi_->Fill( ZeeKinematicTools::calculateZPhi(myZeeCandidate) );
  h1_reco_ZPt_->Fill( ZeeKinematicTools::calculateZPt(myZeeCandidate) );

}



//========================================================================

void ZeePlots::writeZHistograms() {

  file_->cd();

  h1_reco_ZEta_->Write();
  h1_reco_ZTheta_->Write();
  h1_reco_ZRapidity_->Write();
  h1_reco_ZPhi_->Write();
  h1_reco_ZPt_->Write();
  
}

//========================================================================

void ZeePlots::writeMCZHistograms() {

  file_->cd();
  
  h1_gen_ZRapidity_->Write();
  h1_gen_ZPt_->Write();
  h1_gen_ZPhi_->Write();

}

//========================================================================

void ZeePlots::fillZMCInfo( const HepMC::GenEvent* myGenEvent ) {

  file_->cd();

  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {//loop over MC particles
 
    if ( (*p)->pdg_id() == 23 && (*p)->status() == 2 ){
      
      h1_gen_ZMass_->Fill( (*p)->momentum().m() );
      h1_gen_ZEta_->Fill( (*p)->momentum().eta() );
      
      float genZ_Y = 0.5 * log ( ( (*p)->momentum().e() + (*p)->momentum().pz() ) /  ( (*p)->momentum().e() - (*p)->momentum().pz() ) )   ;
      
      h1_gen_ZRapidity_->Fill( genZ_Y );
      h1_gen_ZPt_->Fill((*p)->momentum().perp());
      h1_gen_ZPhi_->Fill((*p)->momentum().phi());

      

    }
  }//end loop over MC particles

  return;  
}

//========================================================================

void ZeePlots::bookEleMCHistograms(){

  file_->cd();

  h1_mcEle_Energy_ = new TH1F("mcEleEnergy","mc EleEnergy",300,0.,300.);
  h1_mcEle_Energy_->SetXTitle("E (GeV)");
  h1_mcEle_Energy_->SetYTitle("events");

  h1_mcElePt_ = new TH1F("mcElePt","p_{T} of MC electrons",300,0.,300.);
  h1_mcElePt_->SetXTitle("p_{T}(GeV/c)");
  h1_mcElePt_->SetYTitle("events");
  
  h1_mcEleEta_ = new TH1F("mcEleEta","Eta of MC electrons",100,-4.,4.);
  h1_mcEleEta_->SetXTitle("#eta");
  h1_mcEleEta_->SetYTitle("events");

  h1_mcElePhi_ = new TH1F("mcElePhi","Phi of MC electrons",100,-4.,4.);
  h1_mcElePhi_->SetXTitle("#phi");
  h1_mcElePhi_->SetYTitle("events");

}

//========================================================================

void ZeePlots::fillEleMCInfo( const HepMC::GenEvent* myGenEvent ) {

  file_->cd();
  
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	p != myGenEvent->particles_end(); ++p ) {
    
    if (  abs( (*p)->pdg_id() ) == 11 )
      {
	h1_mcEle_Energy_->Fill( (*p)->momentum().e() );
	h1_mcElePt_->Fill( (*p)->momentum().perp() );
	h1_mcEleEta_->Fill( (*p)->momentum().eta() );
	h1_mcElePhi_->Fill( (*p)->momentum().phi() );
	
      }//matches if (  abs( (*p)->pdg_id() ) == 11 )

  }//end loop over MC particles
  
}

//========================================================================
void ZeePlots::bookEleHistograms(){

  file_->cd();

  h1_nEleReco_ = new TH1F("h1_nEleReco", "h1_nEleReco", 20, 0, 20 );
  h1_nEleReco_->SetXTitle("Num. of reco electrons");
  
  h1_recoEleEnergy_ = new TH1F("recoEleEnergy","EleEnergy from SC",300,0.,300.);
  h1_recoEleEnergy_->SetXTitle("eleSCEnergy(GeV)");
  h1_recoEleEnergy_->SetYTitle("events");
  
  h1_recoElePt_ = new TH1F("recoElePt","p_{T} of reco electrons",300,0.,300.);
  h1_recoElePt_->SetXTitle("p_{T}(GeV/c)");
  h1_recoElePt_->SetYTitle("events");

  h1_recoEleEta_ = new TH1F("recoEleEta","Eta of reco electrons",100,-4.,4.);
  h1_recoEleEta_->SetXTitle("#eta");
  h1_recoEleEta_->SetYTitle("events");

   
  h1_recoElePhi_ = new TH1F("recoElePhi","Phi of reco electrons",100,-4.,4.);
  h1_recoElePhi_->SetXTitle("#phi");
  h1_recoElePhi_->SetYTitle("events");



}

//========================================================================

void ZeePlots::fillEleInfo(const reco::GsfElectronCollection* electronCollection) {

  file_->cd();

  h1_nEleReco_->Fill(electronCollection->size());
  
  for(reco::GsfElectronCollection::const_iterator eleIt = electronCollection->begin();   eleIt != electronCollection->end(); eleIt++)
    {
      
  file_->cd();

      h1_recoEleEnergy_->Fill( eleIt->superCluster()->energy() );
      h1_recoElePt_->Fill( eleIt->pt() );
      h1_recoEleEta_->Fill( eleIt->eta() );
      h1_recoElePhi_->Fill( eleIt->phi() );
      
    }//end loop on electrons   

}

//========================================================================

void ZeePlots::writeEleHistograms(){

  file_->cd();

  std::cout << "Start with ZeePlots::writeEleHistograms(), done file_->cd(); " << std::endl;
 
  h1_recoEleEnergy_->Write();
  h1_recoElePt_->Write();
  h1_recoEleEta_->Write();
  h1_recoElePhi_->Write();

  std::cout << "Done with ZeePlots::writeEleHistograms() " << std::endl;

}

//========================================================================

void ZeePlots::writeMCEleHistograms(){

  file_->cd();

  std::cout << "Start with ZeePlots::writeMCEleHistograms(), done file_->cd(); " << std::endl;

  h1_mcEle_Energy_->Write();
  h1_mcElePt_->Write();
  h1_mcEleEta_->Write();
  h1_mcElePhi_->Write();

  std::cout << "Done with ZeePlots::writeMCEleHistograms() " << std::endl;

}

//========================================================================

void ZeePlots::bookHLTHistograms(){

  file_->cd();

  h1_FiredTriggers_= new TH1F("h1_FiredTriggers", "h1_FiredTriggers", 5,0,5);
 
  h1_HLTVisitedEvents_ = new TH1F("h1_HLTVisitedEvents", "h1_HLTVisitedEvents", 5,0,5);

  h1_HLT1Electron_FiredEvents_ =  new TH1F("h1_HLT1Electron_FiredEvents", "h1_HLT1Electron_FiredEvents", 5,0,5);
  h1_HLT2Electron_FiredEvents_ =  new TH1F("h1_HLT2Electron_FiredEvents", "h1_HLT2Electron_FiredEvents", 5,0,5);
  h1_HLT2ElectronRelaxed_FiredEvents_ =  new TH1F("h1_HLT2ElectronRelaxed_FiredEvents", "h1_HLT2ElectronRelaxed_FiredEvents", 5,0,5);

  h1_HLT1Electron_HLT2Electron_FiredEvents_ =  new TH1F("h1_HLT1Electron_HLT2Electron_FiredEvents", "h1_HLT1Electron_HLT2Electron_FiredEvents", 5,0,5);
  h1_HLT1Electron_HLT2ElectronRelaxed_FiredEvents_ =  new TH1F("h1_HLT1Electron_HLT2ElectronRelaxed_FiredEvents", "h1_HLT1Electron_HLT2ElectronRelaxed_FiredEvents", 5,0,5);
  h1_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_ =  new TH1F("h1_HLT2Electron_HLT2ElectronRelaxed_FiredEvents", "h1_HLT2Electron_HLT2ElectronRelaxed_FiredEvents", 5,0,5);

  h1_HLT1Electron_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_ =  new TH1F("h1_HLT1Electron_HLT2Electron_HLT2ElectronRelaxed_FiredEvents", "h1_HLT1Electron_HLT2Electron_HLT2ElectronRelaxed_FiredEvents", 5,0,5);


}


//========================================================================

void ZeePlots::fillHLTInfo( edm::Handle<edm::TriggerResults> hltTriggerResultHandle ){

  file_->cd();
  
  int hltCount = hltTriggerResultHandle->size();

  bool aHLTResults[200] = { false };
    
  for(int i = 0 ; i < hltCount ; i++) {

    aHLTResults[i] = hltTriggerResultHandle->accept(i);
    if(aHLTResults[i])
      h1_FiredTriggers_->Fill(i);

    //HLT bit 32 = HLT1Electron
    //HLT bit 34 = HLT2Electron
    //HLT bit 35 = HLT2ElectronRelaxed

  }

  h1_HLTVisitedEvents_->Fill(1);
    
  if(aHLTResults[32] && !aHLTResults[34] && !aHLTResults[35])
    h1_HLT1Electron_FiredEvents_->Fill(1);

  if(aHLTResults[34] && !aHLTResults[32] && !aHLTResults[35])
    h1_HLT2Electron_FiredEvents_->Fill(1);

  if(aHLTResults[35] && !aHLTResults[32] && !aHLTResults[34])
    h1_HLT2ElectronRelaxed_FiredEvents_->Fill(1);

  if(aHLTResults[32] && aHLTResults[34] && !aHLTResults[35])
    h1_HLT1Electron_HLT2Electron_FiredEvents_->Fill(1);

  if(aHLTResults[32] && aHLTResults[35] && !aHLTResults[34])
    h1_HLT1Electron_HLT2ElectronRelaxed_FiredEvents_->Fill(1);

  if(aHLTResults[34] && aHLTResults[35] && !aHLTResults[32])
    h1_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_->Fill(1);

  if(aHLTResults[32] && aHLTResults[34] && aHLTResults[35])
    h1_HLT1Electron_HLT2Electron_HLT2ElectronRelaxed_FiredEvents_->Fill(1);



}


void ZeePlots::fillEleClassesPlots( calib::CalibElectron* myEle ){

  int myClass = myEle->getRecoElectron()->classification();

  float myEta = myEle->getRecoElectron()->eta();
  
  if(myClass==0 || myClass==100)
    h1_occupancyVsEtaGold_->Fill(myEta);
  
  std::cout<< "[ZeePlots::fillEleClassesPlots]Done gold"<< std::endl;
  
  if(myClass==40 || myClass==140)
    h1_occupancyVsEtaCrack_->Fill(myEta);
  
  std::cout<< "[ZeePlots::fillEleClassesPlots]Done crack"<< std::endl;
  
  if( (myClass>=30 && myClass<=34) || (myClass>=130 && myClass<=134) )
    h1_occupancyVsEtaShower_->Fill(myEta);
  
  std::cout<< "[ZeePlots::fillEleClassesPlots]Done shower"<< std::endl;
  
  if( myClass==10 || myClass==20 || myClass==110 || myClass ==120)
    h1_occupancyVsEtaSilver_->Fill(myEta);
  
  std::cout<< "[ZeePlots::fillEleClassesPlots]Done"<< std::endl;
  
}


void ZeePlots::bookEleClassesPlots(){

  file_->cd();

  h1_occupancyVsEtaGold_ = new TH1F("occupancyVsEtaGold","occupancyVsEtaGold", 200, -4.,4.);
  h1_occupancyVsEtaGold_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaGold_->SetXTitle("Eta channel");

  h1_occupancyVsEtaSilver_ = new TH1F("occupancyVsEtaSilver","occupancyVsEtaSilver", 200, -4.,4.);
  h1_occupancyVsEtaSilver_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaSilver_->SetXTitle("Eta channel");

  h1_occupancyVsEtaShower_ = new TH1F("occupancyVsEtaShower","occupancyVsEtaShower", 200, -4.,4.);
  h1_occupancyVsEtaShower_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaShower_->SetXTitle("Eta channel");

  h1_occupancyVsEtaCrack_ = new TH1F("occupancyVsEtaCrack","occupancyVsEtaCrack", 200, -4.,4.);
  h1_occupancyVsEtaCrack_->SetYTitle("Electron statistics");
  h1_occupancyVsEtaCrack_->SetXTitle("Eta channel");

}

void ZeePlots::writeEleClassesPlots(){

  file_->cd();

  h1_occupancyVsEtaGold_->Write();
  h1_occupancyVsEtaSilver_->Write();
  h1_occupancyVsEtaShower_->Write();
  h1_occupancyVsEtaCrack_->Write();

}
