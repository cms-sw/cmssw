#include "UEAnalysisAntiKtJets.h"
#include <vector>
#include <math.h>

UEAnalysisAntiKtJets::UEAnalysisAntiKtJets()
{
  //  cout << "UEAnalysisAntiKtJets constructor " <<endl;
  piG = acos(-1.);

}

//void UEAnalysisAntiKtJets::Begin(TFile * f){
void UEAnalysisAntiKtJets::Begin(TFile * f, string hltBit)
{
  f->cd( hltBit.c_str() );

   h_pTJet = new TH1D("h_pTJet", "h_pTJet;p_{T}(jet 1) (GeV/c)", 150, 0., 300. );
   h_nConstituents = new TH1D("h_nConstituents", "h_nConstituents;N(jet constituents)", 30, 0.5, 30.5);
   h_pTSumConstituents = new TH1D("h_pTSumConstituents", "h_pTSumConstituents;d#Sigmap_{T}(jet constituents) (GeV/c)", 150, 0., 300.);
   h_pTByNConstituents = new TH1D("h_pTByNConstituents", "h_pTByNConstituents;p_{T}(jet 1)/N(jet constituents) (GeV/c)", 100, 0., 100. );
   h_areaJet1 = new TH1D("h_areaJet1", "h_areaJet1;area(jet 1) (rad)", 100, 0., 1.5);
   h_pTConstituent = new TH1D("h_pTConstituent", "h_pTConstituent;p_{T}(jet constituent) (GeV/c)", 100, 0., 100. );
   h_dphiJC = new TH1D("h_dphiJC", "h_dphiJC;#Delta#phi(jet 1, jet constituent) (rad)", 100, 0., TMath::Pi() );
   h_dphiEcal = new TH1D("h_dphiEcal", "h_dphiEcal;constituent #Delta#phi(Vtx, ECAL) (rad)", 100, 0., 1.5 );
   h_pTAllJets = new TH1D("h_pTAllJets", "h_pTAllJets;p_{T}(jet) (GeV/c)", 150, 0., 300.);
   h_areaAllJets = new TH1D("h_areaAllJets", "h_areaAllJets;area(jet) (rad);", 100, 0., 1.5);
   h_pTByAreaAllJets = new TH1D("h_pTByAreaAllJets", "h_pTByAreaAllJets;< p_{T}(jet)/area(jet) > (GeV/c / rad)", 150, 0., 600.);

   h2d_nConstituents_vs_pTJet = new TH2D("h2d_nConstituents_vs_pTJet", 
					 "h2d_nConstituents_vs_pTJet;p_{T}(jet 1) (GeV/c);N(jet constituents)",
					 150, 0., 300., 30, 0.5, 30.5);
   h2d_pTSumConstituents_vs_pTJet = new TH2D("h2d_pTSumConstituents_vs_pTJet",
					     "h2d_pTSumConstituents_vs_pTJet;p_{T}(jet 1) (GeV/c);d#Sigmap_{T}(jet constituents) (GeV/c)",
					     150, 0., 300., 150, 0., 300.);
   h2d_pTByNConstituents_vs_pTJet = new TH2D("h2d_pTByNConstituents_vs_pTJet",
					     "h2d_pTByNConstituents_vs_pTJet;p_{T}(jet 1) (GeV/c);p_{T}(jet 1)/N(jet constituents) (GeV/c)",
					     150, 0., 300., 100, 0., 100.);
   h2d_areaJet1_vs_pTJet1 = new TH2D("h2d_areaJet1_vs_pTJet1", 
				     "h2d_areaJet1_vs_pTJet1;p_{T}(jet 1) (GeV/c);jet 1 area (rad)",
				     150, 0., 300., 100, 0., 1.5);
   h2d_pTConstituent_vs_pTJet = new TH2D("h2d_pTConstituent_vs_pTJet",
					 "h2d_pTConstituent_vs_pTJet;p_{T}(jet 1) (GeV/c);<p_{T}>(jet constituents) (GeV/c)",
					 150, 0., 300., 100, 0., 100.);
   h2d_dphiJC_vs_pTConstituent = new TH2D("h2d_dphiJC_vs_pTConstituent",
					  "h2d_dphiJC_vs_pTConstituent;p_{T}(jet constituent) (GeV/c);#Delta#phi(jet 1, jet constituent) (rad)",
					  100, 0., 50., 100, 0., TMath::Pi() );
   h2d_dphiJC_vs_pTJet = new TH2D("h2d_dphiJC_vs_pTJet", 
				  "h2d_dphiJC_vs_pTJet;p_{T}(jet 1) (GeV/c);#Delta#phi(jet 1, jet constituent) (rad)",
				  150, 0., 300., 100, 0., TMath::Pi() );
   h2d_dphiEcal_vs_pTConstituent = new TH2D("h2d_dphiEcal_vs_pTConstituent",
					    "h2d_dphiEcal_vs_pTConstituent;p_{T}(jet constituent) (GeV/c);constituent #Delta#phi(Vtx, ECAL) (rad)",
					    100, 0., 3., 100, 0., 1.5 );
   h2d_dphiEcal_vs_pTJet = new TH2D("h2d_dphiEcal_vs_pTJet", 
				    "h2d_dphiEcal_vs_pTJet;p_{T}(jet 1) (GeV/c);constituent #Delta#phi(Vtx, ECAL) (rad)",
				    150, 0., 300., 100, 0., 1.5 );
   h2d_pTByAreaAllJets_vs_pTJet = new TH2D("h2d_pTByAreaAllJets_vs_pTJet", 
					   "h2d_pTByAreaAllJets_vs_pTJet;p_{T}(jet 1) (GeV/c);< p_{T}(jet)/area(jet) > (GeV/c / rad)", 
					   150, 0., 300., 150, 0., 600.);
      
}

double UEAnalysisAntiKtJets::ecalPhi(const float ptParticle)
{
  const float R_ECAL           = 136.5;
  const float magFieldInTesla  = 4.;

  float phiParticle = 0.;

  // Magnetic field

  const float RBARM = 1.357 ;  // was 1.31 , updated on 16122003
  const float ZENDM = 3.186 ;  // was 3.15 , updated on 16122003

  float rbend = RBARM;
  float bend  = 0.3 * magFieldInTesla * rbend / 2.0; 
  float phi = 0.0;

  // only valid if track is in barrel
  if (TMath::Abs(bend/ptParticle) <= 1.) 
    {
      phi = (-1.)*(phiParticle - asin(bend/ptParticle));
      if(phi >  TMath::Pi()) phi = phi - TMath::TwoPi();
      if(phi < -TMath::Pi()) phi = phi + TMath::TwoPi();
    } else {
      cout << "[EcalPositionFromTrack::phiTransformation] Warning: "
	   << "Too low Pt, giving up" << endl;
      return phiParticle;
    }
  
  return phi;

}


void UEAnalysisAntiKtJets::jetAnalysis(float weight, float etaRegion, float ptThreshold, TClonesArray* Track, TFile* f, string hltBit)
{
  f->cd( hltBit.c_str() );

  // FastJet package
  //
  // written by Matteo Cacciari, Gavin Salam and Gregory Soyez
  // http://www.lpthe.jussieu.fr/~salam/fastjet/
  //
  // Phys. Lett. B 641 (2006) 57 (FastJet package)
  // arXiv:0802.1188 (jet area)
 

  // return if no four-vectors are provided
  if ( Track->GetSize() == 0 ) return;

  // prepare input 
  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.reserve ( Track->GetSize() );

  int iJet( 0 );
  for(int i=0;i<Track->GetSize();++i) 
    {
      TLorentzVector *v = (TLorentzVector*)Track->At(i);

      if ( TMath::Abs(v->Eta()) > etaRegion   ) continue;
      if ( v->Pt()              < ptThreshold ) continue;
      
      fjInputs.push_back (fastjet::PseudoJet (v->Px(), v->Py(), v->Pz(), v->E()) );
      fjInputs.back().set_user_index(iJet);
      ++iJet;
    }
  // return if no four-vectors in visible phase space
  if ( fjInputs.size() == 0 ) return;

  // create an object that represents your choice of jet finder and 
  // the associated parameters
  // run the jet clustering with the above jet definition
 
  // parameters from RecoJets/JetProducers/data/kt4GenJets.cff (CMSSW)
  double rParam   ( 0.4 );
  double mJetPtMin( 1.  );

  // parameter from RecoJets/JetProducers/data/KtJetParameters.cfi (CMSSW)
  fastjet::Strategy fjStrategy( fastjet::Best );
  //  fastjet::JetDefinition* mJetDefinition( new fastjet::JetDefinition (fastjet::kt_algorithm, rParam, fjStrategy) );
  fastjet::JetDefinition* mJetDefinition( new fastjet::JetDefinition (fastjet::antikt_algorithm, rParam, fjStrategy) );
    
  // parameters from RecoJets/JetProducers/data/FastjetParameters.cfi (CMSSW)
      
  double ghostEtaMax   ( 0. );
  int activeAreaRepeats( 0 );
  double ghostArea     ( 1. );

  // calculate jet areas (if commented out, no areas are calculated)
  ghostEtaMax       = 6.;
  activeAreaRepeats = 5;
  ghostArea         = 0.01;

  fastjet::GhostedAreaSpec* mActiveArea( new fastjet::ActiveAreaSpec (ghostEtaMax, activeAreaRepeats, ghostArea) );

  // print out info on current jet algorithm
  //   cout << endl;
  //   cout << mJetDefinition->description() << endl;
  //   cout << mActiveArea->description() << endl;

  // here we need to keep both pointers, as "area" interfaces are missing in base class
  fastjet::ClusterSequenceActiveArea* clusterSequenceWithArea( 0 );
  fastjet::ClusterSequenceWithArea* clusterSequence( 0 );
  if (mActiveArea) 
    {
      clusterSequenceWithArea = new fastjet::ClusterSequenceActiveArea (fjInputs, *mJetDefinition, *mActiveArea);
      clusterSequence         = clusterSequenceWithArea;
    }
  else 
    {
      clusterSequence = new fastjet::ClusterSequenceWithArea (fjInputs, *mJetDefinition);
    }

  // retrieve jets for selected mode
  std::vector<fastjet::PseudoJet> jets( clusterSequence->inclusive_jets (mJetPtMin) );
     
  // get PU pt
  double median_Pt_Per_Area = clusterSequenceWithArea ? clusterSequenceWithArea->pt_per_unit_area() : 0.;

  //  int columnwidth( 10 );

  // process found jets
  //   cout << "found " << jets.size() << " jets with median_Pt_Per_Area " << median_Pt_Per_Area << endl;
  //   cout.width( 5 );
  //   cout << "jet";
  //   cout.width( columnwidth );
  //   cout << "eta";
  //   cout.width( columnwidth );
  //   cout << "phi";
  //   cout.width( columnwidth );
  //   cout << "pT";
  //   cout.width( columnwidth );
  //   cout << "jetArea";
  //   cout.width( 15 );
  //   cout << "pT / jetArea";
  //   cout << endl;
  
  //   for ( int i(0); i<jets.size(); ++i )
  //     {
  //       cout.width( 5 );
  //       cout << i;
  //       cout.width( columnwidth );
  //       cout << jets[i].eta();
  //       cout.width( columnwidth );
  //       cout << jets[i].phi();
  //       cout.width( columnwidth );
  //       cout << jets[i].perp();
  //       cout.width( columnwidth );
  //       cout << clusterSequenceWithArea->area(jets[i]);
  //       cout.width( 15 );
  //       cout << jets[i].perp()/clusterSequenceWithArea->area(jets[i]);
  //       cout << endl;
  //     }

  vector<fastjet::PseudoJet> sorted_jets = sorted_by_pt(jets); 

  vector<fastjet::PseudoJet>::iterator jet1It  ( sorted_jets.begin() );
  vector<fastjet::PseudoJet>::iterator jetIt   ( sorted_jets.begin() );
  vector<fastjet::PseudoJet>::iterator jetItEnd( sorted_jets.end()   );

  double pTByAreaSum( 0. );
  for ( ; jetIt!=jetItEnd; ++jetIt )
    {
      h_pTAllJets  ->Fill( (*jetIt).perp()                      , weight );
      h_areaAllJets->Fill( clusterSequenceWithArea->area(*jetIt), weight );

      pTByAreaSum += (*jetIt).perp()/clusterSequenceWithArea->area(*jetIt);
    }
  
  if ( sorted_jets.size() > 0 )
    {
      fastjet::PseudoJet jet1( sorted_jets[0] );
      h_pTByAreaAllJets->Fill( pTByAreaSum/sorted_jets.size(), weight );
      h2d_pTByAreaAllJets_vs_pTJet->Fill( jet1.perp() , pTByAreaSum/sorted_jets.size(), weight );

      //std::vector< PseudoJet > constituents (const PseudoJet &jet) const 
      // return a vector of the particles that make up jet 

      std::vector< fastjet::PseudoJet > constituents( clusterSequenceWithArea->constituents(jet1) );

      //       cout << "hardest jet: pT=" << jet1.perp() 
      // 	   << ", n(constituents)=" << constituents.size() 
      // 	   << ", fraction=" << jet1.perp()/constituents.size()
      // 	   << ", area=" << clusterSequenceWithArea->area(jet1)
      // 	   << endl;

      h_pTJet            ->Fill( jet1.perp()                        , weight );
      h_nConstituents    ->Fill( constituents.size()                , weight );
      h_pTByNConstituents->Fill( jet1.perp()/constituents.size()    , weight );
      h_areaJet1         ->Fill( clusterSequenceWithArea->area(jet1), weight );

      h2d_nConstituents_vs_pTJet    ->Fill( jet1.perp(), constituents.size()                , weight );
      h2d_pTByNConstituents_vs_pTJet->Fill( jet1.perp(), jet1.perp()/constituents.size()    , weight );
      h2d_areaJet1_vs_pTJet1        ->Fill( jet1.perp(), clusterSequenceWithArea->area(jet1), weight );

      std::vector<fastjet::PseudoJet>::iterator it   ( constituents.begin() );
      std::vector<fastjet::PseudoJet>::iterator itEnd( constituents.end()   );

      double pTSumConstituents( 0. );
      for ( ; it!=itEnd; ++it )
	  {
		  //	  cout << "\tconstituent pT=" << (*it).perp() << endl;
		  
		  h_pTConstituent->Fill( (*it).perp(), weight );
		  pTSumConstituents += (*it).perp();
		  
		  TVector2* jvec = new TVector2(jet1.px(), jet1.py());
		  TVector2* cvec = new TVector2((*it).px(), (*it).py());
		  h_dphiJC                    ->Fill( TMath::Abs(jvec->DeltaPhi(*cvec)), weight );
		  h2d_dphiJC_vs_pTConstituent ->Fill( (*it).perp(), TMath::Abs(jvec->DeltaPhi(*cvec)), weight );
		  h2d_dphiJC_vs_pTJet         ->Fill( jet1.perp() , TMath::Abs(jvec->DeltaPhi(*cvec)), weight );
		  jvec->Delete();
		  cvec->Delete();
		  
		  //	  cout << "phi=" << (*it).phi() << ", dphi(phi,ecalphi)=" << ecalPhi((*it).perp()) << endl;
		  h_dphiEcal                   ->Fill( ecalPhi((*it).perp()), weight );
		  h2d_dphiEcal_vs_pTConstituent->Fill( (*it).perp(), ecalPhi((*it).perp()), weight );
		  h2d_dphiEcal_vs_pTJet        ->Fill( jet1.perp() , ecalPhi((*it).perp()), weight );
	  }
      h_pTSumConstituents->Fill( pTSumConstituents, weight );
      h2d_pTSumConstituents_vs_pTJet->Fill( jet1.perp(), pTSumConstituents, weight );
	  
      // event average of constituent pT
      h2d_pTConstituent_vs_pTJet->Fill( jet1.perp(), pTSumConstituents/constituents.size(), weight );
	}
  
  
  iJet = 0;
  int iSavedJet( 0 );

  //   // clear jet array from former entries
  //   Jets->Clear();
  //   for (std::vector<fastjet::PseudoJet>::const_iterator jet=jets.begin(); jet!=jets.end();++jet , ++iJet) 
  //     {
  //       if ( jet->perp() < CUTPTJET ) continue; // do not save soft jets
  
  //       new((*Jets)[iSavedJet]) TLorentzVector(jet->px(), jet->py(), jet->pz(), jet->e());
  //       ++iSavedJet;
  //     }
  
  // cleanup
  if (clusterSequenceWithArea) delete clusterSequenceWithArea;
  else                         delete clusterSequence; 


  // ===== end FastJet code

}

void UEAnalysisAntiKtJets::writeToFile(TFile * file){
  //  file->Write();
}
