// -*- C++ -*-
//
// Package:    GeneratorInterface
// Class:      EvtGenTestAnalyzer
// 
//
// Description: Module to analyze Pythia-EvtGen HepMCproducts
//
//
// Original Author:  Roberto Covarelli
//         Created:  April 26, 2007
//

#include <iostream>
#include <fstream>
 
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
 
// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
 
#include "TFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TObjArray.h"
 
#include "FWCore/Framework/interface/MakerMacros.h"

#include "GeneratorInterface/ExternalDecays/test/EvtGenTestAnalyzer.h"

using namespace edm;
using namespace std;
using namespace HepMC;
 
EvtGenTestAnalyzer::EvtGenTestAnalyzer( const ParameterSet& pset )
   : fOutputFileName( pset.getUntrackedParameter<string>("HistOutFile",std::string("TestBs.root")) ),
     theSrc( pset.getUntrackedParameter<string>("theSrc",std::string("source")) ), 
     fOutputFile(0)
{
}

void EvtGenTestAnalyzer::beginJob()
{
 
   nevent = 0;
   nbs = 0;

   fOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   // fHist2muMass  = new TH1D(  "Hist2muMass", "2-mu inv. mass", 100,  60., 120. ) ;  
   hGeneralId = new TH1D( "hGeneralId","LundIDs of all particles",  100, -1000., 1000.) ;
   hnB = new TH1D( "hnB", "N(B)", 10,  0., 10. ) ;
   hnJpsi = new TH1D( "hnJpsi", "N(Jpsi)", 10,  0., 10. ) ;
   hnBz = new TH1D( "hnBz", "N(B0)", 10,  0., 10. ) ;
   hnBzb = new TH1D( "hnBzb", "N(B0bar)", 10,  0., 10. ) ;
   hMinvb  = new TH1D( "hMinvb",  "B invariant mass",  100,  5.0, 6.0 ) ;
   hPtbs = new TH1D( "hPtbs", "Pt Bs", 100,  0., 50. ) ;
   hPbs  = new TH1D( "hPbs",  "P Bs",  100,  0., 200. ) ;
   hPhibs = new TH1D( "hPhibs","Phi Bs",  100,  -3.14, 3.14) ;
   hEtabs = new TH1D( "hEtabs","Eta Bs",  100,  -7.0, 7.0) ;
   hPtmu = new TH1D( "hPtmu", "Pt Mu", 100,  0., 50. ) ;
   hPmu  = new TH1D( "hPmu",  "P Mu",  100,  0., 200. ) ;
   hPhimu = new TH1D( "hPhimu","Phi Mu",  100,  -3.14, 3.14) ;
   hEtamu = new TH1D( "hEtamu","Eta Mu",  100,  -7.0, 7.0) ;
   hPtRadPho  = new TH1D( "hPtRadPho",  "Pt radiated photon",  100,  0., 200. ) ;
   hPhiRadPho = new TH1D( "hPhiRadPho","Phi radiated photon",  100,  -3.14, 3.14) ;
   hEtaRadPho = new TH1D( "hEtaRadPho","Eta radiated photon",  100,  -7.0, 7.0) ;
   htbPlus = new TH1D( "htbPlus", "B+ proper decay time", 50, 0., 12. ) ;
   htbUnmix = new TH1D( "htbUnmix", "B0 proper decay time (unmixed)", 50, 0., 12. ) ;
   htbMix = new TH1D( "htbMix", "B0 proper decay time (mixed)", 50, 0., 12. ) ;
   htbMixPlus = new TH1D( "htbMixPlus", "B0 proper decay time (mixed positive)", 50, 0., 12. ) ;
   htbMixMinus = new TH1D( "htbMixMinus", "B0 proper decay time (mixed negative)", 50, 0., 12. ) ;
   htbsUnmix = new TH1D( "htbsUnmix", "Bs proper decay time (unmixed)", 50, 0., 12. ) ;
   htbsMix = new TH1D( "htbsMix", "Bs proper decay time (mixed)", 50, 0., 12. ) ;
   htbJpsiKs = new TH1D( "htbJpsiKs", "B0 -> J/#psiK_{s} decay time (B0 tags)", 50,  0., 12. ) ;
   htbbarJpsiKs = new TH1D( "htbbarJpsiKs", "B0 -> J/#psiK_{s} decay time (B0bar tags)", 50,  0., 12. ) ;
   hmumuMassSqr = new TH1D( "hmumuMassSqr","#mu^{+}#mu^{-} invariant mass squared",  100, -1.0, 25.0) ;
   hmumuMassSqrPlus = new TH1D( "hmumuMassSqrPlus","#mu^{+}#mu^{-} invariant mass squared (cos#theta > 0)",  100, -1.0, 25.0) ;
   hmumuMassSqrMinus = new TH1D( "hmumuMassSqrMinus","#mu^{+}#mu^{-} invariant mass squared (cos#theta < 0)",  100, -1.0, 25.0) ;
   hIdBsDaugs = new TH1D( "hIdBsDaugs","LundIDs of the Bs's daughters",  100, -1000., 1000.) ;
   hIdPhiDaugs = new TH1D( "hIdPhiDaugs","LundIDs of the phi's daughters",  100, -500., 500.) ;
   hIdJpsiMot = new TH1D( "hIdJpsiMot","LundIDs of the J/psi's mother",  100, -500., 500.) ;
   hIdBDaugs = new TH1D( "hIdBDaugs","LundIDs of the B's daughters",  100, -1000., 1000.) ;
   hCosTheta1 = new TH1D( "hCosTheta1","cos#theta_{1}",  50, -1., 1.) ;
   hCosTheta2 = new TH1D( "hCosTheta2","cos#theta_{2}",  50, -1., 1.) ;
   hPhi1 = new TH1D( "hPhi1","#phi_{1}",  50, -3.14, 3.14) ;
   hPhi2 = new TH1D( "hPhi2","#phi_{2}",  50, -3.14, 3.14) ;
   hCosThetaLambda = new TH1D( "hCosThetaLambda","cos#theta_{#Lambda}",  50, -1., 1.) ;

   decayed = new std::ofstream("decayed.txt") ;
   undecayed = new std::ofstream("undecayed.txt") ;
   return ;
}
 
void EvtGenTestAnalyzer::analyze( const Event& e, const EventSetup& )
{
      
   Handle< HepMCProduct > EvtHandle ;
   
   // find initial HepMCProduct by its label - source
   // OR
   // find HepMCProduct after evtgenlhc by its label - evtgenproducer, that is
   // 
   e.getByLabel( theSrc , EvtHandle ) ;
   
   const GenEvent* Evt = EvtHandle->GetEvent() ;
   if (Evt) nevent++;

   const float mmcToPs = 3.3355; 
   int nB = 0;
   int nJpsi = 0;
   int nBp = 0;
   int nBz = 0;
   int nBzb = 0;
   int nBzmix = 0;
   int nBzunmix = 0;
   int nBzKmumu = 0;
   int nBJpsiKs = 0;
   int nBJpsiKstar = 0;

   for ( GenEvent::particle_const_iterator p = Evt->particles_begin(); p != Evt->particles_end(); ++p ) {

     // General
     TLorentzVector thePart4m(0.,0.,0.,0.);
     GenVertex* endvert = (*p)->end_vertex(); 
     GenVertex* prodvert = (*p)->production_vertex();
     float gamma = (*p)->momentum().e()/(*p)->generated_mass();
     float dectime = 0.0 ;
     int mixed = -1;  // mixed is: -1 = unmixed
                      //           0 = mixed (before mixing)
                      //           1 = mixed (after mixing)
     if (endvert && prodvert) {
       dectime = (endvert->position().t() - prodvert->position().t())*mmcToPs/gamma;

       // Mixed particle ?
       for ( GenVertex::particles_in_const_iterator p2 = prodvert->particles_in_const_begin(); p2 != prodvert->particles_in_const_end(); ++p2 ) { 
	 if ( (*p)->pdg_id() + (*p2)->pdg_id() == 0) {
	   mixed = 1;
           gamma = (*p2)->momentum().e()/(*p2)->generated_mass();
           GenVertex* mixvert = (*p2)->production_vertex();
           dectime = (prodvert->position().t() - mixvert->position().t())*mmcToPs/gamma;
	 }
       }
       for ( GenVertex::particles_out_const_iterator ap = endvert->particles_out_const_begin(); ap != endvert->particles_out_const_end(); ++ap ) {
	 if ( (*p)->pdg_id() + (*ap)->pdg_id() == 0) mixed = 0;
       }
     }

     hGeneralId->Fill((*p)->pdg_id()); 

     // --------------------------------------------------------------
     if ( abs((*p)->pdg_id()) == 521 )   // B+/- 
       // || abs((*p)->pdg_id()/100) == 4 || abs((*p)->pdg_id()/100) == 3) 
     {
         nBp++;
	 htbPlus->Fill( dectime );
	 int isJpsiKstar = 0;
	 for ( GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin(); bp != endvert->particles_out_const_end(); ++bp ) {
	   if ( (*bp)->pdg_id() == 443 || abs( (*bp)->pdg_id() ) == 323 ) isJpsiKstar++ ;          
	 }
	 if (isJpsiKstar == 2) nBJpsiKstar++;       
     }
     // if ((*p)->pdg_id() == 443) *undecayed << (*p)->pdg_id() << endl;
     // --------------------------------------------------------------

     if ( (*p)->pdg_id() == 531 /* && endvert */ ) {  // B_s 
       // nbs++;
       hPtbs->Fill((*p)->momentum().perp());
       hPbs->Fill( sqrt ( pow((*p)->momentum().px(),2)+pow((*p)->momentum().py(),2)+
			  pow((*p)->momentum().pz(),2) )) ;
       hPhibs->Fill((*p)->momentum().phi());
       hEtabs->Fill((*p)->momentum().pseudoRapidity());
       
       for ( GenVertex::particles_out_const_iterator ap = endvert->particles_out_const_begin(); ap != endvert->particles_out_const_end(); ++ap ) {
	 hIdBsDaugs->Fill((*ap)->pdg_id());
       }
       
       if (mixed == 1) {
	 htbsMix->Fill( dectime );
       } else if (mixed == -1) {
         htbsUnmix->Fill( dectime );
       }
       
     }
     // --------------------------------------------------------------
     if ( abs((*p)->pdg_id()) == 511 /* && endvert */ ) {  // B0 
       if (mixed != 0) {
	 nB++;
         if ( (*p)->pdg_id() > 0 ) {
	   nBz++;
	   if ( mixed == 1 ) {nBzmix++;} else {nBzunmix++;}
	 } else {nBzb++;}         
       }
       int isJpsiKs = 0;
       int isKmumu = 0;
       int isSemilept = 0;
       for ( GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin(); bp != endvert->particles_out_const_end(); ++bp ) {
         // Check invariant mass consistency ...   
         TLorentzVector theDaug4m((*bp)->momentum().px(), (*bp)->momentum().py(),
				  (*bp)->momentum().pz(), (*bp)->momentum().e());
         thePart4m += theDaug4m;
	 hIdBDaugs->Fill((*bp)->pdg_id());
         if ( (*bp)->pdg_id() == 443 || (*bp)->pdg_id() == 310 ) isJpsiKs++ ; 
         if ( (*bp)->pdg_id() == 22 ) {
	   hPtRadPho->Fill((*bp)->momentum().perp());
	   hPhiRadPho->Fill((*bp)->momentum().phi());
	   hEtaRadPho->Fill((*bp)->momentum().pseudoRapidity());
	 }
         if ( (*p)->pdg_id() > 0 && ( abs((*bp)->pdg_id()) == 313 || abs((*bp)->pdg_id()) == 13 )) isKmumu++ ; 
         if ( abs((*bp)->pdg_id()) == 11 || abs((*bp)->pdg_id()) == 13 || abs((*bp)->pdg_id()) == 15 ) isSemilept++ ;
       }

       hMinvb->Fill(sqrt(thePart4m.M2()));
       /* if (fabs(sqrt(thePart4m.M2())-5.28) > 0.05) {
	 *undecayed << sqrt(thePart4m.M2()) << "  " << (*p)->pdg_id() << " --> " ;
         for ( GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin(); bp != endvert->particles_out_const_end(); ++bp ) {
	   *undecayed << (*bp)->pdg_id() << " " ;
         }
         *undecayed << endl;
       } */

       if (isSemilept) {
	 if (mixed == 1) {
	   htbMix->Fill( dectime );
           if ( (*p)->pdg_id() > 0 ) { 
	     htbMixPlus->Fill( dectime );
	   } else {
             htbMixMinus->Fill( dectime );
	   }
	 } else if (mixed == -1) {
	   htbUnmix->Fill( dectime );
	 }
       }

       if (isJpsiKs == 2) {
         nBJpsiKs++;
	 if ( (*p)->pdg_id()*mixed < 0 ) { 
	   htbbarJpsiKs->Fill( dectime );
	 } else {
           htbJpsiKs->Fill( dectime );
	 }
       }
      
       if (isKmumu == 3) nBzKmumu++; 
     }
     // --------------------------------------------------------------
     if ( (*p)->pdg_id() == 443 ) {       //Jpsi
       nJpsi++;
       for ( GenVertex::particles_in_const_iterator ap = prodvert->particles_in_const_begin(); ap != prodvert->particles_in_const_end(); ++ap ) {
	 hIdJpsiMot->Fill((*ap)->pdg_id());
       }
     }
     // --------------------------------------------------------------
     if ( (*p)->pdg_id() == 333 ) {  // phi 
       if (endvert) {
         for ( GenVertex::particles_out_const_iterator cp = endvert->particles_out_const_begin(); cp != endvert->particles_out_const_end(); ++cp ) {
	   hIdPhiDaugs->Fill((*cp)->pdg_id());
	 }
       }
     }
     // --------------------------------------------------------------
     if ( (*p)->pdg_id() == 13 ) { // mu+
       for ( GenVertex::particles_in_const_iterator p2 = prodvert->particles_in_const_begin(); p2 != prodvert->particles_in_const_end(); ++p2 ) {
	 if ( abs((*p2)->pdg_id()) == 511 ) { // B0
	   hPtmu->Fill((*p)->momentum().perp());
	   hPmu->Fill( sqrt ( pow((*p)->momentum().px(),2)+pow((*p)->momentum().py(),2)+
			      pow((*p)->momentum().pz(),2) )) ;
	   hPhimu->Fill((*p)->momentum().phi());
	   hEtamu->Fill((*p)->momentum().pseudoRapidity());
	   for ( GenVertex::particles_out_const_iterator p3 = prodvert->particles_out_const_begin(); p3 != prodvert->particles_out_const_end(); ++p3 ) {
	     if ( (*p3)->pdg_id() == -13 ) { // mu-
	       TLorentzVector pmu1((*p)->momentum().px(), (*p)->momentum().py(),
				   (*p)->momentum().pz(), (*p)->momentum().e());
	       TLorentzVector pmu2((*p3)->momentum().px(), (*p3)->momentum().py(),
				   (*p3)->momentum().pz(), (*p3)->momentum().e());
               TLorentzVector pb0((*p2)->momentum().px(), (*p2)->momentum().py(),
				  (*p2)->momentum().pz(), (*p2)->momentum().e());
	       TLorentzVector ptot = pmu1 + pmu2;
               TVector3 booster = ptot.BoostVector();
	       TLorentzVector leptdir = ( ((*p2)->pdg_id() > 0) ? pmu1 : pmu2 );

	       leptdir.Boost( -booster );
	       pb0.Boost( -booster );
	       hmumuMassSqr->Fill(ptot.M2());
               if ( cos( leptdir.Vect().Angle(pb0.Vect()) ) > 0) {
		 hmumuMassSqrPlus->Fill(ptot.M2());
	       } else {
		 hmumuMassSqrMinus->Fill(ptot.M2());
	       }
	     }
	   }
	 }
       }
     }
     // --------------------------------------------------------------
     // Calculate helicity angles to test polarization
     // (from Hrivnac et al., J. Phys. G21, 629)

     if ( (*p)->pdg_id() == 5122 /* && endvert */ ) {   // lambdaB

       TLorentzVector pMuP;
       TLorentzVector pProt;
       TLorentzVector pLambda0;
       TLorentzVector pJpsi;
       TLorentzVector pLambdaB;
       TVector3 enne;

       nbs++; 
       if (!endvert) {
	 *undecayed << (*p)->pdg_id() << endl;
       } else {
	 *decayed << (*p)->pdg_id() << " --> " ;
	 for ( GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin(); bp != endvert->particles_out_const_end(); ++bp ) {
	   *decayed << (*bp)->pdg_id() << " " ;
	 }
       }
       *decayed << endl;

       pLambdaB.SetPxPyPzE((*p)->momentum().px(), (*p)->momentum().py(),
			   (*p)->momentum().pz(), (*p)->momentum().e());
       enne = - (pLambdaB.Vect().Cross(TVector3(0.,0.,1.))).Unit();
 
       if (endvert) {
	 for ( GenVertex::particles_out_const_iterator p2 = endvert->particles_out_const_begin(); p2 != endvert->particles_out_const_end(); ++p2 ) {
	   if ((*p2)->pdg_id() == 443) {  // J/psi
	     pJpsi.SetPxPyPzE((*p2)->momentum().px(), (*p2)->momentum().py(),
			      (*p2)->momentum().pz(), (*p2)->momentum().e());
	     GenVertex* psivert = (*p2)->end_vertex();
	     if (psivert) {
	       for ( GenVertex::particles_out_const_iterator p3 = psivert->particles_out_const_begin(); p3 != psivert->particles_out_const_end(); ++p3 ) {
		 if ((*p3)->pdg_id() == -13) {  // mu+
		   pMuP.SetPxPyPzE((*p3)->momentum().px(), (*p3)->momentum().py(),
				   (*p3)->momentum().pz(), (*p3)->momentum().e());
		 }
	       }
	     }
	   }
	   if ((*p2)->pdg_id() == 3122) {   // Lambda0
	     pLambda0.SetPxPyPzE((*p2)->momentum().px(), (*p2)->momentum().py(),
				 (*p2)->momentum().pz(), (*p2)->momentum().e());
	     GenVertex* lamvert = (*p2)->end_vertex();
	     if (lamvert) {
	       for ( GenVertex::particles_out_const_iterator p3 = lamvert->particles_out_const_begin(); p3 != lamvert->particles_out_const_end(); ++p3 ) {
		 if (abs((*p3)->pdg_id()) == 2212) {   // p
		   pProt.SetPxPyPzE((*p3)->momentum().px(), (*p3)->momentum().py(),
				    (*p3)->momentum().pz(), (*p3)->momentum().e());
		 }
	       }
	     }
	   }
	 }
       }
	       
       TVector3 booster1 = pLambdaB.BoostVector();
       TVector3 booster2 = pLambda0.BoostVector();
       TVector3 booster3 = pJpsi.BoostVector();

       pLambda0.Boost( -booster1 );
       pJpsi.Boost( -booster1 );
       hCosThetaLambda->Fill( cos( pLambda0.Vect().Angle(enne) ));
      
       pProt.Boost( -booster2 );
       hCosTheta1->Fill( cos( pProt.Vect().Angle(pLambda0.Vect()) ));
       TVector3 tempY = (pLambda0.Vect().Cross(enne)).Unit();
       TVector3 xyProj = (enne.Dot(pProt.Vect()))*enne + (tempY.Dot(pProt.Vect()))*tempY;
       // find the sign of phi
       TVector3 crossProd = xyProj.Cross(enne);
       float tempPhi = (crossProd.Dot(pLambda0.Vect()) > 0 ? xyProj.Angle(enne) : -xyProj.Angle(enne));
       hPhi1->Fill( tempPhi );

       pMuP.Boost( -booster3 );
       hCosTheta2->Fill( cos( pMuP.Vect().Angle(pJpsi.Vect()) ));
       tempY = (pJpsi.Vect().Cross(enne)).Unit();
       xyProj = (enne.Dot(pMuP.Vect()))*enne + (tempY.Dot(pMuP.Vect()))*tempY;
       // find the sign of phi
       crossProd = xyProj.Cross(enne);
       tempPhi = (crossProd.Dot(pJpsi.Vect()) > 0 ? xyProj.Angle(enne) : -xyProj.Angle(enne));
       hPhi2->Fill( tempPhi );
       
     }
     
   }
   // ---------------------------------------------------------

   hnB->Fill(nB);
   hnJpsi->Fill(nJpsi);
   hnBz->Fill(nBz);
   hnBzb->Fill(nBzb);
   // *undecayed << "-------------------------------" << endl;
   // *decayed << "-------------------------------" << endl;

   // if (nBz > 0) std::cout << "nBz = " << nBz << " nBz (K*mu+mu-) = " << nBzKmumu << " nMix = " << nBzmix << std::endl;
   // if (nBz > 0 && nBzKmumu == 0) Evt->print();
   // if (nB > 0) std::cout << "nB = " << nB << " nBz (JPsi Ks) = " << nBJpsiKs << std::endl;
   // if (nBp > 0) std::cout << "nB = " << nBp << " nBz (JPsi Kstar) = " << nBJpsiKstar << std::endl;
   // if (nBp > 0 && nBJpsiKstar == 0) Evt->print();

   return ;   
}

void EvtGenTestAnalyzer::endJob()
{
  TObjArray Hlist(0);
  Hlist.Add(hGeneralId);	   
  Hlist.Add(hIdPhiDaugs) ;
  Hlist.Add(hIdJpsiMot) ;
  Hlist.Add(hnB);		   
  Hlist.Add(hnBz) ;		   
  Hlist.Add(hnBzb) ;
  Hlist.Add(hnJpsi) ;
  Hlist.Add(hMinvb) ;
  Hlist.Add(hPtbs) ;	   
  Hlist.Add(hPbs) ;		   
  Hlist.Add(hPhibs) ;	   
  Hlist.Add(hEtabs) ;	   
  Hlist.Add(hPtmu) ;	   
  Hlist.Add(hPmu) ;		   
  Hlist.Add(hPhimu) ;	   
  Hlist.Add(hEtamu) ;
  Hlist.Add(hPtRadPho) ;	   		   
  Hlist.Add(hPhiRadPho) ;	   
  Hlist.Add(hEtaRadPho) ;
  Hlist.Add(htbJpsiKs) ;	   
  Hlist.Add(htbbarJpsiKs) ;	   
  Hlist.Add(htbPlus) ;	   
  Hlist.Add(htbsUnmix) ;	   
  Hlist.Add(htbsMix) ;	   
  Hlist.Add(htbUnmix) ;	   
  Hlist.Add(htbMix) ; 	   
  Hlist.Add(htbMixPlus) ;	   
  Hlist.Add(htbMixMinus) ;     
  Hlist.Add(hmumuMassSqr) ;	   
  Hlist.Add(hmumuMassSqrPlus) ;
  Hlist.Add(hmumuMassSqrMinus); 
  Hlist.Add(hIdBsDaugs) ;	   
  Hlist.Add(hIdBDaugs) ;	   
  Hlist.Add(hCosTheta1) ;   
  Hlist.Add(hCosTheta2) ;
  Hlist.Add(hPhi1) ;   
  Hlist.Add(hPhi2) ;
  Hlist.Add(hCosThetaLambda) ;
  Hlist.Write() ;
  fOutputFile->Close() ;
  cout << "N_events = " << nevent << "\n";
  cout << "N_LambdaB = " << nbs << "\n"; 
  return ;
}
 
DEFINE_FWK_MODULE(EvtGenTestAnalyzer);
