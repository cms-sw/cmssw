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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "GeneratorInterface/ExternalDecays/test/EvtGenTestAnalyzer.h"

EvtGenTestAnalyzer::EvtGenTestAnalyzer(const edm::ParameterSet& pset)
    : fOutputFileName(pset.getUntrackedParameter<std::string>("HistOutFile", std::string("TestBs.root"))),
      tokenHepMC_(consumes<edm::HepMCProduct>(
          edm::InputTag(pset.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      fOutputFile(0) {
  usesResource(TFileService::kSharedResource);
}

void EvtGenTestAnalyzer::beginJob() {
  nevent = 0;
  nbs = 0;

  edm::Service<TFileService> fs;
  // fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
  // fHist2muMass  = new TH1D(  "Hist2muMass", "2-mu inv. mass", 100,  60., 120. ) ;
  hGeneralId = fs->make<TH1D>("hGeneralId", "LundIDs of all particles", 100, -1000., 1000.);
  hnB = fs->make<TH1D>("hnB", "N(B)", 10, 0., 10.);
  hnJpsi = fs->make<TH1D>("hnJpsi", "N(Jpsi)", 10, 0., 10.);
  hnBz = fs->make<TH1D>("hnBz", "N(B0)", 10, 0., 10.);
  hnBzb = fs->make<TH1D>("hnBzb", "N(B0bar)", 10, 0., 10.);
  hMinvb = fs->make<TH1D>("hMinvb", "B invariant mass", 100, 5.0, 6.0);
  hPtbs = fs->make<TH1D>("hPtbs", "Pt Bs", 100, 0., 50.);
  hPbs = fs->make<TH1D>("hPbs", "P Bs", 100, 0., 200.);
  hPhibs = fs->make<TH1D>("hPhibs", "Phi Bs", 100, -3.14, 3.14);
  hEtabs = fs->make<TH1D>("hEtabs", "Eta Bs", 100, -7.0, 7.0);
  hPtmu = fs->make<TH1D>("hPtmu", "Pt Mu", 100, 0., 50.);
  hPmu = fs->make<TH1D>("hPmu", "P Mu", 100, 0., 200.);
  hPhimu = fs->make<TH1D>("hPhimu", "Phi Mu", 100, -3.14, 3.14);
  hEtamu = fs->make<TH1D>("hEtamu", "Eta Mu", 100, -7.0, 7.0);
  hPtRadPho = fs->make<TH1D>("hPtRadPho", "Pt radiated photon", 100, 0., 200.);
  hPhiRadPho = fs->make<TH1D>("hPhiRadPho", "Phi radiated photon", 100, -3.14, 3.14);
  hEtaRadPho = fs->make<TH1D>("hEtaRadPho", "Eta radiated photon", 100, -7.0, 7.0);
  htbPlus = fs->make<TH1D>("htbPlus", "B+ proper decay time", 50, 0., 12.);
  htbUnmix = fs->make<TH1D>("htbUnmix", "B0 proper decay time (unmixed)", 50, 0., 12.);
  htbMix = fs->make<TH1D>("htbMix", "B0 proper decay time (mixed)", 50, 0., 12.);
  htbMixPlus = fs->make<TH1D>("htbMixPlus", "B0 proper decay time (mixed positive)", 50, 0., 12.);
  htbMixMinus = fs->make<TH1D>("htbMixMinus", "B0 proper decay time (mixed negative)", 50, 0., 12.);
  htbsUnmix = fs->make<TH1D>("htbsUnmix", "Bs proper decay time (unmixed)", 50, 0., 12.);
  htbsMix = fs->make<TH1D>("htbsMix", "Bs proper decay time (mixed)", 50, 0., 12.);
  htbJpsiKs = fs->make<TH1D>("htbJpsiKs", "B0 -> J/#psiK_{s} decay time (B0 tags)", 50, 0., 12.);
  htbbarJpsiKs = fs->make<TH1D>("htbbarJpsiKs", "B0 -> J/#psiK_{s} decay time (B0bar tags)", 50, 0., 12.);
  hmumuMassSqr = fs->make<TH1D>("hmumuMassSqr", "#mu^{+}#mu^{-} invariant mass squared", 100, -1.0, 25.0);
  hmumuMassSqrPlus =
      fs->make<TH1D>("hmumuMassSqrPlus", "#mu^{+}#mu^{-} invariant mass squared (cos#theta > 0)", 100, -1.0, 25.0);
  hmumuMassSqrMinus =
      fs->make<TH1D>("hmumuMassSqrMinus", "#mu^{+}#mu^{-} invariant mass squared (cos#theta < 0)", 100, -1.0, 25.0);
  hIdBsDaugs = fs->make<TH1D>("hIdBsDaugs", "LundIDs of the Bs's daughters", 100, -1000., 1000.);
  hIdPhiDaugs = fs->make<TH1D>("hIdPhiDaugs", "LundIDs of the phi's daughters", 100, -500., 500.);
  hIdJpsiMot = fs->make<TH1D>("hIdJpsiMot", "LundIDs of the J/psi's mother", 100, -500., 500.);
  hIdBDaugs = fs->make<TH1D>("hIdBDaugs", "LundIDs of the B's daughters", 100, -1000., 1000.);
  hCosTheta1 = fs->make<TH1D>("hCosTheta1", "cos#theta_{1}", 50, -1., 1.);
  hCosTheta2 = fs->make<TH1D>("hCosTheta2", "cos#theta_{2}", 50, -1., 1.);
  hPhi1 = fs->make<TH1D>("hPhi1", "#phi_{1}", 50, -3.14, 3.14);
  hPhi2 = fs->make<TH1D>("hPhi2", "#phi_{2}", 50, -3.14, 3.14);
  hCosThetaLambda = fs->make<TH1D>("hCosThetaLambda", "cos#theta_{#Lambda}", 50, -1., 1.);

  decayed = new std::ofstream("decayed.txt");
  undecayed = new std::ofstream("undecayed.txt");
  return;
}

void EvtGenTestAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  const edm::Handle<edm::HepMCProduct>& EvtHandle = e.getHandle(tokenHepMC_);

  const HepMC::GenEvent* Evt = EvtHandle->GetEvent();
  if (Evt)
    nevent++;

  const float mmcToPs = 3.3355;
  int nB = 0;
  int nJpsi = 0;
  int nBz = 0;
  int nBzb = 0;

  for (HepMC::GenEvent::particle_const_iterator p = Evt->particles_begin(); p != Evt->particles_end(); ++p) {
    // General
    TLorentzVector thePart4m(0., 0., 0., 0.);
    HepMC::GenVertex* endvert = (*p)->end_vertex();
    HepMC::GenVertex* prodvert = (*p)->production_vertex();
    float gamma = (*p)->momentum().e() / (*p)->generated_mass();
    float dectime = 0.0;
    int mixed = -1;  // mixed is: -1 = unmixed
                     //           0 = mixed (before mixing)
                     //           1 = mixed (after mixing)
    if (endvert && prodvert) {
      dectime = (endvert->position().t() - prodvert->position().t()) * mmcToPs / gamma;

      // Mixed particle ?
      for (HepMC::GenVertex::particles_in_const_iterator p2 = prodvert->particles_in_const_begin();
           p2 != prodvert->particles_in_const_end();
           ++p2) {
        if ((*p)->pdg_id() + (*p2)->pdg_id() == 0) {
          mixed = 1;
          gamma = (*p2)->momentum().e() / (*p2)->generated_mass();
          HepMC::GenVertex* mixvert = (*p2)->production_vertex();
          dectime = (prodvert->position().t() - mixvert->position().t()) * mmcToPs / gamma;
        }
      }
      for (HepMC::GenVertex::particles_out_const_iterator ap = endvert->particles_out_const_begin();
           ap != endvert->particles_out_const_end();
           ++ap) {
        if ((*p)->pdg_id() + (*ap)->pdg_id() == 0)
          mixed = 0;
      }
    }

    hGeneralId->Fill((*p)->pdg_id());

    // --------------------------------------------------------------
    if (std::abs((*p)->pdg_id()) == 521)  // B+/-
    // || abs((*p)->pdg_id()/100) == 4 || abs((*p)->pdg_id()/100) == 3)
    {
      htbPlus->Fill(dectime);
    }
    // if ((*p)->pdg_id() == 443) *undecayed << (*p)->pdg_id() << std::endl;
    // --------------------------------------------------------------

    if ((*p)->pdg_id() == 531 /* && endvert */) {  // B_s
      // nbs++;
      hPtbs->Fill((*p)->momentum().perp());
      hPbs->Fill(sqrt(pow((*p)->momentum().px(), 2) + pow((*p)->momentum().py(), 2) + pow((*p)->momentum().pz(), 2)));
      hPhibs->Fill((*p)->momentum().phi());
      hEtabs->Fill((*p)->momentum().pseudoRapidity());

      for (HepMC::GenVertex::particles_out_const_iterator ap = endvert->particles_out_const_begin();
           ap != endvert->particles_out_const_end();
           ++ap) {
        hIdBsDaugs->Fill((*ap)->pdg_id());
      }

      if (mixed == 1) {
        htbsMix->Fill(dectime);
      } else if (mixed == -1) {
        htbsUnmix->Fill(dectime);
      }
    }
    // --------------------------------------------------------------
    if (std::abs((*p)->pdg_id()) == 511 /* && endvert */) {  // B0
      if (mixed != 0) {
        nB++;
        if ((*p)->pdg_id() > 0) {
          nBz++;
        } else {
          nBzb++;
        }
      }
      int isJpsiKs = 0;
      int isSemilept = 0;
      for (HepMC::GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin();
           bp != endvert->particles_out_const_end();
           ++bp) {
        // Check invariant mass consistency ...
        TLorentzVector theDaug4m(
            (*bp)->momentum().px(), (*bp)->momentum().py(), (*bp)->momentum().pz(), (*bp)->momentum().e());
        thePart4m += theDaug4m;
        hIdBDaugs->Fill((*bp)->pdg_id());
        if ((*bp)->pdg_id() == 443 || (*bp)->pdg_id() == 310)
          isJpsiKs++;
        if ((*bp)->pdg_id() == 22) {
          hPtRadPho->Fill((*bp)->momentum().perp());
          hPhiRadPho->Fill((*bp)->momentum().phi());
          hEtaRadPho->Fill((*bp)->momentum().pseudoRapidity());
        }
        if (std::abs((*bp)->pdg_id()) == 11 || std::abs((*bp)->pdg_id()) == 13 || std::abs((*bp)->pdg_id()) == 15)
          isSemilept++;
      }

      hMinvb->Fill(sqrt(thePart4m.M2()));
      /* if (fabs(sqrt(thePart4m.M2())-5.28) > 0.05) {
	 *undecayed << sqrt(thePart4m.M2()) << "  " << (*p)->pdg_id() << " --> " ;
         for ( GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin(); bp != endvert->particles_out_const_end(); ++bp ) {
	   *undecayed << (*bp)->pdg_id() << " " ;
         }
         *undecayed << std::endl;
       } */

      if (isSemilept) {
        if (mixed == 1) {
          htbMix->Fill(dectime);
          if ((*p)->pdg_id() > 0) {
            htbMixPlus->Fill(dectime);
          } else {
            htbMixMinus->Fill(dectime);
          }
        } else if (mixed == -1) {
          htbUnmix->Fill(dectime);
        }
      }

      if (isJpsiKs == 2) {
        if ((*p)->pdg_id() * mixed < 0) {
          htbbarJpsiKs->Fill(dectime);
        } else {
          htbJpsiKs->Fill(dectime);
        }
      }

    }
    // --------------------------------------------------------------
    if ((*p)->pdg_id() == 443) {  //Jpsi
      nJpsi++;
      for (HepMC::GenVertex::particles_in_const_iterator ap = prodvert->particles_in_const_begin();
           ap != prodvert->particles_in_const_end();
           ++ap) {
        hIdJpsiMot->Fill((*ap)->pdg_id());
      }
    }
    // --------------------------------------------------------------
    if ((*p)->pdg_id() == 333) {  // phi
      if (endvert) {
        for (HepMC::GenVertex::particles_out_const_iterator cp = endvert->particles_out_const_begin();
             cp != endvert->particles_out_const_end();
             ++cp) {
          hIdPhiDaugs->Fill((*cp)->pdg_id());
        }
      }
    }
    // --------------------------------------------------------------
    if ((*p)->pdg_id() == 13) {  // mu+
      for (HepMC::GenVertex::particles_in_const_iterator p2 = prodvert->particles_in_const_begin();
           p2 != prodvert->particles_in_const_end();
           ++p2) {
        if (std::abs((*p2)->pdg_id()) == 511) {  // B0
          hPtmu->Fill((*p)->momentum().perp());
          hPmu->Fill(
              sqrt(pow((*p)->momentum().px(), 2) + pow((*p)->momentum().py(), 2) + pow((*p)->momentum().pz(), 2)));
          hPhimu->Fill((*p)->momentum().phi());
          hEtamu->Fill((*p)->momentum().pseudoRapidity());
          for (HepMC::GenVertex::particles_out_const_iterator p3 = prodvert->particles_out_const_begin();
               p3 != prodvert->particles_out_const_end();
               ++p3) {
            if ((*p3)->pdg_id() == -13) {  // mu-
              TLorentzVector pmu1(
                  (*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
              TLorentzVector pmu2(
                  (*p3)->momentum().px(), (*p3)->momentum().py(), (*p3)->momentum().pz(), (*p3)->momentum().e());
              TLorentzVector pb0(
                  (*p2)->momentum().px(), (*p2)->momentum().py(), (*p2)->momentum().pz(), (*p2)->momentum().e());
              TLorentzVector ptot = pmu1 + pmu2;
              TVector3 booster = ptot.BoostVector();
              TLorentzVector leptdir = (((*p2)->pdg_id() > 0) ? pmu1 : pmu2);

              leptdir.Boost(-booster);
              pb0.Boost(-booster);
              hmumuMassSqr->Fill(ptot.M2());
              if (cos(leptdir.Vect().Angle(pb0.Vect())) > 0) {
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

    if ((*p)->pdg_id() == 5122 /* && endvert */) {  // lambdaB

      TLorentzVector pMuP;
      TLorentzVector pProt;
      TLorentzVector pLambda0;
      TLorentzVector pJpsi;
      TLorentzVector pLambdaB;
      TVector3 enne;

      nbs++;
      if (!endvert) {
        *undecayed << (*p)->pdg_id() << std::endl;
      } else {
        *decayed << (*p)->pdg_id() << " --> ";
        for (HepMC::GenVertex::particles_out_const_iterator bp = endvert->particles_out_const_begin();
             bp != endvert->particles_out_const_end();
             ++bp) {
          *decayed << (*bp)->pdg_id() << " ";
        }
      }
      *decayed << std::endl;

      pLambdaB.SetPxPyPzE((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      enne = -(pLambdaB.Vect().Cross(TVector3(0., 0., 1.))).Unit();

      if (endvert) {
        for (HepMC::GenVertex::particles_out_const_iterator p2 = endvert->particles_out_const_begin();
             p2 != endvert->particles_out_const_end();
             ++p2) {
          if ((*p2)->pdg_id() == 443) {  // J/psi
            pJpsi.SetPxPyPzE(
                (*p2)->momentum().px(), (*p2)->momentum().py(), (*p2)->momentum().pz(), (*p2)->momentum().e());
            HepMC::GenVertex* psivert = (*p2)->end_vertex();
            if (psivert) {
              for (HepMC::GenVertex::particles_out_const_iterator p3 = psivert->particles_out_const_begin();
                   p3 != psivert->particles_out_const_end();
                   ++p3) {
                if ((*p3)->pdg_id() == -13) {  // mu+
                  pMuP.SetPxPyPzE(
                      (*p3)->momentum().px(), (*p3)->momentum().py(), (*p3)->momentum().pz(), (*p3)->momentum().e());
                }
              }
            }
          }
          if ((*p2)->pdg_id() == 3122) {  // Lambda0
            pLambda0.SetPxPyPzE(
                (*p2)->momentum().px(), (*p2)->momentum().py(), (*p2)->momentum().pz(), (*p2)->momentum().e());
            HepMC::GenVertex* lamvert = (*p2)->end_vertex();
            if (lamvert) {
              for (HepMC::GenVertex::particles_out_const_iterator p3 = lamvert->particles_out_const_begin();
                   p3 != lamvert->particles_out_const_end();
                   ++p3) {
                if (std::abs((*p3)->pdg_id()) == 2212) {  // p
                  pProt.SetPxPyPzE(
                      (*p3)->momentum().px(), (*p3)->momentum().py(), (*p3)->momentum().pz(), (*p3)->momentum().e());
                }
              }
            }
          }
        }
      }

      TVector3 booster1 = pLambdaB.BoostVector();
      TVector3 booster2 = pLambda0.BoostVector();
      TVector3 booster3 = pJpsi.BoostVector();

      pLambda0.Boost(-booster1);
      pJpsi.Boost(-booster1);
      hCosThetaLambda->Fill(cos(pLambda0.Vect().Angle(enne)));

      pProt.Boost(-booster2);
      hCosTheta1->Fill(cos(pProt.Vect().Angle(pLambda0.Vect())));
      TVector3 tempY = (pLambda0.Vect().Cross(enne)).Unit();
      TVector3 xyProj = (enne.Dot(pProt.Vect())) * enne + (tempY.Dot(pProt.Vect())) * tempY;
      // find the sign of phi
      TVector3 crossProd = xyProj.Cross(enne);
      float tempPhi = (crossProd.Dot(pLambda0.Vect()) > 0 ? xyProj.Angle(enne) : -xyProj.Angle(enne));
      hPhi1->Fill(tempPhi);

      pMuP.Boost(-booster3);
      hCosTheta2->Fill(cos(pMuP.Vect().Angle(pJpsi.Vect())));
      tempY = (pJpsi.Vect().Cross(enne)).Unit();
      xyProj = (enne.Dot(pMuP.Vect())) * enne + (tempY.Dot(pMuP.Vect())) * tempY;
      // find the sign of phi
      crossProd = xyProj.Cross(enne);
      tempPhi = (crossProd.Dot(pJpsi.Vect()) > 0 ? xyProj.Angle(enne) : -xyProj.Angle(enne));
      hPhi2->Fill(tempPhi);
    }
  }
  // ---------------------------------------------------------

  hnB->Fill(nB);
  hnJpsi->Fill(nJpsi);
  hnBz->Fill(nBz);
  hnBzb->Fill(nBzb);
  // *undecayed << "-------------------------------" << std::endl;
  // *decayed << "-------------------------------" << std::endl;

  return;
}

void EvtGenTestAnalyzer::endJob() {
  /*
  TObjArray Hlist(0);
  Hlist.Add(hGeneralId);
  Hlist.Add(hIdPhiDaugs);
  Hlist.Add(hIdJpsiMot);
  Hlist.Add(hnB);
  Hlist.Add(hnBz);
  Hlist.Add(hnBzb);
  Hlist.Add(hnJpsi);
  Hlist.Add(hMinvb);
  Hlist.Add(hPtbs);
  Hlist.Add(hPbs);
  Hlist.Add(hPhibs);
  Hlist.Add(hEtabs);
  Hlist.Add(hPtmu);
  Hlist.Add(hPmu);
  Hlist.Add(hPhimu);
  Hlist.Add(hEtamu);
  Hlist.Add(hPtRadPho);
  Hlist.Add(hPhiRadPho);
  Hlist.Add(hEtaRadPho);
  Hlist.Add(htbJpsiKs);
  Hlist.Add(htbbarJpsiKs);
  Hlist.Add(htbPlus);
  Hlist.Add(htbsUnmix);
  Hlist.Add(htbsMix);
  Hlist.Add(htbUnmix);
  Hlist.Add(htbMix);
  Hlist.Add(htbMixPlus);
  Hlist.Add(htbMixMinus);
  Hlist.Add(hmumuMassSqr);
  Hlist.Add(hmumuMassSqrPlus);
  Hlist.Add(hmumuMassSqrMinus);
  Hlist.Add(hIdBsDaugs);
  Hlist.Add(hIdBDaugs);
  Hlist.Add(hCosTheta1);
  Hlist.Add(hCosTheta2);
  Hlist.Add(hPhi1);
  Hlist.Add(hPhi2);
  Hlist.Add(hCosThetaLambda);
  Hlist.Write();
  fOutputFile->Close();
  */
  std::cout << "N_events = " << nevent << "\n";
  std::cout << "N_LambdaB = " << nbs << "\n";
  return;
}

DEFINE_FWK_MODULE(EvtGenTestAnalyzer);
