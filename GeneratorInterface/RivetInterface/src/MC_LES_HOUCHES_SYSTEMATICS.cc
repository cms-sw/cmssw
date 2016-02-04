// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Projections/MergedFinalState.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"
#include <boost/lexical_cast.hpp>

// ATLAS=0, CMS=1, CDF=2
#define EXPERIMENT 1


namespace Rivet {

#if EXPERIMENT==0
  class MC_LES_HOUCHES_SYSTEMATICS_ATLAS : public Analysis {
#elif EXPERIMENT==1
  class MC_LES_HOUCHES_SYSTEMATICS_CMS : public Analysis {
#elif EXPERIMENT==2
  class MC_LES_HOUCHES_SYSTEMATICS_CDF : public Analysis {
#endif
  public:

    /// Default constructor
#if EXPERIMENT==0
    MC_LES_HOUCHES_SYSTEMATICS_ATLAS() : Analysis("MC_LES_HOUCHES_SYSTEMATICS_ATLAS")
#elif EXPERIMENT==1
    MC_LES_HOUCHES_SYSTEMATICS_CMS() : Analysis("MC_LES_HOUCHES_SYSTEMATICS_CMS")
#elif EXPERIMENT==2
    MC_LES_HOUCHES_SYSTEMATICS_CDF() : Analysis("MC_LES_HOUCHES_SYSTEMATICS_CDF")
#endif
    {
      setNeedsCrossSection(true);
    }


    void init() {

#if EXPERIMENT==0
      VisibleFinalState fs(-5.0, 5.0, 0.*GeV);
      IdentifiedFinalState electrons(-5.0, 5.0, 20.*GeV);
      electrons.acceptIdPair(ELECTRON);
      IdentifiedFinalState muons(-5.0, 5.0, 20.*GeV);
      muons.acceptIdPair(MUON);
#elif EXPERIMENT==1
      VisibleFinalState fs(-3.0, 3.0, 0.*GeV);
      IdentifiedFinalState electrons(-3.0, 3.0, 20.*GeV);
      electrons.acceptIdPair(ELECTRON);
      IdentifiedFinalState muons(-3.0, 3.0, 20.*GeV);
      muons.acceptIdPair(MUON);
#elif EXPERIMENT==2
      VisibleFinalState fs(-3.0, 3.0, 0.*GeV);
      IdentifiedFinalState electrons(-1.0, 1.0, 20.*GeV);
      electrons.acceptIdPair(ELECTRON);
      IdentifiedFinalState muons(-1.0, 1.0, 20.*GeV);
      muons.acceptIdPair(MUON);
#endif
      MergedFinalState leptons(electrons, muons);

      addProjection(fs, "FS");

      VetoedFinalState vfs(fs);
      vfs.addVetoPairDetail(ELECTRON, 20.*GeV, 7000.*GeV);
      vfs.addVetoPairDetail(MUON,     20.*GeV, 7000.*GeV);

      VisibleFinalState missing(-10.0, 10.0, 0.*GeV);

      addProjection(electrons, "ELECTRONS");
      addProjection(muons, "MUONS");
      addProjection(leptons, "LEPTONS");
      addProjection(vfs, "VFS");
      addProjection(missing, "MISSING");

      _h_njets = bookHistogram1D("njet", 7, -0.5, 6.5);            // inclusive jet multiplicity (0..6)
      _h_njetsratio = bookDataPointSet("njetratio", 6, 0.5, 6.5);  // n/(n-1)

      for (int i=0 ; i<6 ; i++) {
        _h_jetpt.push_back(bookHistogram1D("jetpt"+boost::lexical_cast<string>(i), 50, 0, 250));  // jet pT  (1..6)
        _h_jeteta.push_back(bookHistogram1D("jeteta"+boost::lexical_cast<string>(i), 16, -4, 4));  // jet eta (1..6)
      }

      for (int i=0 ; i<7 ; i++) {
        _h_HTjet.push_back(bookHistogram1D("HTjet"+boost::lexical_cast<string>(i), 50, 0, 1000));  // HT from jets
        _h_HTall.push_back(bookHistogram1D("HTall"+boost::lexical_cast<string>(i), 50, 0, 1000));  // HT from jets + lepton + missing
        _h_sumET.push_back(bookHistogram1D("sumET"+boost::lexical_cast<string>(i), 50, 0, 1000));  // sum ET of visible particles
      }

      _h_dEtaj0j1    = bookHistogram1D("dEtaj0j1", 20, 0, 5);     // deltaEta(leading jets)
      _h_dPhij0j1    = bookHistogram1D("dPhij0j1", 20, 0, M_PI);  // deltaPhi(leading jets)
      _h_dRj0j1      = bookHistogram1D("dRj0j1", 20, 0, 5);       // deltaR(leading jets)
      _h_mj0j1       = bookHistogram1D("mj0j1", 60, 0, 300);      // mass(jet0 + jet1)
      _h_ptratioj1j0 = bookHistogram1D("ptratioj1j0", 20, 0, 1);  // pT(jet1)/pT(jet0)

      _h_dEtaj0l = bookHistogram1D("dEtaj0l", 20, 0, 5);          // deltaEta(leading jet, lepton)
      _h_dPhij0l = bookHistogram1D("dPhij0l", 20, 0, M_PI);       // deltaPhi(leading jet, lepton)
      _h_dRj0l   = bookHistogram1D("dRj0l", 20, 0, 5);            // deltaR(leading jet, lepton)
      _h_mj0l    = bookHistogram1D("mj0l", 60, 0, 300);           // mass(jet0 + lepton)

      _h_mj0j1W = bookHistogram1D("mj0j1W", 50, 0, 500);          // mass(jet0 + jet1 + W)

      _h_beamthrustjets      = bookHistogram1D("beamthrustjets", 50, 0, 100);      // Whatever-1.
      _h_beamthrustparticles = bookHistogram1D("beamthrustparticles", 50, 0, 100); // Whatever-2.

      _h_leptonpt  = bookHistogram1D("leptonpt", 50, 0, 200);     // lepton pT
      _h_leptoneta = bookHistogram1D("leptoneta", 12, -3, 3);     // lepton eta

      _h_Wpt   = bookHistogram1D("Wpt", 50, 0, 200);              // W pT
      _h_Weta  = bookHistogram1D("Weta", 20, -5, 5);              // W eta
      _h_Wmass = bookHistogram1D("Wmass", 100, 0, 500);           // W mass
      _h_Wmt   = bookHistogram1D("Wmt", 40, 0, 200);              // W transverse mass

      _h_sigmatot = bookDataPointSet("sigmatot", 1, 0, 1);        // sigma_tot as reported by the generator
      _h_sigmacut = bookHistogram1D("sigmacut", 6, -0.5, 5.5);    // sigma after each cut 
    }


    void analyze(const Event & event) {
      const double weight = event.weight();

      _h_sigmacut->fill(0, weight);

      const FinalState& allleptonsfs = applyProjection<FinalState>(event, "LEPTONS");
      ParticleVector allleptons = allleptonsfs.particlesByPt();
      if (allleptons.size() < 1) vetoEvent;

      _h_sigmacut->fill(1, weight);

      // Isolation cut
      Particle lepton;
      const FinalState& fullfs = applyProjection<FinalState>(event, "MISSING");
      bool found_lepton = false;
      for (size_t i=0 ; i<allleptons.size() ; i++) {
        FourMomentum testmom = allleptons[i].momentum();
#if EXPERIMENT==0
        if (fabs(testmom.eta())>2.5) continue;
#elif EXPERIMENT==1
        if ((abs(allleptons[i].pdgId())==MUON && fabs(testmom.eta())>2.1) ||
            (abs(allleptons[i].pdgId())==ELECTRON && fabs(testmom.eta())>2.5)) continue;
#elif EXPERIMENT==2
        if (fabs(testmom.eta())>1.0) continue;
#endif
        double etsum(-testmom.Et());
        foreach (Particle hit, fullfs.particles()) {
          FourMomentum trackmom = hit.momentum();
          if (deltaR(testmom,trackmom)<0.5) {
            etsum += trackmom.Et();
            if (etsum>0.1*testmom.Et())
              break;
          }
        }
        if (etsum<0.1*testmom.Et()) {
          lepton = allleptons[i];
          allleptons.erase(allleptons.begin()+i);
          found_lepton = true;
          break;
        }
      }
      if (!found_lepton) vetoEvent;
      _h_sigmacut->fill(2, weight);


      // Missing ET cut
      FourMomentum missingmom;
      foreach (Particle hit, fullfs.particles()) {
        missingmom += hit.momentum();
      }
      missingmom *= -1; // missing is "minus visible"
      missingmom.setE(missingmom.vector3().mod()); // assume neutrinos are massless
#if EXPERIMENT==0
      if (missingmom.Et()<25.*GeV) vetoEvent;
#elif EXPERIMENT==2
      if (missingmom.Et()<25.*GeV) vetoEvent;
#endif
      _h_sigmacut->fill(3, weight);



      // Create a W
      FourMomentum Wmom = missingmom + lepton.momentum();



      // Transverse mass cut
      double mT2 = pow(lepton.momentum().pT()+missingmom.pT(),2)-Wmom.pT2();
#if EXPERIMENT==0
      if (sqrt(mT2) < 40.*GeV) vetoEvent;
#elif EXPERIMENT==1
      if (sqrt(mT2) < 20.*GeV) vetoEvent;
#elif EXPERIMENT==2
      if (sqrt(mT2) < 30.*GeV) vetoEvent;
#endif
      _h_sigmacut->fill(4, weight);


      // Reconstruct jets
      const FinalState& vfs = applyProjection<FinalState>(event, "VFS");
#if EXPERIMENT==0
      FastJets jetsproj(vfs, FastJets::ANTIKT, 0.4);
      jetsproj.calc(vfs.particles()+allleptons);
      Jets alljets = jetsproj.jetsByPt(25.*GeV);
#elif EXPERIMENT==1
      FastJets jetsproj(vfs, FastJets::ANTIKT, 0.5);
      jetsproj.calc(vfs.particles()+allleptons);
      Jets alljets = jetsproj.jetsByPt(30.*GeV);
#elif EXPERIMENT==2
      FastJets jetsproj(vfs, FastJets::CDFJETCLU, 0.4);
      jetsproj.calc(vfs.particles()+allleptons);
      Jets alljets = jetsproj.jetsByPt(30.*GeV);
#endif
      Jets jets;
      foreach (Jet jet, alljets) {
#if EXPERIMENT==0
        if (fabs(jet.momentum().eta())<4.4)
#elif EXPERIMENT==1
        if (fabs(jet.momentum().eta())<2.4)
#elif EXPERIMENT==2
        if (fabs(jet.momentum().eta())<2.4)
#endif
          jets.push_back(jet);
      }


#if EXPERIMENT==2
      if (jets.size()<2 || (jets[0].momentum()+jets[1].momentum()).pT()<40.*GeV) vetoEvent;
#endif
      _h_sigmacut->fill(5, weight);


      // Fill the histograms
      _h_Wpt->fill(Wmom.pT(),weight);
      _h_Weta->fill(Wmom.eta(),weight);
      _h_Wmass->fill(Wmom.mass(),weight);
      _h_Wmt->fill(sqrt(mT2),weight);

      _h_leptonpt->fill(lepton.momentum().pT(),weight);
      _h_leptoneta->fill(lepton.momentum().eta(),weight);

      double HTjet = 0.;
      double HTall = 0.;
      double sumET = 0.;
      double beamthrustjets = 0.;
      double beamthrustparticles = 0.;
      foreach (Jet jet, jets) {
        HTjet += jet.momentum().Et();
        HTall += jet.momentum().Et();
        beamthrustjets += jet.momentum().E() - fabs(jet.momentum().z());
      }
      HTall += Wmom.Et();

      foreach (Particle p, vfs.particles()+allleptons) {
        sumET += p.momentum().Et();
        beamthrustparticles += p.momentum().E() - fabs(p.momentum().z());
      }

      _h_beamthrustjets->fill(beamthrustjets, weight);
      _h_beamthrustparticles->fill(beamthrustparticles, weight);
      _h_njets->fill(0, weight);  // I guess we always have at least 0 jets
      _h_HTjet[0]->fill(HTjet, weight);
      _h_HTall[0]->fill(HTall, weight);
      _h_sumET[0]->fill(sumET, weight);
      for (size_t i=0 ; i<jets.size() ; i++) {
        if (i==6) break;
        _h_njets->fill(i+1, weight);  // njets is inclusive
        _h_jetpt[i]->fill(jets[i].momentum().pT(), weight);
        _h_jeteta[i]->fill(jets[i].momentum().eta(), weight);
        _h_HTjet[i+1]->fill(HTjet, weight);
        _h_HTall[i+1]->fill(HTall, weight);
        _h_sumET[i+1]->fill(sumET, weight);
      }
      if (jets.size() >= 1) {
        _h_dEtaj0l->fill(deltaEta(jets[0],lepton), weight);
        _h_dPhij0l->fill(deltaPhi(jets[0],lepton), weight);
        _h_dRj0l->fill(deltaR(jets[0],lepton), weight);
        _h_mj0l->fill((jets[0].momentum() + lepton.momentum()).mass(), weight);
      }

      if (jets.size() >= 2) {
        _h_dEtaj0j1->fill(deltaEta(jets[0],jets[1]), weight);
        _h_dPhij0j1->fill(deltaPhi(jets[0],jets[1]), weight);
        _h_dRj0j1->fill(deltaR(jets[0],jets[1]), weight);
        _h_mj0j1->fill((jets[0].momentum() + jets[1].momentum()).mass(), weight);
        _h_mj0j1W->fill((jets[0].momentum() + jets[1].momentum() + Wmom).mass(), weight);
        _h_ptratioj1j0->fill(jets[1].momentum().pT()/jets[0].momentum().pT(), weight);
      }

    }

    /// Finalize
    void finalize() {
      AIDA::IHistogramFactory& hf = histogramFactory();
      const string dir = histoDir();

      for (int i=0 ; i<6 ; i++) {
        hf.divide(dir + "/HTjet" + boost::lexical_cast<string>(i+1) + "over" + boost::lexical_cast<string>(i), *_h_HTjet[i+1], *_h_HTjet[i]);
        hf.divide(dir + "/HTall" + boost::lexical_cast<string>(i+1) + "over" + boost::lexical_cast<string>(i), *_h_HTall[i+1], *_h_HTall[i]);
        hf.divide(dir + "/sumET" + boost::lexical_cast<string>(i+1) + "over" + boost::lexical_cast<string>(i), *_h_sumET[i+1], *_h_sumET[i]);
      }

      std::vector<double> y, yerr;
      for (int i=0; i<_h_njets->axis().bins()-1; i++) {
        double val = 0.;
        double err = 0.;
        if (!fuzzyEquals(_h_njets->binHeight(i), 0)) {
          val = _h_njets->binHeight(i+1) / _h_njets->binHeight(i);
          err = val * sqrt(  pow(_h_njets->binError(i+1)/_h_njets->binHeight(i+1), 2)
                           + pow(_h_njets->binError(i)  /_h_njets->binHeight(i)  , 2) );
        }
        y.push_back(val);
        yerr.push_back(err);
      }
      _h_njetsratio->setCoordinate(1, y, yerr);

      std::vector<double> sigma, sigmaerr;
      sigma.push_back(crossSection());
      sigmaerr.push_back(0);
      _h_sigmatot->setCoordinate(1, sigma, sigmaerr);

      scale(_h_njets, crossSection()/sumOfWeights());
      scale(_h_dEtaj0j1, crossSection()/sumOfWeights());
      scale(_h_dPhij0j1, crossSection()/sumOfWeights());
      scale(_h_dRj0j1, crossSection()/sumOfWeights());
      scale(_h_mj0j1, crossSection()/sumOfWeights());
      scale(_h_ptratioj1j0, crossSection()/sumOfWeights());
      scale(_h_dEtaj0l, crossSection()/sumOfWeights());
      scale(_h_dPhij0l, crossSection()/sumOfWeights());
      scale(_h_dRj0l, crossSection()/sumOfWeights());
      scale(_h_mj0l, crossSection()/sumOfWeights());
      scale(_h_mj0j1W, crossSection()/sumOfWeights());
      scale(_h_beamthrustjets, crossSection()/sumOfWeights());
      scale(_h_beamthrustparticles, crossSection()/sumOfWeights());
      scale(_h_leptonpt, crossSection()/sumOfWeights());
      scale(_h_leptoneta, crossSection()/sumOfWeights());
      scale(_h_Wpt, crossSection()/sumOfWeights());
      scale(_h_Weta, crossSection()/sumOfWeights());
      scale(_h_Wmass, crossSection()/sumOfWeights());
      scale(_h_Wmt, crossSection()/sumOfWeights());
      scale(_h_sigmacut, crossSection()/sumOfWeights());

      for (int i=0 ; i<6 ; i++) {
        scale(_h_jetpt[i], crossSection()/sumOfWeights());
        scale(_h_jeteta[i], crossSection()/sumOfWeights());
      }

      for (int i=0 ; i<7 ; i++) {
        scale(_h_HTjet[i], crossSection()/sumOfWeights());
        scale(_h_HTall[i], crossSection()/sumOfWeights());
        scale(_h_sumET[i], crossSection()/sumOfWeights());
      }
    }

  private:

    AIDA::IHistogram1D * _h_njets;               // inclusive jet multiplicity (0..6)
    AIDA::IDataPointSet* _h_njetsratio;          // n/(n-1)

    std::vector<AIDA::IHistogram1D*> _h_jetpt;   // jet pT  (1..6)
    std::vector<AIDA::IHistogram1D*> _h_jeteta;  // jet eta (1..6)

    std::vector<AIDA::IHistogram1D*> _h_HTjet;   // HT from jets
    std::vector<AIDA::IHistogram1D*> _h_HTall;   // HT from jets + lepton + missing
    std::vector<AIDA::IHistogram1D*> _h_sumET;   // sum ET of visible particles

    AIDA::IHistogram1D * _h_dEtaj0j1;            // deltaEta(leading jets)
    AIDA::IHistogram1D * _h_dPhij0j1;            // deltaPhi(leading jets)
    AIDA::IHistogram1D * _h_dRj0j1;              // deltaR(leading jets)
    AIDA::IHistogram1D * _h_mj0j1;               // mass(jet0 + jet1)
    AIDA::IHistogram1D * _h_ptratioj1j0;         // pT(jet1)/pT(jet0)

    AIDA::IHistogram1D * _h_dEtaj0l;             // deltaEta(leading jet, lepton)
    AIDA::IHistogram1D * _h_dPhij0l;             // deltaPhi(leading jet, lepton)
    AIDA::IHistogram1D * _h_dRj0l;               // deltaR(leading jet, lepton)
    AIDA::IHistogram1D * _h_mj0l;                // mass(jet0 + lepton)

    AIDA::IHistogram1D * _h_mj0j1W;              // mass(jet0 + jet1 + W)

    AIDA::IHistogram1D * _h_beamthrustjets;      // Whatever-1.
    AIDA::IHistogram1D * _h_beamthrustparticles; // Whatever-2.

    AIDA::IHistogram1D * _h_leptonpt;            // lepton pT
    AIDA::IHistogram1D * _h_leptoneta;           // lepton eta

    AIDA::IHistogram1D * _h_Wpt;                 // W pT
    AIDA::IHistogram1D * _h_Weta;                // W eta
    AIDA::IHistogram1D * _h_Wmass;               // W mass
    AIDA::IHistogram1D * _h_Wmt;                 // W tranverse mass

    AIDA::IDataPointSet* _h_sigmatot;            // sigma_tot as reported by the generator
    AIDA::IHistogram1D * _h_sigmacut;            // sigma after each cut
  };

  // This global object acts as a hook for the plugin system
#if EXPERIMENT==0
  AnalysisBuilder<MC_LES_HOUCHES_SYSTEMATICS_ATLAS> plugin_MC_LES_HOUCHES_SYSTEMATICS_ATLAS;
#elif EXPERIMENT==1
  AnalysisBuilder<MC_LES_HOUCHES_SYSTEMATICS_CMS> plugin_MC_LES_HOUCHES_SYSTEMATICS_CMS;
#elif EXPERIMENT==2
  AnalysisBuilder<MC_LES_HOUCHES_SYSTEMATICS_CDF> plugin_MC_LES_HOUCHES_SYSTEMATICS_CDF;
#endif
}

