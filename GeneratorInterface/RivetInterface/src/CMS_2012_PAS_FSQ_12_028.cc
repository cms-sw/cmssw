// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/LeptonClusters.hh"
#include "Rivet/Projections/LeadingParticlesFinalState.hh"
#include "Rivet/Projections/InvMassFinalState.hh"
/// @todo Include more projections as required, e.g. ChargedFinalState, FastJets, ZFinder...

namespace Rivet {


  class CMS_2012_PAS_FSQ_12_028 : public Analysis {
  public:

    /// Constructor
    CMS_2012_PAS_FSQ_12_028() : Analysis("CMS_2012_PAS_FSQ_12_028") {}


    /// Book histograms and initialise projections before the run
    void init() {
 
       const FinalState fs(-MAXRAPIDITY,MAXRAPIDITY);
       addProjection(fs, "FS");
    
       vector<pair<PdgId,PdgId> > vidsW;
       vidsW.push_back(make_pair(MUON, NU_MUBAR));
       vidsW.push_back(make_pair(ANTIMUON, NU_MU));

       FinalState fsW(-MAXRAPIDITY,MAXRAPIDITY);
       InvMassFinalState invfsW(fsW, vidsW, 20*GeV, 99999*GeV);
       addProjection(invfsW, "INVFSW");

       VetoedFinalState vfs(fs);
       vfs.addVetoOnThisFinalState(invfsW);
       addProjection(vfs, "VFS");
       addProjection(FastJets(vfs, FastJets::ANTIKT, 0.5), "Jets");


       _h_deltaS_eq2jet_Norm = bookHistogram1D(1,1,1);
       _h_deltaS_ge2jet_BalPt_Norm = bookHistogram1D(2,1,1);
       _h_deltaS_ge2jet_Norm = bookHistogram1D(3,1,1);
       _h_deltaPhi_eq2jet_Norm = bookHistogram1D(4,1,1);
       _h_deltaPhi_ge2jet_BalPt_Norm = bookHistogram1D(5,1,1);
       _h_deltaPhi_ge2jet_Norm = bookHistogram1D(6,1,1);
       _h_rel_deltaPt_eq2jet_Norm = bookHistogram1D(7,1,1);
       _h_rel_deltaPt_ge2jet_BalPt_Norm = bookHistogram1D(8,1,1);
       _h_rel_deltaPt_ge2jet_Norm = bookHistogram1D(9,1,1);

       _h_deltaS_eq2jet = bookHistogram1D(10,1,1);
       _h_deltaS_ge2jet_BalPt = bookHistogram1D(11,1,1);
       _h_deltaS_ge2jet = bookHistogram1D(12,1,1);
       _h_deltaPhi_eq2jet = bookHistogram1D(13,1,1);
       _h_deltaPhi_ge2jet_BalPt = bookHistogram1D(14,1,1);
       _h_deltaPhi_ge2jet = bookHistogram1D(15,1,1);
       _h_rel_deltaPt_eq2jet = bookHistogram1D(16,1,1);
       _h_rel_deltaPt_ge2jet_BalPt = bookHistogram1D(17,1,1);
       _h_rel_deltaPt_ge2jet = bookHistogram1D(18,1,1);

    } 




    /// Perform the per-event analysis
    void analyze(const Event& event) {

      const double weight = event.weight();

      const InvMassFinalState& invMassFinalStateW = applyProjection<InvMassFinalState>(event, "INVFSW");
      bool isW(false);
      isW = (!(invMassFinalStateW.empty()));

      const ParticleVector& WDecayProducts = invMassFinalStateW.particles();
      if (WDecayProducts.size() <2) vetoEvent;
      
      double pt1=-9999.,  pt2=-9999.;
      double phi1=-9999., phi2=-9999.;
      double eta1=-9999.;

      double mt = -9999; 

      int iNU_MU=-9999, iAN_MU=-9999;

      if (isW) {
        iNU_MU = (fabs(WDecayProducts[1].pdgId()) == NU_MU) ? 1 : 0;
        iAN_MU = 1 - iNU_MU;
        pt1  = WDecayProducts[iAN_MU].momentum().pT();
        pt2  = WDecayProducts[iNU_MU].momentum().Et();
        eta1 = WDecayProducts[iAN_MU].momentum().eta();
        phi1 = WDecayProducts[iAN_MU].momentum().phi();
        phi2 = WDecayProducts[iNU_MU].momentum().phi();
        mt   = sqrt(2.0*pt1*pt2*(1.0-cos(phi1-phi2)));
      }
      
      if (!isW || mt < 50. || pt1 < 35. || fabs(eta1) > 2.1 || pt2 < 30.) vetoEvent; 


      const FastJets& jetpro = applyProjection<FastJets>(event, "Jets");
      vector<FourMomentum> jets;
      foreach (const Jet& jet, jetpro.jetsByPt(20)) {
        if (fabs(jet.momentum().rapidity()) < 2.0) {
          jets.push_back(jet.momentum());
        }
      }
      
      /// Njet==2 && Njet>=2
      if (jets.size() < 2) vetoEvent;

      double mupx    = pt1*cos(phi1);
      double mupy    = pt1*sin(phi1);
      double met_x   = pt2*cos(phi2);
      double met_y   = pt2*sin(phi2);

      double dpt = ((jets[0].px() + jets[1].px())*(jets[0].px() + jets[1].px()) + \
                    (jets[0].py() + jets[1].py())*(jets[0].py() + jets[1].py())); 
      double rel_dpt = sqrt(dpt)/ (jets[0].pT() + jets[1].pT());
      double dphi = fabs(deltaPhi(jets[0], jets[1]));
         
      double pT2 = (mupx + met_x)*(mupx + met_x) + \
                   (mupy + met_y)*(mupy + met_y); 
      double Px       = (mupx + met_x)*(jets[0].px() + jets[1].px());
      double Py       = (mupy + met_y)*(jets[0].py() + jets[1].py());
      double p1p2_mag = sqrt(dpt)*sqrt(pT2);
      double dS       = acos((Px+Py)/p1p2_mag);

  
      // Njet==2
      if (jets.size() == 2) {
        _h_rel_deltaPt_eq2jet->fill(rel_dpt,weight);
        _h_deltaPhi_eq2jet->fill(dphi,weight);
        _h_deltaS_eq2jet->fill(dS,weight);

        _h_rel_deltaPt_eq2jet_Norm->fill(rel_dpt,weight);
        _h_deltaPhi_eq2jet_Norm->fill(dphi,weight);
        _h_deltaS_eq2jet_Norm->fill(dS,weight); 
      }


      // Njet>=2
      if (jets.size() >= 2) {
        _h_rel_deltaPt_ge2jet->fill(rel_dpt,weight);
        _h_deltaPhi_ge2jet->fill(dphi,weight);
        _h_deltaS_ge2jet->fill(dS,weight);

        _h_rel_deltaPt_ge2jet_Norm->fill(rel_dpt,weight);
        _h_deltaPhi_ge2jet_Norm->fill(dphi,weight);
        _h_deltaS_ge2jet_Norm->fill(dS,weight);
        
     
        // Find the best balanced jet pair and calculate DPS observables accordingly
        int BALpt_gen_lead = 0;
        int BALpt_gen_sublead = 1;
        double pt_gen_min = 999;


        for (unsigned int i = 0; i < jets.size(); ++i) {
            
          if (i > 5) continue;

          for (unsigned int j = i+1; j < jets.size(); ++j) {
            double jet1pt = jets[i].pT();
            double jet2pt = jets[j].pT();
            double jet1_px = jet1pt*cos(jets[i].phi());
            double jet2_px = jet2pt*cos(jets[j].phi());
            double jet1_py = jet1pt*sin(jets[i].phi());
            double jet2_py = jet2pt*sin(jets[j].phi());

          // vector sum of j1 and j2 momentum
            double pT21 = ((jet1_px + jet2_px)*(jet1_px + jet2_px) + \
                           (jet1_py + jet2_py)*(jet1_py + jet2_py));
            double rel_pT2_tmp = fabs(sqrt(pT21)/(jet1pt+jet2pt)); 

            if (rel_pT2_tmp < pt_gen_min) {
              pt_gen_min = rel_pT2_tmp;
              BALpt_gen_lead = i;
              BALpt_gen_sublead = j;
            }
  
          }
        }


        double dpt_v1 = ((jets[BALpt_gen_lead].px() + jets[BALpt_gen_sublead].px()) * \
          (jets[BALpt_gen_lead].px() + jets[BALpt_gen_sublead].px()) + \
          (jets[BALpt_gen_lead].py() + jets[BALpt_gen_sublead].py()) * \
          (jets[BALpt_gen_lead].py() + jets[BALpt_gen_sublead].py()));
        double rel_dpt_v1 = sqrt(dpt_v1) / (jets[BALpt_gen_lead].pT() + jets[BALpt_gen_sublead].pT());
        double dphi_v1 = fabs(deltaPhi(jets[BALpt_gen_lead], jets[BALpt_gen_sublead]));

        double pT2_v1      = (mupx + met_x)*(mupx + met_x) + (mupy + met_y)*(mupy + met_y);
        double Px_v1       = (mupx + met_x)*(jets[BALpt_gen_lead].px() + jets[BALpt_gen_sublead].px());
        double Py_v1       = (mupy + met_y)*(jets[BALpt_gen_lead].py() + jets[BALpt_gen_sublead].py());
        double p1p2_mag_v1 = sqrt(dpt_v1)*sqrt(pT2_v1);
        double dS_v1       = acos((Px_v1+Py_v1)/p1p2_mag_v1);

        _h_rel_deltaPt_ge2jet_BalPt->fill(rel_dpt_v1,weight);
        _h_deltaPhi_ge2jet_BalPt->fill(dphi_v1,weight);
        _h_deltaS_ge2jet_BalPt->fill(dS_v1,weight);

        _h_rel_deltaPt_ge2jet_BalPt_Norm->fill(rel_dpt_v1,weight);
        _h_deltaPhi_ge2jet_BalPt_Norm->fill(dphi_v1,weight);
        _h_deltaS_ge2jet_BalPt_Norm->fill(dS_v1,weight);
  
      }

    } 




    /// Normalise histograms etc., after the run
    void finalize() {

      double rel_dpt_bw = (1.0002 - 0.) / 30.0;
      double dphi_bw = (3.14160 - 0.) / 30.0;

      normalize(_h_rel_deltaPt_eq2jet_Norm, 1.*rel_dpt_bw);
      normalize(_h_deltaPhi_eq2jet_Norm, 1.*dphi_bw);
      normalize(_h_deltaS_eq2jet_Norm, 1.*dphi_bw);

      normalize(_h_rel_deltaPt_ge2jet_Norm, 1.*rel_dpt_bw);
      normalize(_h_deltaPhi_ge2jet_Norm, 1.*dphi_bw);
      normalize(_h_deltaS_ge2jet_Norm, 1.*dphi_bw);

      normalize(_h_rel_deltaPt_ge2jet_BalPt_Norm, 1.*rel_dpt_bw);
      normalize(_h_deltaPhi_ge2jet_BalPt_Norm, 1.*dphi_bw);
      normalize(_h_deltaS_ge2jet_BalPt_Norm, 1.*dphi_bw);


      scale(_h_rel_deltaPt_eq2jet, rel_dpt_bw*crossSection()/sumOfWeights());
      scale(_h_deltaPhi_eq2jet, dphi_bw*crossSection()/sumOfWeights());
      scale(_h_deltaS_eq2jet, dphi_bw*crossSection()/sumOfWeights());

      scale(_h_rel_deltaPt_ge2jet, rel_dpt_bw*crossSection()/sumOfWeights());
      scale(_h_deltaPhi_ge2jet, dphi_bw*crossSection()/sumOfWeights());
      scale(_h_deltaS_ge2jet, dphi_bw*crossSection()/sumOfWeights());

      scale(_h_rel_deltaPt_ge2jet_BalPt, rel_dpt_bw*crossSection()/sumOfWeights());
      scale(_h_deltaPhi_ge2jet_BalPt, dphi_bw*crossSection()/sumOfWeights());
      scale(_h_deltaS_ge2jet_BalPt, dphi_bw*crossSection()/sumOfWeights());

    }




  private:
    
    AIDA::IHistogram1D *_h_rel_deltaPt_eq2jet;
    AIDA::IHistogram1D *_h_deltaPhi_eq2jet;
    AIDA::IHistogram1D *_h_deltaS_eq2jet;
    AIDA::IHistogram1D *_h_rel_deltaPt_ge2jet;
    AIDA::IHistogram1D *_h_deltaPhi_ge2jet;
    AIDA::IHistogram1D *_h_deltaS_ge2jet;
    AIDA::IHistogram1D *_h_rel_deltaPt_ge2jet_BalPt;
    AIDA::IHistogram1D *_h_deltaPhi_ge2jet_BalPt;
    AIDA::IHistogram1D *_h_deltaS_ge2jet_BalPt;

    AIDA::IHistogram1D *_h_rel_deltaPt_eq2jet_Norm;
    AIDA::IHistogram1D *_h_deltaPhi_eq2jet_Norm;
    AIDA::IHistogram1D *_h_deltaS_eq2jet_Norm;
    AIDA::IHistogram1D *_h_rel_deltaPt_ge2jet_Norm;
    AIDA::IHistogram1D *_h_deltaPhi_ge2jet_Norm;
    AIDA::IHistogram1D *_h_deltaS_ge2jet_Norm;
    AIDA::IHistogram1D *_h_rel_deltaPt_ge2jet_BalPt_Norm;
    AIDA::IHistogram1D *_h_deltaPhi_ge2jet_BalPt_Norm;
    AIDA::IHistogram1D *_h_deltaS_ge2jet_BalPt_Norm;

  };




  // The hook for the plugin system
  AnalysisBuilder<CMS_2012_PAS_FSQ_12_028> plugin_CMS_2012_PAS_FSQ_12_028;
}
