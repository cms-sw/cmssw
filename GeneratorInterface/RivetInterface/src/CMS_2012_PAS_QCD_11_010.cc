// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/UnstableFinalState.hh"
#include "Rivet/Projections/FastJets.hh"

namespace Rivet {

  class CMS_2012_PAS_QCD_11_010 : public Analysis {
  public:

    CMS_2012_PAS_QCD_11_010()
      : Analysis("CMS_2012_PAS_QCD_11_010")
    {  }

    void init() {

      const ChargedFinalState cfs(-2.5, 2.5, 0.5*GeV); // tracks are accepted with pT > 0.5 GeV
      addProjection(cfs, "CFS");

      const FastJets jetpro(cfs, FastJets::ANTIKT, 0.5);
      addProjection(jetpro, "Jets");

      const UnstableFinalState ufs(-2.0, 2.0, 0.6*GeV);
      addProjection(ufs, "UFS");

      _hist_nTrans_Lambda = bookProfile1D(1, 1, 1);
      _hist_nTrans_Kaon = bookProfile1D(2, 1, 1);
      _hist_ptsumTrans_Lambda = bookProfile1D(3, 1, 1);
      _hist_ptsumTrans_Kaon = bookProfile1D(4, 1, 1);

    }

    void analyze(const Event& event) {

      const double weight = event.weight();

      Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(1.0*GeV);
      if (jets.size() < 1) vetoEvent;

      if (fabs(jets[0].momentum().eta()) >= 2 || jets[0].momentum().pT() < 1.) { // cuts on leading jets
	vetoEvent;
      }

      FourMomentum p_lead = jets[0].momentum();
      const double philead = p_lead.phi();
      const double pTlead  = p_lead.perp();

      const UnstableFinalState& ufs = applyProjection<UnstableFinalState>(event, "UFS");

      double numTrans_Kaon(0.0), ptSumTrans_Kaon(0.0);
      double numTrans_Lambda(0.), ptSumTrans_Lambda(0.0);

      foreach (const Particle& p, ufs.particles()) {
        double dphi = fabs(deltaPhi(philead, p.momentum().phi()));
        double pT = p.momentum().pT();
        const PdgId id = p.pdgId();
      
        switch (id) {
	case 310: // K0s
	if (((PI/3.0 < dphi) && (dphi < 2.0*PI/3.0)) && (pT > 0.6)) {
	      ptSumTrans_Kaon += pT;	
	      numTrans_Kaon += 1.0;
	}
	break;

	case 3122: case -3122: // Lambda, Lambdabar 
	if (((PI/3.0 < dphi) && (dphi < 2.0*PI/3.0)) && (pT > 1.5)) {
              ptSumTrans_Lambda += pT;	
              numTrans_Lambda += 1.0;
	}
	break;

	}

      }

      _hist_nTrans_Kaon->fill(pTlead/GeV, numTrans_Kaon / ((8.0 * PI/3.0)), weight);
      _hist_nTrans_Lambda->fill(pTlead/GeV, numTrans_Lambda / ((8.0 * PI/3.0)), weight);
      _hist_ptsumTrans_Kaon->fill(pTlead/GeV , ptSumTrans_Kaon / ((GeV * (8.0 * PI/3.0))) , weight);
      _hist_ptsumTrans_Lambda->fill(pTlead/GeV , ptSumTrans_Lambda / ((GeV * (8.0 * PI/3.0))) , weight);

    }

    void finalize() {

    }

  private:

    AIDA::IProfile1D *_hist_nTrans_Kaon;
    AIDA::IProfile1D *_hist_nTrans_Lambda;
    AIDA::IProfile1D *_hist_ptsumTrans_Kaon;
    AIDA::IProfile1D *_hist_ptsumTrans_Lambda;

  };

  // AK DECLARE_RIVET_PLUGIN(CMS_2012_PAS_QCD_11_010);
  AnalysisBuilder<CMS_2012_PAS_QCD_11_010> plugin_CMS_2012_PAS_QCD_11_010;
}
