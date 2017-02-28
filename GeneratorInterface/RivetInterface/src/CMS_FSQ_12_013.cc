// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/UnstableFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include <cmath>

/// @todo Include more projections as required, e.g. ChargedFinalState, FastJets, ZFinder...

namespace Rivet {


  class CMS_FSQ_12_013 : public Analysis {
  
  private:
  
    AIDA::IHistogram1D *_h_DeltaPtRelSoft;
    AIDA::IHistogram1D *_h_DeltaPhiSoft;
    AIDA::IHistogram1D *_h_DeltaS;
    
    AIDA::IHistogram1D *_h_DeltaPtRelSoft_norm;
    AIDA::IHistogram1D *_h_DeltaPhiSoft_norm;
    AIDA::IHistogram1D *_h_DeltaS_norm;

    AIDA::IHistogram1D *_h_LeadingHardJetPt;
    AIDA::IHistogram1D *_h_SubLeadingHardJetPt;
    AIDA::IHistogram1D *_h_LeadingSoftJetPt;
    AIDA::IHistogram1D *_h_SubLeadingSoftJetPt;

    AIDA::IHistogram1D *_h_LeadingHardJetEta;
    AIDA::IHistogram1D *_h_SubLeadingHardJetEta;
    AIDA::IHistogram1D *_h_LeadingSoftJetEta;
    AIDA::IHistogram1D *_h_SubLeadingSoftJetEta;


  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_FSQ_12_013()
      : Analysis("CMS_FSQ_12_013")
    {
      /// @todo Set whether your finalize method needs the generator cross section
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }

    //@}


    /// Book histograms and initialise projections before the run
    void init() {

      /// @todo Initialise and register projections here
      const FinalState cnfs(-4.7,4.7);
      addProjection(FastJets(cnfs, FastJets::ANTIKT, 0.5), "Jets");

      _h_DeltaPtRelSoft = bookHistogram1D(13,1,1);
      _h_DeltaPhiSoft = bookHistogram1D(4,1,1);
      _h_DeltaS = bookHistogram1D(5,1,1);

      _h_DeltaPtRelSoft_norm = bookHistogram1D(13,1,1);
      _h_DeltaPhiSoft_norm = bookHistogram1D(4,1,1);
      _h_DeltaS_norm = bookHistogram1D(5,1,1);
    
      _h_LeadingHardJetPt = bookHistogram1D(8,1,1);
      _h_SubLeadingHardJetPt = bookHistogram1D(19,1,1);
      _h_LeadingSoftJetPt = bookHistogram1D(15,1,1);
      _h_SubLeadingSoftJetPt = bookHistogram1D(17,1,1);

      _h_LeadingHardJetEta = bookHistogram1D(7,1,1);
      _h_SubLeadingHardJetEta = bookHistogram1D(18,1,1);
      _h_LeadingSoftJetEta = bookHistogram1D(14,1,1);
      _h_SubLeadingSoftJetEta = bookHistogram1D(16,1,1);

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();
      
      /// @todo Do the event by event analysis here
      
      unsigned int num_jets = 0;
      unsigned int num_Softjets = 0;
      const Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(20*GeV);
      
      if (jets.size() < 4) vetoEvent;
      
      Jets jetAnalysis;
      Jets jetSoftAnalysis;
      foreach (const Jet& j, jets) {
	
	if(fabs(j.momentum().eta()) < 4.7 && j.momentum().pT()>=50) {
	
	  num_jets +=1;
	  jetAnalysis.push_back(j);
	  	  
	}

	if(fabs(j.momentum().eta()) < 4.7 && j.momentum().pT()>=20) {
	  
	  num_Softjets +=1;
	  jetSoftAnalysis.push_back(j);
          	  
        }
      }

      if (num_jets >= 2 && num_Softjets == 4) {

	double dphisoft=-100;
	dphisoft = deltaPhi(jetSoftAnalysis[3].momentum().azimuthalAngle(),jetSoftAnalysis[2].momentum().azimuthalAngle());
	_h_DeltaPhiSoft->fill(dphisoft,weight);
	_h_DeltaPhiSoft_norm->fill(dphisoft,weight);

	double vecsumlightsoft=-100;		
	vecsumlightsoft=sqrt(pow(jetSoftAnalysis[2].momentum().px()+jetSoftAnalysis[3].momentum().px(),2)+pow(jetSoftAnalysis[2].momentum().py()+jetSoftAnalysis[3].momentum().py(),2));
	double SptSoft=(vecsumlightsoft/(fabs(jetSoftAnalysis[2].momentum().pT())+fabs(jetSoftAnalysis[3].momentum().pT())));
	_h_DeltaPtRelSoft->fill(SptSoft, weight);
	_h_DeltaPtRelSoft_norm->fill(SptSoft, weight);

	double DPtHard = ((jetAnalysis[0].momentum().px()+jetAnalysis[1].momentum().px())*(jetAnalysis[0].momentum().px()+jetAnalysis[1].momentum().px())+(jetAnalysis[0].momentum().py()+jetAnalysis[1].momentum().py())*(jetAnalysis[0].momentum().py()+jetAnalysis[1].momentum().py()));
	double DPtSoft = ((jetSoftAnalysis[2].momentum().px()+jetSoftAnalysis[3].momentum().px())*(jetSoftAnalysis[2].momentum().px()+jetSoftAnalysis[3].momentum().px())+(jetSoftAnalysis[2].momentum().py()+jetSoftAnalysis[3].momentum().py())*(jetSoftAnalysis[2].momentum().py()+jetSoftAnalysis[3].momentum().py()));
	
	double Px = (jetAnalysis[0].momentum().px()+jetAnalysis[1].momentum().px())*(jetSoftAnalysis[2].momentum().px()+jetSoftAnalysis[3].momentum().px());
	double Py = (jetAnalysis[0].momentum().py()+jetAnalysis[1].momentum().py())*(jetSoftAnalysis[2].momentum().py()+jetSoftAnalysis[3].momentum().py());  
	
	double p1p2_mag = sqrt(DPtHard)*sqrt(DPtSoft);
	double DeltaS = acos((Px+Py)/p1p2_mag);
	
	_h_DeltaS->fill(DeltaS,weight);
	_h_DeltaS_norm->fill(DeltaS,weight);

	_h_LeadingHardJetPt->fill(jetAnalysis[0].momentum().pT(), weight);
	_h_SubLeadingHardJetPt->fill(jetAnalysis[1].momentum().pT(), weight);
	_h_LeadingSoftJetPt->fill(jetSoftAnalysis[2].momentum().pT(), weight);
	_h_SubLeadingSoftJetPt->fill(jetSoftAnalysis[3].momentum().pT(), weight);

	_h_LeadingHardJetEta->fill(jetAnalysis[0].momentum().eta(), weight);
        _h_SubLeadingHardJetEta->fill(jetAnalysis[1].momentum().eta(), weight);
	_h_LeadingSoftJetEta->fill(jetSoftAnalysis[2].momentum().eta(), weight);
	_h_SubLeadingSoftJetEta->fill(jetSoftAnalysis[3].momentum().eta(), weight);

	}    

    }

    /// Normalise histograms etc., after the run
    void finalize() {

      cout<<"cross section "<<crossSection()/picobarn<<endl;
      cout<<"sum of weights "<<sumOfWeights()<<endl;
      double invlumi = crossSection()/picobarn/sumOfWeights(); //norm to cross section
      
      scale(_h_DeltaPtRelSoft, invlumi);
      scale(_h_DeltaPhiSoft, invlumi);
      scale(_h_DeltaS, invlumi);

      scale(_h_DeltaPtRelSoft, 1/integral(_h_DeltaPtRelSoft));
      scale(_h_DeltaPhiSoft, 1/integral(_h_DeltaPhiSoft));
      scale(_h_DeltaS, 1/integral(_h_DeltaS));

      scale(_h_LeadingHardJetPt, invlumi);
      scale(_h_SubLeadingHardJetPt, invlumi);
      scale(_h_LeadingSoftJetPt, invlumi);
      scale(_h_SubLeadingSoftJetPt, invlumi);

      scale(_h_LeadingHardJetEta, invlumi);
      scale(_h_SubLeadingHardJetEta, invlumi);
      scale(_h_LeadingSoftJetEta, invlumi);
      scale(_h_SubLeadingSoftJetEta, invlumi);

    }

  };

  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_FSQ_12_013> plugin_CMS_FSQ_12_013;


}
