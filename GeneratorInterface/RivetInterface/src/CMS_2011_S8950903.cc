// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Tools/BinnedHistogram.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/RivetAIDA.hh"

namespace Rivet {
 
 
   // CMS azimuthal decorellations
   
  class CMS_2011_S8950903 : public Analysis {
  public:
    
    CMS_2011_S8950903() : Analysis("CMS_2011_S8950903") {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }
 
 
// ======================================================== init

    void init() {
      FinalState fs;
      FastJets akt(fs, FastJets::ANTIKT, 0.5);
      addProjection(akt, "antikT");
      
      if(fuzzyEquals(sqrtS(), 7000*GeV, 1E-3)){ 
        
        _h_dPhi_pT1 = bookHistogram1D(1, 1, 1);
        _h_dPhi_pT2 = bookHistogram1D(2, 1, 1);
        _h_dPhi_pT3 = bookHistogram1D(3, 1, 1);
        _h_dPhi_pT4 = bookHistogram1D(4, 1, 1);
        _h_dPhi_pT5 = bookHistogram1D(5, 1, 1);
        
      }
    }
    
// ======================================================== analyze
    
    void analyze(const Event & event) {
      const double weight = event.weight();
      
      const Jets& jets = applyProjection<JetAlg>(event, "antikT").jetsByPt();
      if (jets.size() < 2) vetoEvent;
      
      if (fabs(jets[0].momentum().eta()) > 1.1 || jets[0].momentum().pT() < 80.) vetoEvent;
      if (fabs(jets[1].momentum().eta()) > 1.1 || jets[1].momentum().pT() < 30.) vetoEvent;
      
      double phi1 = jets[0].momentum().phi();
      double phi2 = jets[1].momentum().phi();
      
      double dphi = fabs(phi1-phi2);
      if (fabs(phi1-phi2) > 3.14159265358) dphi = 2*3.14159265358 - fabs(phi1-phi2);
      
      // ordering based on max pT
      if (jets[0].momentum().pT() > 80.  && jets[0].momentum().pT() < 110.)  _h_dPhi_pT1->fill(dphi, weight);
      if (jets[0].momentum().pT() > 110. && jets[0].momentum().pT() < 140.)  _h_dPhi_pT2->fill(dphi, weight);
      if (jets[0].momentum().pT() > 140. && jets[0].momentum().pT() < 200.)  _h_dPhi_pT3->fill(dphi, weight);
      if (jets[0].momentum().pT() > 200. && jets[0].momentum().pT() < 300.)  _h_dPhi_pT4->fill(dphi, weight);
      if (jets[0].momentum().pT() > 300.) _h_dPhi_pT5->fill(dphi, weight);
      
    }
    
// ======================================================== finalize 
 
    void finalize() {

      normalize(_h_dPhi_pT1,1.);
      normalize(_h_dPhi_pT2,1.);
      normalize(_h_dPhi_pT3,1.);
      normalize(_h_dPhi_pT4,1.);
      normalize(_h_dPhi_pT5,1.);
      
    }
    
  private:
    
    AIDA::IHistogram1D* _h_dPhi_pT1;
    AIDA::IHistogram1D* _h_dPhi_pT2;
    AIDA::IHistogram1D* _h_dPhi_pT3;
    AIDA::IHistogram1D* _h_dPhi_pT4;
    AIDA::IHistogram1D* _h_dPhi_pT5;
    
  }; 
 
  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8950903> plugin_CMS_2011_S8950903;
  
}

