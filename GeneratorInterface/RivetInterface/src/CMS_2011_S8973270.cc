// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/UnstableFinalState.hh"
#include "Rivet/Projections/FastJets.hh"

namespace Rivet {


  class CMS_2011_S8973270 : public Analysis {
  public:

    /// Constructor
    CMS_2011_S8973270()
      : Analysis("CMS_2011_S8973270")
    {

      //setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }

  public:

    void init() {

      const FinalState fs;
      UnstableFinalState ufs;

      // Initialize the projectors:
      FastJets jetproj(fs, FastJets::ANTIKT, 0.5);
      jetproj.useInvisibles(); 
      addProjection(jetproj, "Jets");

      addProjection(ufs, "UFS");

      // Book histograms:
      _h_dsigma_dR_56GeV = bookHistogram1D(1,1,1); 
      _h_dsigma_dR_84GeV = bookHistogram1D(2,1,1); 
      _h_dsigma_dR_120GeV = bookHistogram1D(3,1,1); 
      _h_dsigma_dPhi_56GeV = bookHistogram1D(4,1,1); 
      _h_dsigma_dPhi_84GeV = bookHistogram1D(5,1,1); 
      _h_dsigma_dPhi_120GeV = bookHistogram1D(6,1,1); 

    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();

      // Apply the projection:
      const FastJets &fj = applyProjection<FastJets>(event,"Jets");
      const Jets& jets = fj.jets();       
      const UnstableFinalState& ufs = applyProjection<UnstableFinalState>(event, "UFS");


      //find the leading jet pT
      double ljpT = -1.0; 
      double ljeta = -10.0;
      
      foreach(const Jet &j, jets) {
         if(j.momentum().pT() > ljpT) {
           ljpT = j.momentum().pT(); 
           ljeta = j.momentum().eta();
         } 
      }
      MSG_DEBUG("\nleading jet pT / eta: " << ljpT << " / " << ljeta << "\n");

      //minimum requirement for event
      if(ljpT > 56 && fabs(ljeta) < 3.0){
        //find B hadrons in event
        int nab = 0, nb = 0; //counters for all B and independent B hadrons
        double etaB1 = 7.7, etaB2 = 7.7; 
        double phiB1 = 7.7, phiB2 = 7.7; 
        double pTB1 = 7.7, pTB2 = 7.7;
 
        foreach (const Particle& p, ufs.particles()){
          int aid = fabs(p.pdgId()); 
          if((aid/100)==5 || (aid/1000)==5) {
            nab++;
            //2J+1 == 1 (mesons) or 2 (baryons) 
            if((aid%10)==1 || (aid%10)==2 ) {
              //no B decaying to B
              if(aid!=5222 && aid!=5112 && aid!=5212 && aid!=5322){ 
                if(nb==0){
                  etaB1 = p.momentum().eta(); 
                  phiB1 = p.momentum().phi(); 
                  pTB1 = p.momentum().pT(); 
                }
                else if(nb==1){
                  etaB2 = p.momentum().eta();
                  phiB2 = p.momentum().phi();
                  pTB2 = p.momentum().pT(); 
                }
                nb++;
              }
            } 
            MSG_DEBUG("id " << aid <<  " B Hadron\n");
          }
        }
    
        if(nb==2 && pTB1 > 15 && pTB2 > 15 && fabs(etaB1) < 2.0 && fabs(etaB2) < 2.0){ 
          double dPhi = deltaPhi(phiB1, phiB2); 
          double dR = deltaR(etaB1, phiB1, etaB2, phiB2); 
          MSG_DEBUG("DR/DPhi " << dR << " " << dPhi << "\n");

          _h_dsigma_dR_56GeV->fill(dR, weight); 
          if(ljpT > 84) _h_dsigma_dR_84GeV->fill(dR, weight); 
          if(ljpT > 120) _h_dsigma_dR_120GeV->fill(dR, weight); 
          _h_dsigma_dPhi_56GeV->fill(dPhi, weight); 
          if(ljpT > 84) _h_dsigma_dPhi_84GeV->fill(dPhi, weight); 
          if(ljpT > 120) _h_dsigma_dPhi_120GeV->fill(dPhi, weight); 
          /// @todo Do the event by event analysis here
          MSG_DEBUG("nb " << nb << " " << nab << "\n");
        }
      }
    }

    /// Normalise histograms etc., after the run
    void finalize() {

      /// @todo Normalise, scale and otherwise manipulate histograms here
      
      MSG_DEBUG("crossSection " << crossSection() << " sumOfWeights " 
                                << sumOfWeights() << "\n"); 

      //hardcoded bin widths
      double DRbin = 0.4; 
      double DPhibin = PI/8.0; 
      scale(_h_dsigma_dR_56GeV, crossSection()/sumOfWeights()*DRbin);
      scale(_h_dsigma_dR_84GeV, crossSection()/sumOfWeights()*DRbin);
      scale(_h_dsigma_dR_120GeV, crossSection()/sumOfWeights()*DRbin);
      scale(_h_dsigma_dPhi_56GeV, crossSection()/sumOfWeights()*DPhibin);
      scale(_h_dsigma_dPhi_84GeV, crossSection()/sumOfWeights()*DPhibin);
      scale(_h_dsigma_dPhi_120GeV, crossSection()/sumOfWeights()*DPhibin);

    }

    //@}


  private:

    /// @name Histograms
    //@{

    AIDA::IHistogram1D *_h_dsigma_dR_56GeV;
    AIDA::IHistogram1D *_h_dsigma_dR_84GeV;
    AIDA::IHistogram1D *_h_dsigma_dR_120GeV;
    AIDA::IHistogram1D *_h_dsigma_dPhi_56GeV;
    AIDA::IHistogram1D *_h_dsigma_dPhi_84GeV;
    AIDA::IHistogram1D *_h_dsigma_dPhi_120GeV;
    //@}

  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8973270> plugin_CMS_2011_S8973270;


}
