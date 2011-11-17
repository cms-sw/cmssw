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
      
      _countMCDR56 = 0.; 
      _countMCDR84 = 0.; 
      _countMCDR120 = 0.; 
      _countMCDPhi56 = 0.; 
      _countMCDPhi84 = 0.; 
      _countMCDPhi120 = 0.; 

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
          
          //MC counters
          if(dR > 2.4) _countMCDR56 += weight; 
          if(dR > 2.4 && ljpT > 84.) _countMCDR84 += weight; 
          if(dR > 2.4 && ljpT > 120.) _countMCDR120 += weight; 
          if(dPhi > 3.*PI/4.) _countMCDPhi56 += weight; 
          if(dPhi > 3.*PI/4. && ljpT > 84.) _countMCDPhi84 += weight; 
          if(dPhi > 3.*PI/4. && ljpT > 120.) _countMCDPhi120 += weight; 

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
      
      MSG_DEBUG("crossSection " << crossSection() << " sumOfWeights " 
                                << sumOfWeights() << "\n"); 

      //hardcoded bin widths
      double DRbin = 0.4; 
      double DPhibin = PI/8.0; 
      //Find out the correct numbers
      double nDataDR56 = 25862.20; 
      double nDataDR84 = 5675.55; 
      double nDataDR120 = 1042.72; 
      double nDataDPhi56 = 24220.00; 
      double nDataDPhi84 = 4964.00; 
      double nDataDPhi120 = 919.10; 
      double normDR56 = (_countMCDR56 > 0.) ? nDataDR56/_countMCDR56 : crossSection()/sumOfWeights();
      double normDR84 = (_countMCDR84 > 0.) ? nDataDR84/_countMCDR84 : crossSection()/sumOfWeights();
      double normDR120 = (_countMCDR120 > 0.) ? nDataDR120/_countMCDR120 : crossSection()/sumOfWeights();
      double normDPhi56 = (_countMCDPhi56 > 0.) ? nDataDPhi56/_countMCDPhi56 : crossSection()/sumOfWeights();
      double normDPhi84 = (_countMCDPhi84 > 0.) ? nDataDPhi84/_countMCDPhi84 : crossSection()/sumOfWeights();
      double normDPhi120 = (_countMCDPhi120 > 0.) ? nDataDPhi120/_countMCDPhi120 : crossSection()/sumOfWeights();
      scale(_h_dsigma_dR_56GeV, normDR56*DRbin);
      scale(_h_dsigma_dR_84GeV, normDR84*DRbin);
      scale(_h_dsigma_dR_120GeV, normDR120*DRbin);
      scale(_h_dsigma_dPhi_56GeV, normDPhi56*DPhibin);
      scale(_h_dsigma_dPhi_84GeV, normDPhi84*DPhibin);
      scale(_h_dsigma_dPhi_120GeV, normDPhi120*DPhibin);

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
    double _countMCDR56;
    double _countMCDR84;
    double _countMCDR120;
    double _countMCDPhi56;
    double _countMCDPhi84;
    double _countMCDPhi120;
    //@}

  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8973270> plugin_CMS_2011_S8973270;


}
