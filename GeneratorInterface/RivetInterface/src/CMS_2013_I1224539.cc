// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"

namespace Rivet {


  class CMS_2013_I1224539 : public Analysis {
  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_2013_I1224539()
      : Analysis("CMS_2013_I1224539")
    {    }

    //@}


  public:

    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {

      MSG_INFO("Hello from CMS_2013_I1224539::init! About to book some unnamed histograms.");

      FinalState fs(-2.4, 2.4, 0.0*GeV);
      addProjection(fs, "FS");
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.7), "Jets");
      // GG Rivet. No 2-d histograms. Boooo. 
      for( unsigned i = 0; i < N_PT_BINS; ++i ) {
	_h_ungroomedAvgJetMass[i] = bookHistogram1D(i+1+0*N_PT_BINS,1,1);
      }
      for( unsigned i = 0; i < N_PT_BINS; ++i ) {
	_h_filteredAvgJetMass[i] = bookHistogram1D(i+1+1*N_PT_BINS,1,1);
      }
      for( unsigned i = 0; i < N_PT_BINS; ++i ) {
	_h_trimmedAvgJetMass[i] = bookHistogram1D(i+1+2*N_PT_BINS,1,1);
      }
      for( unsigned i = 0; i < N_PT_BINS; ++i ) {
	_h_prunedAvgJetMass[i] = bookHistogram1D(i+1+3*N_PT_BINS,1,1);
      }

      MSG_INFO("... Done.");


    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {


      // Hacking 2-d histograms. 
      double ptBins[N_PT_BINS+1] = {
	220.0,
	300.0,
	450.0,
	500.0,
	600.0,
	800.0,
	1000.0,
	1500.0};

      const double weight = event.weight() > 0 ? event.weight() : 1;

      // Get the "projections"
      Jets jets = applyProjection<JetAlg>(event, "Jets").jetsByPt(20*GeV);

      // Get the pseudojets. 
      const PseudoJets& psjets = applyProjection<FastJets>(event, "Jets").pseudoJetsByPt( 50.0*GeV );


      // Define the FJ3 grooming algorithms
      double rFilt = 0.3;
      int nFilt = 3;
      double rTrim = 0.2;
      double trimPtFracMin = 0.03;
      double zCut = 0.1;
      double RcutFactor = 0.5;

      fastjet::Filter filter( fastjet::Filter(fastjet::JetDefinition(fastjet::cambridge_algorithm, rFilt), fastjet::SelectorNHardest(nFilt)));
      fastjet::Filter trimmer( fastjet::Filter(fastjet::JetDefinition(fastjet::kt_algorithm, rTrim), fastjet::SelectorPtFractionMin(trimPtFracMin)));
      fastjet::Pruner pruner(fastjet::cambridge_algorithm, zCut, RcutFactor);      

      // Look at events with >= 2 jets
      if (!psjets.empty() && psjets.size() > 1) {

	// Get the leading two jets
        const fastjet::PseudoJet& j0 = psjets[0];
	const fastjet::PseudoJet& j1 = psjets[1];

	// Find their average pt
	double ptAvg = (j0.pt() + j1.pt()) * 0.5;

	// Find the histogram bin that this belongs to. 
	unsigned int njetBin = N_PT_BINS;
	MSG_DEBUG("ptAvg = " << ptAvg);
	for ( unsigned int ibin = 0; ibin < N_PT_BINS; ++ibin ) {
	  if ( ptAvg >= ptBins[ibin] && ptAvg < ptBins[ibin+1] ) {
	    njetBin = ibin;
	    break;
	  }
	}

	if ( njetBin >= N_PT_BINS ) 
	  return;
	MSG_DEBUG("njetBin = " <<njetBin);

	// Now run the substructure algs...
	fastjet::PseudoJet filtered0 = filter(j0); 
	fastjet::PseudoJet filtered1 = filter(j1); 

	fastjet::PseudoJet trimmed0 = trimmer(j0); 
	fastjet::PseudoJet trimmed1 = trimmer(j1); 

	fastjet::PseudoJet pruned0 = pruner(j0); 
	fastjet::PseudoJet pruned1 = pruner(j1); 

	// ... and fill the hists
        _h_ungroomedAvgJetMass[njetBin]->fill( (j0.m() + j1.m()) * 0.5 / GeV, weight);
        _h_filteredAvgJetMass[njetBin]->fill( (filtered0.m() + filtered1.m()) * 0.5 / GeV, weight);
        _h_trimmedAvgJetMass[njetBin]->fill( (trimmed0.m() + trimmed1.m()) * 0.5 / GeV, weight);
        _h_prunedAvgJetMass[njetBin]->fill( (pruned0.m() + pruned1.m()) * 0.5 / GeV, weight);

	MSG_DEBUG("Ungroomed : " << j0.m());
	MSG_DEBUG("Filtered : " << filtered0.m());
	MSG_DEBUG("Trimmed : " << trimmed0.m());
	MSG_DEBUG("Pruned : " << pruned0.m());
      }

    }


    /// Normalise histograms etc., after the run
    void finalize() {
      for ( unsigned int i = 0; i < N_PT_BINS; ++i ) {
	normalize( _h_ungroomedAvgJetMass[i], 1000. );
	normalize( _h_filteredAvgJetMass[i], 1000. );
	normalize( _h_trimmedAvgJetMass[i], 1000. );
	normalize( _h_prunedAvgJetMass[i], 1000. );
      }
    }

    //@}


  private:

    /// @name Histograms
    //@{
    enum {
      PT_220_300=0,
      PT_300_450,
      PT_450_500,
      PT_500_600,
      PT_600_800,
      PT_800_1000,
      PT_1000_1500,
      N_PT_BINS
    } BINS;
    
    AIDA::IHistogram1D * _h_ungroomedJet0pt, * _h_ungroomedJet1pt;
    AIDA::IHistogram1D * _h_ungroomedAvgJetMass[N_PT_BINS];
    AIDA::IHistogram1D * _h_filteredAvgJetMass[N_PT_BINS];
    AIDA::IHistogram1D * _h_trimmedAvgJetMass[N_PT_BINS];
    AIDA::IHistogram1D * _h_prunedAvgJetMass[N_PT_BINS];
    //@}


  };



  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(CMS_2013_I1224539);

}
