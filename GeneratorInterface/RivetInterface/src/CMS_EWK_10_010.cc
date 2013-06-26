// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
//#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/ZFinder.hh"


namespace Rivet {


  /// @brief Measurement of Z(->muon muon) pT and y differential cross-section
  /// @author Justin Hugon
  class CMS_EWK_10_010 : public Analysis {
  public:

    /// @name Construction
    //@{

    /// Constructor
    CMS_EWK_10_010() : Analysis("CMS_EWK_10_010")
    {
    }

    //@}


    ///@name Analysis methods
    //@{

    /// Add projections and book histograms
    void init() {
	
	//Only Mu and El unless Taus are set stable
      ZFinder zfinder(-MAXRAPIDITY, MAXRAPIDITY, 0.0*GeV, MUON, 
				60.0*GeV, 120.0*GeV, 0.2, false, true);

      addProjection(zfinder, "ZFinder");

      _h_Z_pT_normalised = bookHistogram1D(1, 1, 1);
      _h_Z_pT_peak_normalised = bookHistogram1D(2, 1, 1);
      _h_Z_y_normalised = bookHistogram1D(3, 1, 1);


    }


    // Do the analysis
    void analyze(const Event& e) {
      const double weight = e.weight();

      const ZFinder& zfinder = applyProjection<ZFinder>(e, "ZFinder");

      if (zfinder.particles().size() != 2) 
	vetoEvent;

      Particle lepton0 = zfinder.particles().at(0);
      Particle lepton1 = zfinder.particles().at(1);

      if (lepton0.pdgId() != -lepton1.pdgId())
	vetoEvent;

      double pt0 = lepton0.momentum().pT()/GeV;
      double pt1 = lepton1.momentum().pT()/GeV;
      double eta0 = lepton0.momentum().eta();
      double eta1 = lepton1.momentum().eta();

      double Zpt = zfinder.bosons()[0].momentum().pT()/GeV;
      double Zy = zfinder.bosons()[0].momentum().rapidity();

	//Begin Pt Part
      bool inAcceptance = fabs(eta0)<2.1 && fabs(eta1)<2.1 && pt0>20 && pt1>20;
      if (inAcceptance)
      {
        _h_Z_pT_normalised->fill(Zpt, weight);
        _h_Z_pT_peak_normalised->fill(Zpt, weight);
      }
	//Begin Rapidity Part
      _h_Z_y_normalised->fill(Zy,weight);

    }


    /// Finalize
    void finalize() {
      double pT_integral = _h_Z_pT_normalised->sumBinHeights();
      double pT_peak_integral = _h_Z_pT_peak_normalised->sumBinHeights();

      normalize(_h_Z_pT_normalised,1.0);
      normalize(_h_Z_pT_peak_normalised,pT_peak_integral/pT_integral);
      normalize(_h_Z_y_normalised,2.0);

    }

    //@}


  private:

    /// @name Histogram
    AIDA::IHistogram1D * _h_Z_pT_normalised;
    AIDA::IHistogram1D * _h_Z_pT_peak_normalised;
    AIDA::IHistogram1D * _h_Z_y_normalised;

  };


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_EWK_10_010> plugin_CMS_EWK_10_010;


}
