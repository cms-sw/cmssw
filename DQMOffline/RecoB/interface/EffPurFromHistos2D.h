#ifndef EffPurFromHistos2D_H
#define EffPurFromHistos2D_H

#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TH2F.h"
#include "TCanvas.h"

#include <string>

//class DQMStore;

class EffPurFromHistos2D {


 public:

  EffPurFromHistos2D ( const std::string & ext, TH2F * h_d, TH2F * h_u,
                     TH2F * h_s, TH2F * h_c, TH2F * h_b, TH2F * h_g, TH2F * h_ni,
                     TH2F * h_dus, TH2F * h_dusg, TH2F * h_pu,
		     const std::string& label, const unsigned int& mc,
		     int nBinX = 50 , double startOX = -1.011 , double endOX = 1.011, int nBinY = 50 , double startOY = -1.011 , double endOY = 1.011) ;
	// defaults reasonable for lifetime based tags

  EffPurFromHistos2D (const FlavourHistograms2D<double, double> * dDiscriminatorFC,  
                      const std::string& label, const unsigned int& mc, 
		      DQMStore::IBooker & ibook,
		      int nBinX = 50 , double startOX = -1.011 , double endOX = 1.011, int nBinY = 50 , double startOY = -1.011 , double endOY = 1.011 ) ;
	// defaults reasonable for lifetime based tags

  ~EffPurFromHistos2D () ;

  // do the computation
  void compute (DQMStore::IBooker & ibook) ;
  
  // return the newly created histos
  //TH2F * getDUSG_reject_vs_B_reject_at_cEff () { return DUSG_reject_vs_B_reject_at_cEff->getTH2F()    ; };
/*
  TH1F * getEffFlavVsBEff_d    () { return EffFlavVsXEff_d->getTH1F()    ; };
  TH1F * getEffFlavVsBEff_u    () { return EffFlavVsXEff_u->getTH1F()    ; };
  TH1F * getEffFlavVsBEff_s    () { return EffFlavVsXEff_s ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_c    () { return EffFlavVsXEff_c ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_b    () { return EffFlavVsXEff_b ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_g    () { return EffFlavVsXEff_g ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_ni   () { return EffFlavVsXEff_ni ->getTH1F()  ; };
  TH1F * getEffFlavVsBEff_dus  () { return EffFlavVsXEff_dus ->getTH1F() ; };
  TH1F * getEffFlavVsBEff_dusg () { return EffFlavVsXEff_dusg ->getTH1F(); };
  TH1F * getEffFlavVsBEff_pu   () { return EffFlavVsXEff_pu ->getTH1F(); };
*/
 
  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  void plot(TPad * theCanvas = 0) ;

  void plot(const std::string & name, const std::string & ext);

//   void print () const ;

  FlavourHistograms2D<double,double> * discriminatorNoCutEffic() const {return discrNoCutEffic;}
  FlavourHistograms2D<double,double> * discriminatorCutEfficScan() const {return discrCutEfficScan;}

  bool doCTagPlots(bool Ctag) {doCTagPlots_ = Ctag; return doCTagPlots_;};
 
 private:


  // consistency check (same binning)
  void check () ;
  bool fromDiscriminatorDistr;


  // the string for the histo name extension
  std::string histoExtension ;

  FlavourHistograms2D<double, double> * discrNoCutEffic, * discrCutEfficScan;

  // the input histograms (efficiency versus discriminator cut)
  // IMPORTANT: IT'S ASSUMED THAT ALL HISTOS HAVE THE SAME BINNING!!
  // (can in principle be relaxed by checking explicitely for the discriminator value
  //  instead of bin index)
  TH2F * effVersusDiscr_d    ;
  TH2F * effVersusDiscr_u    ;
  TH2F * effVersusDiscr_s    ;
  TH2F * effVersusDiscr_c    ;
  TH2F * effVersusDiscr_b    ;
  TH2F * effVersusDiscr_g    ;
  TH2F * effVersusDiscr_ni   ;
  TH2F * effVersusDiscr_dus  ;
  TH2F * effVersusDiscr_dusg ;
  TH2F * effVersusDiscr_pu   ;

  // the corresponding output histograms (flavour-eff vs. b-efficiency)

  // binning for output histograms
  int nBinOutputX;
  double startOutputX; 
  double endOutputX;
  int nBinOutputY; 
  double startOutputY; 
  double endOutputY;
  
  unsigned int mcPlots_;
  bool doCTagPlots_;
  /*
  MonitorElement * EffFlavVsXEff_d    ;
  MonitorElement * EffFlavVsXEff_u    ;
  MonitorElement * EffFlavVsXEff_s    ;
  MonitorElement * EffFlavVsXEff_c    ;
  MonitorElement * EffFlavVsXEff_b    ;
  MonitorElement * EffFlavVsXEff_g    ;
  MonitorElement * EffFlavVsXEff_ni   ;
  MonitorElement * EffFlavVsXEff_dus  ;
  MonitorElement * EffFlavVsXEff_dusg ;
  MonitorElement * EffFlavVsXEff_pu   ;
 */
  MonitorElement * DUSG_reject_vs_B_reject_at_cEff;

  //  DQMStore * dqmStore_; 
  std::string label_;

} ;

#endif
