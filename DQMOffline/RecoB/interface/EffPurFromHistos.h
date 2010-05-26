#ifndef EffPurFromHistos_H
#define EffPurFromHistos_H

#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include "TH1F.h"
#include "TString.h"
#include "TCanvas.h"


//class DQMStore;

class EffPurFromHistos {


 public:

  EffPurFromHistos ( const TString & ext, TH1F * h_d, TH1F * h_u,
	TH1F * h_s, TH1F * h_c, TH1F * h_b, TH1F * h_g,	TH1F * h_ni,
		     TH1F * h_dus, TH1F * h_dusg, std::string label, bool mc,
	int nBin = 100 , double startO = 0.005 , double endO = 1.005 ) ;
	// defaults reasonable for lifetime based tags

  EffPurFromHistos (const FlavourHistograms<double> * dDiscriminatorFC, std::string label, bool mc,
	int nBin = 100 , double startO = 0.005 , double endO = 1.005 ) ;
	// defaults reasonable for lifetime based tags

  ~EffPurFromHistos () ;

  // do the computation
  void compute () ;

  // return the newly created histos
  TH1F * getEffFlavVsBEff_d    () { 
    return EffFlavVsBEff_d->getTH1F()    ; };
  TH1F * getEffFlavVsBEff_u    () { return EffFlavVsBEff_u->getTH1F()    ; };
  TH1F * getEffFlavVsBEff_s    () { return EffFlavVsBEff_s ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_c    () { return EffFlavVsBEff_c ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_b    () { return EffFlavVsBEff_b ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_g    () { return EffFlavVsBEff_g ->getTH1F()   ; };
  TH1F * getEffFlavVsBEff_ni   () { return EffFlavVsBEff_ni ->getTH1F()  ; };
  TH1F * getEffFlavVsBEff_dus  () { return EffFlavVsBEff_dus ->getTH1F() ; };
  TH1F * getEffFlavVsBEff_dusg () { return EffFlavVsBEff_dusg ->getTH1F(); };

 

  void epsPlot(const TString & name);

  void psPlot(const TString & name);

  void plot(TPad * theCanvas = 0) ;

  void plot(const TString & name, const TString & ext);

//   void print () const ;

  FlavourHistograms<double> * discriminatorNoCutEffic() const {return discrNoCutEffic;}
  FlavourHistograms<double> * discriminatorCutEfficScan() const {return discrCutEfficScan;}

 private:


  // consistency check (same binning)
  void check () ;
  bool fromDiscriminatorDistr;


  // the TString for the histo name extension
  TString histoExtension ;

  FlavourHistograms<double> * discrNoCutEffic, *discrCutEfficScan;

  // the input histograms (efficiency versus discriminator cut)
  // IMPORTANT: IT'S ASSUMED THAT ALL HISTOS HAVE THE SAME BINNING!!
  // (can in principle be relaxed by checking explicitely for the discriminator value
  //  instead of bin index)
  TH1F * effVersusDiscr_d    ;
  TH1F * effVersusDiscr_u    ;
  TH1F * effVersusDiscr_s    ;
  TH1F * effVersusDiscr_c    ;
  TH1F * effVersusDiscr_b    ;
  TH1F * effVersusDiscr_g    ;
  TH1F * effVersusDiscr_ni   ;
  TH1F * effVersusDiscr_dus  ;
  TH1F * effVersusDiscr_dusg ;


  // the corresponding output histograms (flavour-eff vs. b-efficiency)

  // binning for output histograms
  int   nBinOutput ;
  double startOutput ;
  double endOutput ;

  bool mcPlots_;


  MonitorElement * EffFlavVsBEff_d    ;
  MonitorElement * EffFlavVsBEff_u    ;
  MonitorElement * EffFlavVsBEff_s    ;
  MonitorElement * EffFlavVsBEff_c    ;
  MonitorElement * EffFlavVsBEff_b    ;
  MonitorElement * EffFlavVsBEff_g    ;
  MonitorElement * EffFlavVsBEff_ni   ;
  MonitorElement * EffFlavVsBEff_dus  ;
  MonitorElement * EffFlavVsBEff_dusg ;

  //  DQMStore * dqmStore_; 
  std::string label_;

} ;

#endif
