#ifndef EffPurFromHistos_H
#define EffPurFromHistos_H

#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TH1F.h"
#include "TCanvas.h"

#include <string>

class EffPurFromHistos {

 public:

  EffPurFromHistos(const std::string & ext, TH1F * h_d, TH1F * h_u,
             TH1F * h_s, TH1F * h_c, TH1F * h_b, TH1F * h_g, TH1F * h_ni,
             TH1F * h_dus, TH1F * h_dusg, TH1F * h_pu, 
             const std::string& label, unsigned int mc,
             int nBin = 100, double startO = 0.005, double endO = 1.005);

  EffPurFromHistos(const FlavourHistograms<double>& dDiscriminatorFC, const std::string& label, unsigned int mc, 
            DQMStore::IBooker & ibook, int nBin = 100, double startO = 0.005, double endO = 1.005);

  ~EffPurFromHistos();

  // do the computation
  void compute (DQMStore::IBooker & ibook) ;

  // return the newly created histos
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

 
  void epsPlot(const std::string & name);
  void psPlot(const std::string & name);

  void plot(TPad * theCanvas = nullptr);
  void plot(const std::string & name, const std::string & ext);

  FlavourHistograms<double>& discriminatorNoCutEffic() const { return *discrNoCutEffic; }
  FlavourHistograms<double>& discriminatorCutEfficScan() const { return *discrCutEfficScan; }

  bool doCTagPlots(bool Ctag) { doCTagPlots_ = Ctag; return doCTagPlots_; }
 
 private:

  // consistency check (same binning)
  void check();
  bool fromDiscriminatorDistr;

  unsigned int mcPlots_;
  bool doCTagPlots_;
  std::string label_;
  // the string for the histo name extension
  std::string histoExtension;

  std::unique_ptr<FlavourHistograms<double>> discrNoCutEffic, discrCutEfficScan;

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
  TH1F * effVersusDiscr_pu   ;

  // the corresponding output histograms (flavour-eff vs. b-efficiency)

  // binning for output histograms
  int nBinOutput;
  double startOutput;
  double endOutput;

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
};

#endif
