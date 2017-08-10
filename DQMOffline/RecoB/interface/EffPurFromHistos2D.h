#ifndef EffPurFromHistos2D_H
#define EffPurFromHistos2D_H

#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TH2F.h"
#include "TCanvas.h"

#include <string>

class EffPurFromHistos2D {

 public:

  EffPurFromHistos2D(const std::string & ext, TH2F * h_d, TH2F * h_u,
                        TH2F * h_s, TH2F * h_c, TH2F * h_b, TH2F * h_g, TH2F * h_ni,
                        TH2F * h_dus, TH2F * h_dusg, TH2F * h_pu,
                        const std::string& label, unsigned int mc,
                        int nBinX = 100, double startOX = 0.05, double endOX = 1.05);

  EffPurFromHistos2D(const FlavourHistograms2D<double, double>& dDiscriminatorFC,  
                        const std::string& label, unsigned int mc, 
                        DQMStore::IBooker & ibook,
                        int nBinX = 100, double startOX = 0.05, double endOX = 1.05);

  ~EffPurFromHistos2D();

  // do the computation
  void compute(DQMStore::IBooker & ibook, std::vector<double> fixedEff);
   
  void epsPlot(const std::string & name);
  void psPlot(const std::string & name);

  void plot(TPad * theCanvas = 0);
  void plot(const std::string & name, const std::string & ext);

  FlavourHistograms2D<double,double>& discriminatorNoCutEffic() const { return *discrNoCutEffic; }
  FlavourHistograms2D<double,double>& discriminatorCutEfficScan() const { return *discrCutEfficScan; }

  bool doCTagPlots(bool Ctag) { doCTagPlots_ = Ctag; return doCTagPlots_; }
 
 private:

  // consistency check(same binning)
  void check();
  bool fromDiscriminatorDistr;

  unsigned int mcPlots_;
  bool doCTagPlots_;
  std::string label_;
  // the string for the histo name extension
  std::string histoExtension;

  std::unique_ptr< FlavourHistograms2D<double, double> > discrNoCutEffic, discrCutEfficScan;

  // the input histograms(efficiency versus discriminator cut)
  // IMPORTANT: IT'S ASSUMED THAT ALL HISTOS HAVE THE SAME BINNING!!
  //(can in principle be relaxed by checking explicitely for the discriminator value
  //  instead of bin index)
  TH2F * effVersusDiscr_d   ;
  TH2F * effVersusDiscr_u   ;
  TH2F * effVersusDiscr_s   ;
  TH2F * effVersusDiscr_c   ;
  TH2F * effVersusDiscr_b   ;
  TH2F * effVersusDiscr_g   ;
  TH2F * effVersusDiscr_ni  ;
  TH2F * effVersusDiscr_dus ;
  TH2F * effVersusDiscr_dusg;
  TH2F * effVersusDiscr_pu  ;

  // the corresponding output histograms(flavour-eff vs. b-efficiency)

  // binning for output histograms
  int nBinOutputX;
  double startOutputX; 
  double endOutputX;
  int nBinOutputY; 
  double startOutputY; 
  double endOutputY;
  
  std::vector<MonitorElement*> X_vs_Y_eff_at_fixedZeff;
};

#endif
