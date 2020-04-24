#ifndef TrackCountingTagPlotter_H
#define TrackCountingTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackCountingTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackCountingTagPlotter(const std::string & tagName, const EtaPtBin & etaPtBin,
               const edm::ParameterSet& pSet, 
               unsigned int mc, bool willfinalize, DQMStore::IBooker & ibook);

  ~TrackCountingTagPlotter () override;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, double jec, int jetFlavour, float w/*=1*/) override;

  void finalize (DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) override;

  void epsPlot(const std::string & name) override;

  void psPlot(const std::string & name) override;

 private:
  unsigned int mcPlots_;
  int nBinEffPur_;
  double startEffPur_; 
  double endEffPur_; 

  bool willFinalize_;

  std::vector< std::unique_ptr<FlavourHistograms<double>> > tkcntHistosSig3D;
  std::vector< std::unique_ptr<FlavourHistograms<double>> > tkcntHistosSig2D;
  std::unique_ptr<FlavourHistograms<int>> trkNbr3D, trkNbr2D;
  double lowerIPSBound, upperIPSBound;

  std::vector< std::unique_ptr<EffPurFromHistos> > effPurFromHistos;
};

#endif
