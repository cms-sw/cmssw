#ifndef TrackProbabilityTagPlotter_H
#define TrackProbabilityTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackProbabilityTagPlotter: public BaseTagInfoPlotter {


 public:

  TrackProbabilityTagPlotter(const std::string & tagName, const EtaPtBin & etaPtBin,
                  const edm::ParameterSet& pSet, 
                  const unsigned int& mc, const bool& wf, DQMStore::IBooker & ibook);

  ~TrackProbabilityTagPlotter() override;

  void analyzeTag(const reco::BaseTagInfo * tagInfo, double jec, int jetFlavour, float w=1) override;

  void finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) override;

  void epsPlot(const std::string & name) override;

  void psPlot(const std::string & name) override;

 private:

  int nBinEffPur_;
  double startEffPur_; 
  double endEffPur_; 
  std::vector< std::unique_ptr<FlavourHistograms<double>> > tkcntHistosSig3D_;
  std::vector< std::unique_ptr<FlavourHistograms<double>> > tkcntHistosSig2D_;
  std::vector< std::unique_ptr<EffPurFromHistos> > effPurFromHistos_;
  unsigned int mcPlots_;  
  bool willFinalize_;
};

#endif
