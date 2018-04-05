#ifndef DQMOffline_RecoB_TagCorrelationPlotter_h
#define DQMOffline_RecoB_TagCorrelationPlotter_h

#include <string>

#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DQMOffline/RecoB/interface/EffPurFromHistos2D.h"

class TagCorrelationPlotter: public BaseBTagPlotter {
  public:
    TagCorrelationPlotter(const std::string& tagName1, const std::string& tagName2, const EtaPtBin& etaPtBin,
	                  const edm::ParameterSet& pSet, unsigned int mc, bool doCTagPlots, bool finalize, DQMStore::IBooker & ibook);

    ~TagCorrelationPlotter() override;

    void finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) override;

    void epsPlot(const std::string & name) override;
    void psPlot(const std::string& name) override {}

    void analyzeTags(const reco::JetTag& jetTag1, const reco::JetTag& jetTag2, int jetFlavour, float w=1);
    void analyzeTags(float discr1, float discr2, int jetFlavour, float w=1);

  protected:
    double lowerBound1_, lowerBound2_;
    double upperBound1_, upperBound2_;
    int nBinEffPur_;
    double startEffPur_;
    double endEffPur_;
    bool createProfile_;

    std::vector<double> fixedEff_;

    unsigned int mcPlots_;
    bool doCTagPlots_;
    bool finalize_;

    std::unique_ptr<FlavourHistograms2D<double, double>> correlationHisto_;

    std::unique_ptr<EffPurFromHistos2D> effPurFromHistos2D;
};

#endif
