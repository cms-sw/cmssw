#ifndef DQMOffline_RecoB_TagCorrelationPlotter_h
#define DQMOffline_RecoB_TagCorrelationPlotter_h

#include <string>

#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

class TagCorrelationPlotter : public BaseBTagPlotter {
  public:
    TagCorrelationPlotter(const std::string& tagName1, const std::string& tagName2, const EtaPtBin& etaPtBin,
                          const edm::ParameterSet& pSet, const unsigned int& mc , const bool& update);

    virtual ~TagCorrelationPlotter();

    void finalize() {}
    void epsPlot(const std::string& name) {}
    void psPlot (const std::string& name) {}

    void analyzeTags(const reco::JetTag& jetTag1, const reco::JetTag& jetTag2, const int& jetFlavour);
    void analyzeTags(const reco::JetTag& jetTag1, const reco::JetTag& jetTag2, const int& jetFlavour, const float & w);

    void analyzeTags(const float& discr1, const float& discr2, const int& jetFlavour);
    void analyzeTags(const float& discr1, const float& discr2, const int& jetFlavour, const float & w);

  protected:
    double lowerBound1_, lowerBound2_;
    double upperBound1_, upperBound2_;
    bool createProfile_;

    FlavourHistograms2D<double, double>* correlationHisto_;
};

#endif
