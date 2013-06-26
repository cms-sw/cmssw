#ifndef PF_MCTRUTH_TREE_PRODUCER_H
#define PF_MCTRUTH_TREE_PRODUCER_H

#include "TTree.h"
#include "TFile.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

//namespace cms
//{
class PFMCTruthTreeProducer : public edm::EDAnalyzer 
{
  public:
    explicit PFMCTruthTreeProducer(edm::ParameterSet const& cfg);
    virtual void beginJob();
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();
    ~PFMCTruthTreeProducer();

  private:
    std::string histogramFile_;
    std::string jets_;
    std::string genjets_;
    TFile* file_;
    TTree* mcTruthTree_;
    float ptJet_,chfJet_,nhfJet_,cemfJet_,nemfJet_,ptGen_,ptHat_,dR_,etaJet_,etaGen_,phiJet_,phiGen_;
    int rank_,cmultiJet_,nmultiJet_;
};
//}

#endif
