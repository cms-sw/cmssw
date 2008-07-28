#ifndef MCTRUTHTREEPRODUCER_H
#define MCTRUTHTREEPRODUCER_H

#include "TTree.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace cms
{
class MCTruthTreeProducer : public edm::EDAnalyzer 
{
  public:

    explicit MCTruthTreeProducer(edm::ParameterSet const& cfg);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();
    MCTruthTreeProducer();

  private:
    std::string histogramFile_;
    std::string calojets_;
    std::string genjets_;
    bool PFJet_;
    TFile* m_file_;
    TTree* mcTruthTree_;
    float ptCalo_,ptGen_,dR_,etaCalo_,etaGen_,phiCalo_,phiGen_;
    int rank_;
};
}
#endif
