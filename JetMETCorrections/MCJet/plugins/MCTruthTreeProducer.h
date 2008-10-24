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
template<class Jet>
class MCTruthTreeProducer : public edm::EDAnalyzer 
{
  public:
    typedef std::vector<Jet> JetCollection;
    explicit MCTruthTreeProducer(edm::ParameterSet const& cfg);
    virtual void beginJob(edm::EventSetup const& iSetup);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();
    MCTruthTreeProducer();

  private:
    std::string histogramFile_;
    std::string jets_;
    std::string genjets_;
    TFile* m_file_;
    TTree* mcTruthTree_;
    float ptCalo_,ptGen_,ptHat_,dR_,etaCalo_,etaGen_,phiCalo_,phiGen_;
    int rank_;
};
}
#include "JetMETCorrections/MCJet/plugins/MCTruthTreeProducer.icc"
#endif
