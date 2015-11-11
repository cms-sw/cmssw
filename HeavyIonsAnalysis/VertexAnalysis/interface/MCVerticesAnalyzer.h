#ifndef MCVerticesAnalyzer_h
#define MCVerticesAnalyzer_h

/** \class MCVerticesAnalyzer
 * ----------------------------------------------------------------------
 * MCVerticesAnalyzer
 * ---------
 * Summary: pileup vertex analyzer
 *
 * ----------------------------------------------------------------------
 * Author:  Marta Verweij
 * ----------------------------------------------------------------------
 ************************************************************/

#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TObject.h"
#include "TH1F.h"


using namespace reco;

class TObject;
class TTree;
class TH1D;
class TFile;


class MCVerticesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchLuminosityBlocks> {
  public:
    MCVerticesAnalyzer(const edm::ParameterSet&);
    virtual ~MCVerticesAnalyzer();
    virtual void beginJob() override;
    virtual void endJob() override;
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;


  protected:
    void Reset();
    
  private:
    edm::EDGetTokenT<reco::VertexCollection> recoVtxToken;
    edm::EDGetTokenT<std::vector< PileupSummaryInfo> > pileUpToken;
    
    edm::InputTag   fPrimaryVertexCollectionLabel;
    edm::InputTag   fPileUpInfoLabel;

    static const int MAX_VERTICES=200;

    int eventCounter=0;
    int totalEvents;

    TTree *tree;
    int nPU;
    int nVtx;
    int vtx_nTrk[MAX_VERTICES];
    int vtx_ndof[MAX_VERTICES];
    float vtx_x[MAX_VERTICES];
    float vtx_y[MAX_VERTICES];
    float vtx_z[MAX_VERTICES];
    float vtx_xError[MAX_VERTICES];
    float vtx_yError[MAX_VERTICES];
    float vtx_zError[MAX_VERTICES];
    float vtx_chi2[MAX_VERTICES];
    float vtx_normchi2[MAX_VERTICES];
    bool vtx_isValid[MAX_VERTICES];
    bool vtx_isFake[MAX_VERTICES];
    bool vtx_isGood[MAX_VERTICES];

    std::map<int,int> nGoodVtx;
    std::map<int,int> nValidVtx;
       
    TH1F* pileup;
};

#endif
