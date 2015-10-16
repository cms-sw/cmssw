#ifndef PCCNTupler_h
#define PCCNTupler_h

/** \class PCCNTupler
 * ----------------------------------------------------------------------
 * PCCNTupler
 * ---------
 * Summary: The pixel clusters are summed per pixel module per lumi
 *          lumi section.
 *
 * ----------------------------------------------------------------------
 * Author:  Chris Palmer
 * ----------------------------------------------------------------------
 *
 *
 ************************************************************/

#include <map>

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
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
class RectangularPixelTopology;
class DetId; 


class PCCNTupler : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchLuminosityBlocks> {
  public:
    PCCNTupler(const edm::ParameterSet&);
    virtual ~PCCNTupler();
    virtual void beginJob() override;
    virtual void endJob() override;
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;


  protected:
    void Reset();
    void SaveAndReset();
    void ComputeMeanAndMeanError();

  private:
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> >  pixelToken;
    edm::EDGetTokenT<reco::VertexCollection> recoVtxToken;
    edm::EDGetTokenT<std::vector< PileupSummaryInfo> > pileUpToken;
    
    edm::InputTag   fPrimaryVertexCollectionLabel;
    edm::InputTag   fPixelClusterLabel;
    edm::InputTag   fPileUpInfoLabel;
  
    static const int MAX_VERTICES=200;

    // saving events per LS, LN or event
    std::string saveType = "LumiSect"; // LumiSect or LumiNib or Event
    std::string sampleType="MC"; // MC or DATA
    bool saveAndReset;
    bool sameEvent;
    bool sameLumiNib;
    bool sameLumiSect;
    bool firstEvent;

     // Lumi stuff
    TTree * tree;
    int run;
    int LS=-99;    // set to indicate first pass of analyze method
    int LN=-99;    // set to indicate first pass of analyze method
    int event=-99; // set to indicate first pass of analyze method
    int bunchCrossing=-99;    // local variable only
    int orbit=-99;
    
    std::pair<int,int> bxModKey;    // local variable only
   
    int eventCounter=0;
    int totalEvents;
    
    bool includeVertexInformation;
    bool includePixels;

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
    std::map<std::pair<int,int>,int> nPixelClusters;
    std::map<std::pair<int,int>,int> nClusters;
    std::map<int,int> layers;

    std::map<std::pair<int,int>,float> meanPixelClusters;
    std::map<std::pair<int,int>,float> meanPixelClustersError;
    
    TH1F* pileup;

    UInt_t timeStamp_begin;
    UInt_t timeStamp_local;
    UInt_t timeStamp_end;
    std::map<int,int> BXNo;

};


#endif
