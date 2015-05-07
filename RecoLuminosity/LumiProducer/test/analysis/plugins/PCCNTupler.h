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
    void init();
    void fillEvent();
    void fillTracks();
    void fillRecHits();
    void fillVertex();
    void fillDigis();
    
    void bpixNames(const DetId &pID, int &DBlayer, int &DBladder, int &DBmodule);
    void fpixNames(const DetId &pID, int &DBdisk, int &DBblade, int &DBpanel, int &DBplaquette);
    
    void onlineRocColRow(const DetId &pID, int offlineRow, int offlineCol, int &roc, int &col, int &row);
    void isPixelTrack(const edm::Ref<std::vector<Trajectory> > &refTraj, bool &isBpixtrack, bool &isFpixtrack);


  private:
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> >  pixelToken;
    edm::EDGetTokenT<reco::VertexCollection> recoVtxToken;
    edm::EDGetTokenT<std::vector< PileupSummaryInfo> > pileUpToken;
    
    int             fVerbose; 
    std::string     fRootFileName; 
    std::string     fGlobalTag, fType;
    int             fDumpAllEvents;
    edm::InputTag   fPrimaryVertexCollectionLabel;
    edm::InputTag   fPixelClusterLabel;
    edm::InputTag   fPileUpInfoLabel;

    bool fAccessSimHitInfo;

    TFile *fFile; 
    TTree *fTree;

    std::map<int, int>     fFEDID; 

    // -- general stuff
    unsigned int fRun, fEvent, fLumiBlock; 
    int          fBX, fOrbit; 
    unsigned int fTimeLo, fTimeHi; 
 
    float fBz;
    int fFED1, fFED2; 

    // -- clusters
    static const int CLUSTERMAX = 100000; 
    static const int DGPERCLMAX = 100;  
    static const int TKPERCLMAX = 100;  

    // module information
    int nDeadModules;
    uint32_t  deadModules[6]; 
    int nDeadPrint; 

    // saving events per LS, LN or event
    std::string saveType = "LumiSect"; // LumiSect or LumiNib or Event
    bool saveAndReset;
    bool sameEvent;
    bool sameLumiNib;
    bool sameLumiSect;
    bool firstEvent;
    std::string sampleType="MC"; // MC or DATA

     // Lumi stuff
    TTree * tree;
    int runNo;
    int LSNo=-99;    // set to indicate first pass of analyze method
    int LNNo=-99;    // set to indicate first pass of analyze method
    int eventNo=-99; // set to indicate first pass of analyze method
    int bxNo=-99;    // local variable only
    
    std::pair<int,int> bxModKey;    // local variable only
   
    int eventCounter=0;
    int totalEvents;
    
    bool includeVertexInformation, includePixels;
    int nVtx, nTrk, ndof;
    std::map<int,int> nGoodVtx;
    std::map<std::pair<int,int>,int> nPixelClusters;
    std::map<std::pair<int,int>,int> nClusters;
    std::map<int,int> layers;

    TH1F* pileup;

    float xV, yV, zV, chi2;
    UInt_t timeStamp;
    int nPrint;
    std::map<int,int> BXNo;
    edm::InputTag vertexTags_; //used to select what vertices to read from configuration file 
    edm::InputTag vertexBSTags_; //used to select what vertices with BS correction 

};


#endif
