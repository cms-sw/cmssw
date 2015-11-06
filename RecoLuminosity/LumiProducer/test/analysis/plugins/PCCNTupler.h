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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TObject.h"

using namespace reco;

class TObject;
class TTree;
class TH1D;
class TFile;
class RectangularPixelTopology;
class DetId; 

class PCCNTupler : public edm::EDAnalyzer {
  public:
    explicit PCCNTupler(const edm::ParameterSet& ps);
    virtual ~PCCNTupler();
    virtual void beginJob();
    virtual void beginRun(const edm::Run &, const edm::EventSetup &);
    virtual void endRun(edm::Run const&, edm::EventSetup const&);
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
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
    int             fVerbose; 
    std::string     fRootFileName; 
    std::string     fGlobalTag, fType;
    int             fDumpAllEvents;
    edm::InputTag   fPrimaryVertexCollectionLabel;
    edm::InputTag   fMuonCollectionLabel, fTrackCollectionLabel, fTrajectoryInputLabel, fPixelClusterLabel, fPixelRecHitLabel;
    std::string     fHLTProcessName; 

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

    HLTConfigProvider fHltConfig;  
    bool fValidHLTConfig;

     // Lumi stuff
    TTree * tree;
    int runNo, eventNo;
    int LSNo=-99; // set to indicate first pass of analyze method
    
    int eventCounter=0;
    int totalEvents;
    
    bool includeVertexInformation, includeTracks, includePixels;
    int nVtx, nTrk, ndof;
    std::map<int,int> nGoodVtx;
    int nGeneralTracks, n1GeV, n2GeV;
    std::map<int,int> nPixelClusters;
    std::map<int,int> nClusters;
    std::map<int,int> layers;

    float xV, yV, zV, chi2;
    UInt_t timeStamp;
    int nPrint;
    std::map<int,int> BXNo;
    edm::InputTag vertexTags_; //used to select what vertices to read from configuration file 
    edm::InputTag vertexBSTags_; //used to select what vertices with BS correction 

};

#endif
