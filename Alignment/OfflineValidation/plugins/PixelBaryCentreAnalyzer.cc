/** \class PixelBaryCentreAnalyzer
 *  The analyer works as the following :
 *  - Read global tracker position from global tag
 *  - Read tracker alignment constants from different ESsource with different labels
 *  - Calculate barycentres for different pixel substructures using global tracker position and alignment constants and store them in trees, one for each ESsource label.
 *
 *  Python script plotBaryCentre_VS_BeamSpot.py under script dir is used to plot barycentres from alignment constants used in Prompt-Reco, End-of-Year Rereco and so-called Run-2 (Ultra)Legacy Rereco. Options of the plotting script can be found from the helper in the script.
 *
 *  $Date: 2021/01/05 $
 *  $Revision: 1.0 $
 *  \author Tongguang Cheng - Beihang University <tongguang.cheng@cern.ch>
 *
*/

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// Phase-1 Pixel
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"

// pixel quality
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
// global postion
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
// tracker alignment
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/Alignment/interface/Alignments.h"
// beamspot
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

// Point and Vector
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

// TFileService
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ROOT
#include "TTree.h"
#include "TString.h"

//
// class declaration
//

class PixelBaryCentreAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PixelBaryCentreAnalyzer(const edm::ParameterSet&);
  ~PixelBaryCentreAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  struct SimplePoint {
    float x, y, z;
    SimplePoint(const GlobalPoint& p) : x(p.x()), y(p.y()), z(p.z()){};
    SimplePoint() : x(0), y(0), z(0){};
  };
  static const unsigned int nPixelLayers = 4;
  static const unsigned int nPixelDiscs = 3;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void initBC();
  void initBS();

  bool usePixelQuality_;
  int phase_;

  // ----------member data ---------------------------
  edm::ESWatcher<BeamSpotObjectsRcd> watcherBS_;
  edm::ESWatcher<TrackerAlignmentRcd> watcherTkAlign_;

  // labels of TkAlign tags
  std::vector<std::string> bcLabels_;
  // labels of beamspot tags
  std::vector<std::string> bsLabels_;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::ESGetToken<SiPixelQuality, SiPixelQualityFromDbRcd> siPixelQualityToken_;

  edm::ESGetToken<Alignments, GlobalPositionRcd> gprToken_;
  std::map<std::string, edm::ESGetToken<Alignments, TrackerAlignmentRcd>> tkAlignTokens_;
  std::map<std::string, edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd>> bsTokens_;

  // tree content
  int run_;
  int ls_;

  GlobalPoint BS_;

  GlobalPoint PIX_, BPIX_, FPIX_;
  GlobalPoint BPIX_Flipped_, BPIX_NonFlipped_, BPIX_DiffFlippedNonFlipped_;

  GlobalPoint BPIXLayer_[nPixelLayers];
  GlobalPoint BPIXLayer_Flipped_[nPixelLayers];
  GlobalPoint BPIXLayer_NonFlipped_[nPixelLayers];
  GlobalPoint BPIXLayer_DiffFlippedNonFlipped_[nPixelLayers];

  GlobalPoint FPIX_plus_, FPIX_minus_;
  GlobalPoint FPIXDisks_plus_[nPixelDiscs];
  GlobalPoint FPIXDisks_minus_[nPixelDiscs];

  SimplePoint vBS_;

  SimplePoint vPIX_, vBPIX_, vFPIX_;
  SimplePoint vBPIX_Flipped_, vBPIX_NonFlipped_, vBPIX_DiffFlippedNonFlipped_;

  SimplePoint vBPIXLayer_[nPixelLayers];
  SimplePoint vBPIXLayer_Flipped_[nPixelLayers];
  SimplePoint vBPIXLayer_NonFlipped_[nPixelLayers];
  SimplePoint vBPIXLayer_DiffFlippedNonFlipped_[nPixelLayers];

  SimplePoint vFPIX_plus_, vFPIX_minus_;
  SimplePoint vFPIXDisks_plus_[nPixelDiscs];
  SimplePoint vFPIXDisks_minus_[nPixelDiscs];

  edm::Service<TFileService> tFileService;
  std::map<std::string, TTree*> bcTrees_;
  std::map<std::string, TTree*> bsTrees_;
};

//
// constructors and destructor
//
PixelBaryCentreAnalyzer::PixelBaryCentreAnalyzer(const edm::ParameterSet& iConfig)
    : usePixelQuality_(iConfig.getUntrackedParameter<bool>("usePixelQuality")),
      bcLabels_(iConfig.getUntrackedParameter<std::vector<std::string>>("tkAlignLabels")),
      bsLabels_(iConfig.getUntrackedParameter<std::vector<std::string>>("beamSpotLabels")),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      siPixelQualityToken_(esConsumes<SiPixelQuality, SiPixelQualityFromDbRcd>()),
      gprToken_(esConsumes<Alignments, GlobalPositionRcd>()) {
  for (const auto& label : bcLabels_) {
    bcTrees_[label] = nullptr;
    tkAlignTokens_[label] = esConsumes<Alignments, TrackerAlignmentRcd>(edm::ESInputTag{"", label});
  }

  for (const auto& label : bsLabels_) {
    bsTrees_[label] = nullptr;
    bsTokens_[label] = esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>(edm::ESInputTag{"", label});
  }

  usesResource("TFileService");
}

//
// member functions
//

void PixelBaryCentreAnalyzer::initBS() {
  double dummy_float = 999999.0;

  BS_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  vBS_ = SimplePoint(BS_);
}

void PixelBaryCentreAnalyzer::initBC() {
  // init to large number (unreasonable number) not zero
  double dummy_float = 999999.0;

  PIX_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  BPIX_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  FPIX_ = GlobalPoint(dummy_float, dummy_float, dummy_float);

  BPIX_Flipped_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  BPIX_NonFlipped_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  BPIX_DiffFlippedNonFlipped_ = GlobalPoint(dummy_float, dummy_float, dummy_float);

  FPIX_plus_ = GlobalPoint(dummy_float, dummy_float, dummy_float);
  FPIX_minus_ = GlobalPoint(dummy_float, dummy_float, dummy_float);

  for (unsigned int i = 0; i < nPixelLayers; i++) {
    BPIXLayer_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
    BPIXLayer_Flipped_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
    BPIXLayer_NonFlipped_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
    BPIXLayer_DiffFlippedNonFlipped_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
  }

  for (unsigned int i = 0; i < nPixelDiscs; i++) {
    FPIXDisks_plus_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
    FPIXDisks_minus_[i] = GlobalPoint(dummy_float, dummy_float, dummy_float);
  }

  vPIX_ = SimplePoint(PIX_);
  vBPIX_ = SimplePoint(BPIX_);
  vFPIX_ = SimplePoint(FPIX_);

  vBPIX_Flipped_ = SimplePoint(BPIX_Flipped_);
  vBPIX_NonFlipped_ = SimplePoint(BPIX_NonFlipped_);
  vBPIX_DiffFlippedNonFlipped_ = SimplePoint(BPIX_DiffFlippedNonFlipped_);

  vFPIX_plus_ = SimplePoint(FPIX_plus_);
  vFPIX_minus_ = SimplePoint(FPIX_minus_);

  for (unsigned int i = 0; i < nPixelLayers; i++) {
    vBPIXLayer_[i] = SimplePoint(BPIXLayer_[i]);
    vBPIXLayer_Flipped_[i] = SimplePoint(BPIXLayer_Flipped_[i]);
    vBPIXLayer_NonFlipped_[i] = SimplePoint(BPIXLayer_NonFlipped_[i]);
    vBPIXLayer_DiffFlippedNonFlipped_[i] = SimplePoint(BPIXLayer_DiffFlippedNonFlipped_[i]);
  }

  for (unsigned int i = 0; i < nPixelDiscs; i++) {
    vFPIXDisks_plus_[i] = SimplePoint(FPIXDisks_plus_[i]);
    vFPIXDisks_minus_[i] = SimplePoint(FPIXDisks_minus_[i]);
  }
}

// ------------ method called for each event  ------------
void PixelBaryCentreAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool prepareTkAlign = false;
  bool prepareBS = false;

  // ES watcher can noly run once in the same event,
  // otherwise it will turn false whatsoever because the condition doesn't change in the second time call.
  if (watcherTkAlign_.check(iSetup))
    prepareTkAlign = true;
  if (watcherBS_.check(iSetup))
    prepareBS = true;

  if (!prepareTkAlign && !prepareBS)
    return;

  run_ = iEvent.id().run();
  ls_ = iEvent.id().luminosityBlock();

  if (prepareTkAlign) {  // check for new IOV for TKAlign

    phase_ = -1;

    const TrackerGeometry* tkGeo = &iSetup.getData(trackerGeometryToken_);
    const TrackerTopology* tkTopo = &iSetup.getData(trackerTopologyToken_);

    if (tkGeo->isThere(GeomDetEnumerators::PixelBarrel) && tkGeo->isThere(GeomDetEnumerators::PixelEndcap))
      phase_ = 0;
    else if (tkGeo->isThere(GeomDetEnumerators::P1PXB) && tkGeo->isThere(GeomDetEnumerators::P1PXEC))
      phase_ = 1;

    // pixel quality
    const SiPixelQuality* badPixelInfo = &iSetup.getData(siPixelQualityToken_);

    // Tracker global position
    const Alignments* globalAlignments = &iSetup.getData(gprToken_);
    std::unique_ptr<const Alignments> globalPositions = std::make_unique<Alignments>(*globalAlignments);
    const AlignTransform& globalCoordinates = align::DetectorGlobalPosition(*globalPositions, DetId(DetId::Tracker));
    GlobalVector globalTkPosition(
        globalCoordinates.translation().x(), globalCoordinates.translation().y(), globalCoordinates.translation().z());

    // loop over bclabels
    for (const auto& label : bcLabels_) {
      // init tree content
      PixelBaryCentreAnalyzer::initBC();

      // Get TkAlign from EventSetup:
      const Alignments* alignments = &iSetup.getData(tkAlignTokens_[label]);
      std::vector<AlignTransform> tkAlignments = alignments->m_align;

      // PIX
      GlobalVector barycentre_PIX(0.0, 0.0, 0.0);
      // BPIX
      GlobalVector barycentre_BPIX(0.0, 0.0, 0.0);
      float nmodules_BPIX(0.);
      // FPIX
      GlobalVector barycentre_FPIX(0.0, 0.0, 0.0);
      float nmodules_FPIX(0.);

      // Per-layer/ladder barycentre for BPIX
      std::map<int, std::map<int, float>> nmodules_bpix;           // layer-ladder
      std::map<int, std::map<int, GlobalVector>> barycentre_bpix;  // layer-ladder

      // Per-disk/ring barycentre for FPIX
      std::map<int, std::map<int, float>> nmodules_fpix;           // disk-ring
      std::map<int, std::map<int, GlobalVector>> barycentre_fpix;  // disk-ring

      // Loop over tracker module
      for (const auto& ali : tkAlignments) {
        //DetId
        const DetId& detId = DetId(ali.rawId());
        // remove bad module
        if (usePixelQuality_ && badPixelInfo->IsModuleBad(detId))
          continue;

        // alignment for a given module
        GlobalVector ali_translation(ali.translation().x(), ali.translation().y(), ali.translation().z());

        int subid = DetId(detId).subdetId();
        // BPIX
        if (subid == PixelSubdetector::PixelBarrel) {
          nmodules_BPIX += 1;
          barycentre_BPIX += ali_translation;
          barycentre_PIX += ali_translation;

          int layer = tkTopo->pxbLayer(detId);
          int ladder = tkTopo->pxbLadder(detId);
          nmodules_bpix[layer][ladder] += 1;
          barycentre_bpix[layer][ladder] += ali_translation;

        }  // BPIX

        // FPIX
        if (subid == PixelSubdetector::PixelEndcap) {
          nmodules_FPIX += 1;
          barycentre_FPIX += ali_translation;
          barycentre_PIX += ali_translation;

          int disk = tkTopo->pxfDisk(detId);
          int quadrant = PixelEndcapName(detId, tkTopo, phase_).halfCylinder();
          if (quadrant < 3)
            disk *= -1;

          int ring = -9999;
          if (phase_ == 0) {
            ring = 1 + (tkTopo->pxfPanel(detId) + tkTopo->pxfModule(detId.rawId()) > 3);
          } else if (phase_ == 1) {
            ring = PixelEndcapName(detId, tkTopo, phase_).ringName();
          }

          nmodules_fpix[disk][ring] += 1;
          barycentre_fpix[disk][ring] += ali_translation;

        }  // FPIX

      }  // loop over tracker module

      //PIX
      float nmodules_PIX = nmodules_BPIX + nmodules_FPIX;
      barycentre_PIX *= (1.0 / nmodules_PIX);
      barycentre_PIX += globalTkPosition;
      PIX_ = GlobalPoint(barycentre_PIX.x(), barycentre_PIX.y(), barycentre_PIX.z());
      vPIX_ = SimplePoint(PIX_);

      //BPIX
      barycentre_BPIX *= (1.0 / nmodules_BPIX);
      barycentre_BPIX += globalTkPosition;
      BPIX_ = GlobalPoint(barycentre_BPIX.x(), barycentre_BPIX.y(), barycentre_BPIX.z());
      vBPIX_ = SimplePoint(BPIX_);
      //FPIX
      barycentre_FPIX *= (1.0 / nmodules_FPIX);
      barycentre_FPIX += globalTkPosition;
      FPIX_ = GlobalPoint(barycentre_FPIX.x(), barycentre_FPIX.y(), barycentre_FPIX.z());
      vFPIX_ = SimplePoint(FPIX_);
      // Pixel substructures

      // BPix barycentre per-layer/per-ladder
      // assuming each ladder has the same number of modules in the same layer
      // inner =  flipped; outer = non-flipped
      //
      // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
      // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
      //

      int nmodules_BPIX_Flipped = 0;
      int nmodules_BPIX_NonFlipped = 0;
      GlobalVector BPIX_Flipped(0.0, 0.0, 0.0);
      GlobalVector BPIX_NonFlipped(0.0, 0.0, 0.0);

      // loop over layers
      for (std::map<int, std::map<int, GlobalVector>>::iterator il = barycentre_bpix.begin();
           il != barycentre_bpix.end();
           ++il) {
        int layer = il->first;

        int nmodulesLayer = 0;
        int nmodulesLayer_Flipped = 0;
        int nmodulesLayer_NonFlipped = 0;
        GlobalVector BPIXLayer(0.0, 0.0, 0.0);
        GlobalVector BPIXLayer_Flipped(0.0, 0.0, 0.0);
        GlobalVector BPIXLayer_NonFlipped(0.0, 0.0, 0.0);

        // loop over ladder
        std::map<int, GlobalVector> barycentreLayer = barycentre_bpix[layer];
        for (std::map<int, GlobalVector>::iterator it = barycentreLayer.begin(); it != barycentreLayer.end(); ++it) {
          int ladder = it->first;
          //BPIXLayerLadder_[layer][ladder] = (1.0/nmodules[layer][ladder])*barycentreLayer[ladder] + globalTkPosition;

          nmodulesLayer += nmodules_bpix[layer][ladder];
          BPIXLayer += barycentreLayer[ladder];

          // Phase-1
          //
          // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
          if (phase_ == 1) {
            if (layer != 4) {  // layer 1-3

              if (ladder % 2 != 0) {  // odd ladder = outer ladder = unflipped
                nmodulesLayer_NonFlipped += nmodules_bpix[layer][ladder];
                BPIXLayer_NonFlipped += barycentreLayer[ladder];
              } else {  // even ladder = inner ladder = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              }
            } else {  // layer-4

              if (ladder % 2 != 0) {  // odd ladder = inner = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              } else {  //even ladder = outer ladder  = unflipped
                nmodulesLayer_NonFlipped += nmodules_bpix[layer][ladder];
                BPIXLayer_NonFlipped += barycentreLayer[ladder];
              }
            }

          }  // phase-1

          // Phase-0
          //
          // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
          if (phase_ == 0) {
            if (layer == 2) {  // layer-2

              if (ladder % 2 != 0) {  // odd ladder = inner = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              } else {
                nmodulesLayer_NonFlipped += nmodules_bpix[layer][ladder];
                BPIXLayer_NonFlipped += barycentreLayer[ladder];
              }
            } else {  // layer-1,3

              if (ladder % 2 == 0) {  // even ladder = inner = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              } else {  // odd ladder = outer = non-flipped
                nmodulesLayer_NonFlipped += nmodules_bpix[layer][ladder];
                BPIXLayer_NonFlipped += barycentreLayer[ladder];
              }
            }

          }  // phase-0

        }  //loop over ladders

        // total BPIX flipped/non-flipped
        BPIX_Flipped += BPIXLayer_Flipped;
        BPIX_NonFlipped += BPIXLayer_NonFlipped;
        nmodules_BPIX_Flipped += nmodulesLayer_Flipped;
        nmodules_BPIX_NonFlipped += nmodulesLayer_NonFlipped;

        //BPIX per-layer
        BPIXLayer *= (1.0 / nmodulesLayer);
        BPIXLayer += globalTkPosition;
        BPIXLayer_Flipped *= (1.0 / nmodulesLayer_Flipped);
        BPIXLayer_Flipped += globalTkPosition;
        BPIXLayer_NonFlipped *= (1.0 / nmodulesLayer_NonFlipped);
        BPIXLayer_NonFlipped += globalTkPosition;

        BPIXLayer_[layer - 1] = GlobalPoint(BPIXLayer.x(), BPIXLayer.y(), BPIXLayer.z());
        vBPIXLayer_[layer - 1] = SimplePoint(BPIXLayer_[layer - 1]);
        BPIXLayer_Flipped_[layer - 1] =
            GlobalPoint(BPIXLayer_Flipped.x(), BPIXLayer_Flipped.y(), BPIXLayer_Flipped.z());
        vBPIXLayer_Flipped_[layer - 1] = SimplePoint(BPIXLayer_Flipped_[layer - 1]);
        BPIXLayer_NonFlipped_[layer - 1] =
            GlobalPoint(BPIXLayer_NonFlipped.x(), BPIXLayer_NonFlipped.y(), BPIXLayer_NonFlipped.z());
        vBPIXLayer_NonFlipped_[layer - 1] = SimplePoint(BPIXLayer_NonFlipped_[layer - 1]);
        BPIXLayer_DiffFlippedNonFlipped_[layer - 1] = GlobalPoint(BPIXLayer_Flipped.x() - BPIXLayer_NonFlipped.x(),
                                                                  BPIXLayer_Flipped.y() - BPIXLayer_NonFlipped.y(),
                                                                  BPIXLayer_Flipped.z() - BPIXLayer_NonFlipped.z());
        vBPIXLayer_DiffFlippedNonFlipped_[layer - 1] = SimplePoint(BPIXLayer_DiffFlippedNonFlipped_[layer - 1]);

      }  // loop over layers

      BPIX_Flipped *= (1.0 / nmodules_BPIX_Flipped);
      BPIX_Flipped += globalTkPosition;
      BPIX_Flipped_ = GlobalPoint(BPIX_Flipped.x(), BPIX_Flipped.y(), BPIX_Flipped.z());
      vBPIX_Flipped_ = SimplePoint(BPIX_Flipped_);
      BPIX_NonFlipped *= (1.0 / nmodules_BPIX_NonFlipped);
      BPIX_NonFlipped += globalTkPosition;
      BPIX_NonFlipped_ = GlobalPoint(BPIX_NonFlipped.x(), BPIX_NonFlipped.y(), BPIX_NonFlipped.z());
      vBPIX_NonFlipped_ = SimplePoint(BPIX_NonFlipped_);
      BPIX_DiffFlippedNonFlipped_ = GlobalPoint(BPIX_Flipped.x() - BPIX_NonFlipped.x(),
                                                BPIX_Flipped.y() - BPIX_NonFlipped.y(),
                                                BPIX_Flipped.z() - BPIX_NonFlipped.z());
      vBPIX_DiffFlippedNonFlipped_ = SimplePoint(BPIX_DiffFlippedNonFlipped_);

      // FPIX substructures per-(signed)disk/per-ring
      int nmodules_FPIX_plus = 0;
      int nmodules_FPIX_minus = 0;
      GlobalVector FPIX_plus(0.0, 0.0, 0.0);
      GlobalVector FPIX_minus(0.0, 0.0, 0.0);
      // loop over disks
      for (std::map<int, std::map<int, GlobalVector>>::iterator id = barycentre_fpix.begin();
           id != barycentre_fpix.end();
           ++id) {
        int disk = id->first;

        int nmodulesDisk = 0;
        GlobalVector FPIXDisk(0.0, 0.0, 0.0);

        std::map<int, GlobalVector> baryCentreDisk = id->second;
        for (std::map<int, GlobalVector>::iterator ir = baryCentreDisk.begin(); ir != baryCentreDisk.end();
             ++ir) {  // loop over rings
          int ring = ir->first;
          nmodulesDisk += nmodules_fpix[disk][ring];
          FPIXDisk += ir->second;
          if (disk > 0) {
            nmodules_FPIX_plus += nmodules_fpix[disk][ring];
            FPIX_plus += ir->second;
          }
          if (disk < 0) {
            nmodules_FPIX_minus += nmodules_fpix[disk][ring];
            FPIX_minus += ir->second;
          }

        }  // loop over rings

        FPIXDisk *= (1.0 / nmodulesDisk);
        FPIXDisk += globalTkPosition;

        if (disk > 0) {
          FPIXDisks_plus_[disk - 1] = GlobalPoint(FPIXDisk.x(), FPIXDisk.y(), FPIXDisk.z());
          vFPIXDisks_plus_[disk - 1] = SimplePoint(FPIXDisks_plus_[disk - 1]);
        }
        if (disk < 0) {
          FPIXDisks_minus_[-disk - 1] = GlobalPoint(FPIXDisk.x(), FPIXDisk.y(), FPIXDisk.z());
          vFPIXDisks_minus_[-disk - 1] = SimplePoint(FPIXDisks_minus_[-disk - 1]);
        }
      }  // loop over disks

      FPIX_plus *= (1.0 / nmodules_FPIX_plus);
      FPIX_plus += globalTkPosition;
      FPIX_plus_ = GlobalPoint(FPIX_plus.x(), FPIX_plus.y(), FPIX_plus.z());
      vFPIX_plus_ = SimplePoint(FPIX_plus_);
      FPIX_minus *= (1.0 / nmodules_FPIX_minus);
      FPIX_minus += globalTkPosition;
      FPIX_minus_ = GlobalPoint(FPIX_minus.x(), FPIX_minus.y(), FPIX_minus.z());
      vFPIX_minus_ = SimplePoint(FPIX_minus_);

      bcTrees_[label]->Fill();

    }  // bcLabels_

  }  // check for new IOV for TKAlign

  // beamspot
  if (prepareBS) {
    // loop over bsLabels_
    for (const auto& label : bsLabels_) {
      // init bstree content
      PixelBaryCentreAnalyzer::initBS();

      // Get BeamSpot from EventSetup
      const BeamSpotObjects* mybeamspot = &iSetup.getData(bsTokens_[label]);

      BS_ = GlobalPoint(mybeamspot->x(), mybeamspot->y(), mybeamspot->z());
      vBS_ = SimplePoint(BS_);

      bsTrees_[label]->Fill();
    }  // bsLabels_

  }  // check for new IOV for BS
}

// ------------ method called once each job just before starting event loop  ------------
void PixelBaryCentreAnalyzer::beginJob() {
  // init bc bs trees
  for (const auto& label : bsLabels_) {
    std::string treeName = "BeamSpot";
    if (!label.empty())
      treeName = "BeamSpot_";
    treeName += label;

    bsTrees_[label] = tFileService->make<TTree>(TString(treeName), "PixelBarycentre analyzer ntuple");

    bsTrees_[label]->Branch("run", &run_, "run/I");
    bsTrees_[label]->Branch("ls", &ls_, "ls/I");

    bsTrees_[label]->Branch("BS", &vBS_, "x/F:y/F:z/F");

  }  // bsLabels_

  for (const auto& label : bcLabels_) {
    std::string treeName = "PixelBarycentre";
    if (!label.empty())
      treeName = "PixelBarycentre_";
    treeName += label;
    bcTrees_[label] = tFileService->make<TTree>(TString(treeName), "PixelBarycentre analyzer ntuple");

    bcTrees_[label]->Branch("run", &run_, "run/I");
    bcTrees_[label]->Branch("ls", &ls_, "ls/I");

    bcTrees_[label]->Branch("PIX", &vPIX_, "x/F:y/F:z/F");

    bcTrees_[label]->Branch("BPIX", &vBPIX_, "x/F:y/F:z/F");
    bcTrees_[label]->Branch("BPIX_Flipped", &vBPIX_Flipped_, "x/F:y/F:z/F");
    bcTrees_[label]->Branch("BPIX_NonFlipped", &vBPIX_NonFlipped_, "x/F:y/F:z/F");
    bcTrees_[label]->Branch("BPIX_DiffFlippedNonFlipped", &vBPIX_DiffFlippedNonFlipped_, "x/F:y/F:z/F");

    bcTrees_[label]->Branch("FPIX", &vFPIX_, "x/F:y/F:z/F");
    bcTrees_[label]->Branch("FPIX_plus", &vFPIX_plus_, "x/F:y/F:z/F");
    bcTrees_[label]->Branch("FPIX_minus", &vFPIX_minus_, "x/F:y/F:z/F");

    //per-layer
    for (unsigned int i = 0; i < nPixelLayers; i++) {
      TString structure = "BPIXLYR";
      int layer = i + 1;
      structure += layer;

      bcTrees_[label]->Branch(structure, &vBPIXLayer_[i], "x/F:y/F:z/F");
      bcTrees_[label]->Branch(structure + "_Flipped", &vBPIXLayer_Flipped_[i], "x/F:y/F:z/F");
      bcTrees_[label]->Branch(structure + "_NonFlipped", &vBPIXLayer_NonFlipped_[i], "x/F:y/F:z/F");
      bcTrees_[label]->Branch(
          structure + "_DiffFlippedNonFlipped", &vBPIXLayer_DiffFlippedNonFlipped_[i], "x/F:y/F:z/F");
    }

    //per-disk/ring
    for (unsigned int i = 0; i < nPixelDiscs; i++) {
      TString structure = "FPIXDisk_plus";
      int disk = i + 1;
      structure += disk;
      bcTrees_[label]->Branch(structure, &vFPIXDisks_plus_[i], "x/F:y/F:z/F");

      structure = "FPIXDisk_minus";
      structure += disk;
      bcTrees_[label]->Branch(structure, &vFPIXDisks_minus_[i], "x/F:y/F:z/F");
    }

  }  // bcLabels_
}

// ------------ method called once each job just after ending the event loop  ------------
void PixelBaryCentreAnalyzer::endJob() {
  bcLabels_.clear();
  bsLabels_.clear();

  bcTrees_.clear();
  bsTrees_.clear();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PixelBaryCentreAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Validates alignment payloads by providing the position of the pixel barycenter positions");
  desc.addUntracked<bool>("usePixelQuality", false);
  desc.addUntracked<std::vector<std::string>>("tkAlignLabels", {});
  desc.addUntracked<std::vector<std::string>>("beamSpotLabels", {});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelBaryCentreAnalyzer);
