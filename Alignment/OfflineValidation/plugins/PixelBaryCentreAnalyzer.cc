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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
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

// TFileService
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ROOT
#include <TTree.h>
#include <TString.h>
#include <TVector3.h>


//
// class declaration
//

class PixelBaryCentreAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit PixelBaryCentreAnalyzer(const edm::ParameterSet &);
  ~PixelBaryCentreAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
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
  std::map<std::string, edm::ESGetToken<Alignments, TrackerAlignmentRcd> > tkAlignTokens_;
  std::map<std::string, edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> > bsTokens_;

  // tree content
  int run_;
  int ls_;

  double BSx0_, BSy0_, BSz0_;
  TVector3 BS_;

  double PIXx0_, PIXy0_, PIXz0_;
  TVector3 PIX_, BPIX_, FPIX_;
  TVector3 BPIX_Flipped_, BPIX_NonFlipped_, BPIX_DiffFlippedNonFlipped_;

  TVector3 BPIXLayer_[4];
  TVector3 BPIXLayer_Flipped_[4];
  TVector3 BPIXLayer_NonFlipped_[4];
  TVector3 BPIXLayer_DiffFlippedNonFlipped_[4];

  TVector3 FPIX_plus_, FPIX_minus_;
  TVector3 FPIXDisks_plus_[3];
  TVector3 FPIXDisks_minus_[3];

  edm::Service<TFileService> tFileService;
  std::map<std::string, TTree *> bcTrees_;
  std::map<std::string, TTree *> bsTrees_;
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
      gprToken_(esConsumes<Alignments, GlobalPositionRcd>()){
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

PixelBaryCentreAnalyzer::~PixelBaryCentreAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

void PixelBaryCentreAnalyzer::initBS() {
  double dummy_float = 999999.0;

  BSx0_ = dummy_float;
  BSy0_ = dummy_float;
  BSz0_ = dummy_float;

  BS_ = TVector3(dummy_float, dummy_float, dummy_float);
}

void PixelBaryCentreAnalyzer::initBC() {
  // init to large number (unreasonable number) not zero
  double dummy_float = 999999.0;

  PIXx0_ = dummy_float;
  PIXy0_ = dummy_float;
  PIXz0_ = dummy_float;

  PIX_ = TVector3(dummy_float, dummy_float, dummy_float);
  BPIX_ = TVector3(dummy_float, dummy_float, dummy_float);
  FPIX_ = TVector3(dummy_float, dummy_float, dummy_float);

  BPIX_Flipped_ = TVector3(dummy_float, dummy_float, dummy_float);
  BPIX_NonFlipped_ = TVector3(dummy_float, dummy_float, dummy_float);
  BPIX_DiffFlippedNonFlipped_ = TVector3(dummy_float, dummy_float, dummy_float);

  FPIX_plus_ = TVector3(dummy_float, dummy_float, dummy_float);
  FPIX_minus_ = TVector3(dummy_float, dummy_float, dummy_float);

  for (unsigned int i = 0; i < 4; i++) {
    BPIXLayer_[i] = TVector3(dummy_float, dummy_float, dummy_float);
    BPIXLayer_Flipped_[i] = TVector3(dummy_float, dummy_float, dummy_float);
    BPIXLayer_NonFlipped_[i] = TVector3(dummy_float, dummy_float, dummy_float);
    BPIXLayer_DiffFlippedNonFlipped_[i] = TVector3(dummy_float, dummy_float, dummy_float);
  }

  for (unsigned int i = 0; i < 3; i++) {
    FPIXDisks_plus_[i] = TVector3(dummy_float, dummy_float, dummy_float);
    FPIXDisks_minus_[i] = TVector3(dummy_float, dummy_float, dummy_float);
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

    // global position
    const Alignments* globalAlignments = &iSetup.getData(gprToken_);
    std::unique_ptr<const Alignments> globalPositions = std::make_unique<Alignments>(*globalAlignments);
    const AlignTransform& globalCoordinates = align::DetectorGlobalPosition(*globalPositions, DetId(DetId::Tracker));
    TVector3 globalTkPosition(
        globalCoordinates.translation().x(), globalCoordinates.translation().y(), globalCoordinates.translation().z());

    // loop over bclabels
    for (const auto& label : bcLabels_) {
      // init tree content
      PixelBaryCentreAnalyzer::initBC();

      // Get TkAlign from EventSetup:
      const Alignments* alignments = &iSetup.getData(tkAlignTokens_[label]);
      std::vector<AlignTransform> tkAlignments = alignments->m_align;

      TVector3 barycentre_BPIX;
      float nmodules_BPIX(0.);

      TVector3 barycentre_FPIX;
      float nmodules_FPIX(0.);

      // per-ladder barycentre
      std::map<int, std::map<int, float>> nmodules_bpix;       // layer-ladder
      std::map<int, std::map<int, TVector3>> barycentre_bpix;  // layer-ladder

      // per-ladder barycentre
      std::map<int, std::map<int, float>> nmodules_fpix;       // disk-ring
      std::map<int, std::map<int, TVector3>> barycentre_fpix;  // disk-ring

      // loop over tracker module
      for (const auto& ali : tkAlignments) {
        //DetId
        const DetId& detId = DetId(ali.rawId());
        // remove bad module
        if (usePixelQuality_ && badPixelInfo->IsModuleBad(detId))
          continue;

        TVector3 ali_translation(ali.translation().x(), ali.translation().y(), ali.translation().z());

        int subid = DetId(detId).subdetId();
        // BPIX
        if (subid == PixelSubdetector::PixelBarrel) {
          nmodules_BPIX += 1;
          barycentre_BPIX += ali_translation;

          int layer = tkTopo->pxbLayer(detId);
          int ladder = tkTopo->pxbLadder(detId);
          nmodules_bpix[layer][ladder] += 1;
          barycentre_bpix[layer][ladder] += ali_translation;

        }  // BPIX

        // FPIX
        if (subid == PixelSubdetector::PixelEndcap) {
          nmodules_FPIX += 1;
          barycentre_FPIX += ali_translation;

          int disk = tkTopo->pxfDisk(detId);
          int quadrant = PixelEndcapName(detId, tkTopo, phase_).halfCylinder();
          if(quadrant<3) disk *= -1;

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
      TVector3 barycentre_PIX = barycentre_BPIX + barycentre_FPIX;
      float nmodules_PIX = nmodules_BPIX + nmodules_FPIX;
      PIX_ = (1.0 / nmodules_PIX) * barycentre_PIX + globalTkPosition;
      PIXx0_ = PIX_.X();
      PIXy0_ = PIX_.Y();
      PIXz0_ = PIX_.Z();

      //BPIX
      BPIX_ = (1.0 / nmodules_BPIX) * barycentre_BPIX + globalTkPosition;
      //FPIX
      FPIX_ = (1.0 / nmodules_FPIX) * barycentre_FPIX + globalTkPosition;

      // BPIX substructures

      // BPix barycentre per-ladder per-layer
      // !!! Based on assumption : each ladder has the same number of modules in the same layer
      // inner =  flipped; outer = non-flipped
      //
      // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
      // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
      //

      int nmodules_BPIX_Flipped = 0;
      int nmodules_BPIX_NonFlipped = 0;
      TVector3 BPIX_Flipped(0.0, 0.0, 0.0);
      TVector3 BPIX_NonFlipped(0.0, 0.0, 0.0);

      // loop over layers
      for (std::map<int, std::map<int, TVector3>>::iterator il = barycentre_bpix.begin(); il != barycentre_bpix.end(); ++il) {
        int layer = il->first;

        int nmodulesLayer = 0;
        int nmodulesLayer_Flipped = 0;
        int nmodulesLayer_NonFlipped = 0;
        TVector3 BPIXLayer(0.0, 0.0, 0.0);
        TVector3 BPIXLayer_Flipped(0.0, 0.0, 0.0);
        TVector3 BPIXLayer_NonFlipped(0.0, 0.0, 0.0);

        // loop over ladder
        std::map<int, TVector3> barycentreLayer = barycentre_bpix[layer];
        for (std::map<int, TVector3>::iterator it = barycentreLayer.begin(); it != barycentreLayer.end(); ++it) {
          int ladder = it->first;
          //BPIXLayerLadder_[layer][ladder] = (1.0/nmodules[layer][ladder])*barycentreLayer[ladder] + globalTkPosition;

          nmodulesLayer += nmodules_bpix[layer][ladder];
          BPIXLayer += barycentreLayer[ladder];

          // Phase-1
          //
          // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
          if (phase_ == 1) {
            if (layer != 4) {  // layer 1-3

              if (ladder % 2 != 0) {  // odd ladder = inner = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              } else {
                nmodulesLayer_NonFlipped += nmodules_bpix[layer][ladder];
                BPIXLayer_NonFlipped += barycentreLayer[ladder];
              }
            } else {  // layer-4

              if (ladder % 2 == 0) {  // even ladder = inner = flipped
                nmodulesLayer_Flipped += nmodules_bpix[layer][ladder];
                BPIXLayer_Flipped += barycentreLayer[ladder];
              } else {  // odd ladder = outer = non-flipped
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

        BPIXLayer_[layer-1] = BPIXLayer;
        BPIXLayer_Flipped_[layer-1] = BPIXLayer_Flipped;
        BPIXLayer_NonFlipped_[layer-1] = BPIXLayer_NonFlipped;

        BPIXLayer_DiffFlippedNonFlipped_[layer-1] = BPIXLayer_Flipped - BPIXLayer_NonFlipped;

      }  // loop over layers

      BPIX_Flipped *= (1.0 / nmodules_BPIX_Flipped);
      BPIX_Flipped += globalTkPosition;
      BPIX_NonFlipped *= (1.0 / nmodules_BPIX_NonFlipped);
      BPIX_NonFlipped += globalTkPosition;

      BPIX_Flipped_ = BPIX_Flipped;
      BPIX_NonFlipped_ = BPIX_NonFlipped;
      BPIX_DiffFlippedNonFlipped_ = BPIX_Flipped - BPIX_NonFlipped;


      // FPIX substructures
      int nmodules_FPIX_plus = 0;
      int nmodules_FPIX_minus = 0;
      TVector3 FPIX_plus(0.0, 0.0, 0.0);
      TVector3 FPIX_minus(0.0, 0.0, 0.0);
      // loop over disks
      for (std::map<int, std::map<int, TVector3>>::iterator id = barycentre_fpix.begin(); id != barycentre_fpix.end(); ++id) {
           int disk = id->first;

           int nmodulesDisk = 0;
           TVector3 FPIXDisk(0.0, 0.0, 0.0);

           std::map<int, TVector3> baryCentreDisk = id->second;
           for (std::map<int, TVector3>::iterator ir = baryCentreDisk.begin(); ir != baryCentreDisk.end(); ++ir) { // loop over rings
                int ring = ir->first;
                nmodulesDisk += nmodules_fpix[disk][ring];
                FPIXDisk     += ir->second;
                if(disk > 0) {
                  nmodules_FPIX_plus += nmodules_fpix[disk][ring];
                  FPIX_plus += ir->second;
                }
                if(disk < 0) {
                  nmodules_FPIX_minus += nmodules_fpix[disk][ring];
                  FPIX_minus += ir->second;
                }

           } // loop over rings

           FPIXDisk *= (1.0 / nmodulesDisk);
           FPIXDisk += globalTkPosition;

           if(disk > 0)
             FPIXDisks_plus_[disk-1] = FPIXDisk;
           if(disk < 0)
             FPIXDisks_minus_[-disk-1] = FPIXDisk;
      } // loop over disks

      FPIX_plus *= (1.0 / nmodules_FPIX_plus);
      FPIX_plus += globalTkPosition;
      FPIX_minus *= (1.0 / nmodules_FPIX_minus);
      FPIX_minus += globalTkPosition;

      FPIX_plus_ = FPIX_plus;
      FPIX_minus_ = FPIX_minus;

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

      BSx0_ = mybeamspot->GetX();
      BSy0_ = mybeamspot->GetY();
      BSz0_ = mybeamspot->GetZ();

      BS_ = TVector3(BSx0_, BSy0_, BSz0_);

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
    bsTrees_[label]->Branch("BSx0", &BSx0_, "BSx0/D");
    bsTrees_[label]->Branch("BSy0", &BSy0_, "BSy0/D");
    bsTrees_[label]->Branch("BSz0", &BSz0_, "BSz0/D");

    bsTrees_[label]->Branch("BS", &BS_);

  }  // bsLabels_

  for (const auto& label : bcLabels_) {
    std::string treeName = "PixelBarycentre";
    if (!label.empty())
      treeName = "PixelBarycentre_";
    treeName += label;
    bcTrees_[label] = tFileService->make<TTree>(TString(treeName), "PixelBarycentre analyzer ntuple");

    bcTrees_[label]->Branch("run", &run_, "run/I");
    bcTrees_[label]->Branch("ls", &ls_, "ls/I");
    bcTrees_[label]->Branch("PIXx0", &PIXx0_);
    bcTrees_[label]->Branch("PIXy0", &PIXy0_);
    bcTrees_[label]->Branch("PIXz0", &PIXz0_);

    bcTrees_[label]->Branch("PIX", &PIX_);

    bcTrees_[label]->Branch("BPIX", &BPIX_);
    bcTrees_[label]->Branch("BPIX_Flipped", &BPIX_Flipped_);
    bcTrees_[label]->Branch("BPIX_NonFlipped", &BPIX_NonFlipped_);
    bcTrees_[label]->Branch("BPIX_DiffFlippedNonFlipped", &BPIX_DiffFlippedNonFlipped_);

    bcTrees_[label]->Branch("FPIX", &FPIX_);
    bcTrees_[label]->Branch("FPIX_plus", &FPIX_plus_);
    bcTrees_[label]->Branch("FPIX_minus", &FPIX_minus_);

    //per-layer
    for (unsigned int i = 0; i < 4; i++) {
      TString structure = "BPIXLYR";
      int layer = i + 1;
      structure += layer;

      bcTrees_[label]->Branch(structure, &BPIXLayer_[i]);
      bcTrees_[label]->Branch(structure + "_Flipped", &BPIXLayer_Flipped_[i]);
      bcTrees_[label]->Branch(structure + "_NonFlipped", &BPIXLayer_NonFlipped_[i]);
      bcTrees_[label]->Branch(structure + "_DiffFlippedNonFlipped", &BPIXLayer_DiffFlippedNonFlipped_[i]);
    }

    //per-disk/ring
    for (unsigned int i = 0; i < 3; i++) {
      TString structure = "FPIXDisk+";
      int disk = i + 1;
      structure += disk;
      bcTrees_[label]->Branch(structure, &FPIXDisks_plus_[i]);

      structure = "FPIXDisk-";
      structure += disk;
      bcTrees_[label]->Branch(structure, &FPIXDisks_minus_[i]);

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
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelBaryCentreAnalyzer);
