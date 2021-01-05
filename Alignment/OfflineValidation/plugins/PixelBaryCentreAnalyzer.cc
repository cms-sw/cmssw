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

// header file
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Alignment/OfflineValidation/plugins/PixelBaryCentreAnalyzer.h"

#include <memory>
#include <TTree.h>
#include <TString.h>
#include <TVector3.h>

#include <sstream>
#include <fstream>

//
// constructors and destructor
//
PixelBaryCentreAnalyzer::PixelBaryCentreAnalyzer(const edm::ParameterSet& iConfig) :
  usePixelQuality_(iConfig.getUntrackedParameter<bool>("usePixelQuality")),
  bcLabels_(iConfig.getUntrackedParameter<std::vector<std::string>>("tkAlignLabels")),
  bsLabels_(iConfig.getUntrackedParameter<std::vector<std::string>>("beamSpotLabels")),
  trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
  trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
  siPixelQualityToken_(esConsumes<SiPixelQuality, SiPixelQualityFromDbRcd>())
{

  for(const auto& label: bcLabels_) {
     bcTrees_[label] = nullptr;
  }

  for(const auto& label: bsLabels_) {
     bsTrees_[label] = nullptr;
  }

  usesResource("TFileService");

}


PixelBaryCentreAnalyzer::~PixelBaryCentreAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

void PixelBaryCentreAnalyzer::initBS(){

  double dummy_float = 999999.0;

  BSx0_  = dummy_float;
  BSy0_  = dummy_float;
  BSz0_  = dummy_float;

  BS_ = TVector3(dummy_float,dummy_float,dummy_float);
}

void PixelBaryCentreAnalyzer::initBC(){

  // init to large number (unreasonable number) not zero
  double dummy_float = 999999.0;

  PIXx0_ = dummy_float;
  PIXy0_ = dummy_float;
  PIXz0_ = dummy_float;

  PIX_  =  TVector3(dummy_float,dummy_float,dummy_float);
  BPIX_ =  TVector3(dummy_float,dummy_float,dummy_float);
  FPIX_ =  TVector3(dummy_float,dummy_float,dummy_float);

  BPIX_Flipped_    =  TVector3(dummy_float,dummy_float,dummy_float);
  BPIX_NonFlipped_ =  TVector3(dummy_float,dummy_float,dummy_float);
  BPIX_DiffFlippedNonFlipped_ =  TVector3(dummy_float,dummy_float,dummy_float);

  for(unsigned int i = 0; i<4; i++){

     BPIXLayer_[i]            = TVector3(dummy_float,dummy_float,dummy_float);
     BPIXLayer_Flipped_[i]    = TVector3(dummy_float,dummy_float,dummy_float);
     BPIXLayer_NonFlipped_[i] = TVector3(dummy_float,dummy_float,dummy_float);
     BPIXLayer_DiffFlippedNonFlipped_[i] = TVector3(dummy_float,dummy_float,dummy_float);

  }

}


// ------------ method called for each event  ------------
void PixelBaryCentreAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   bool prepareTkAlign = false;
   bool prepareBS = false;

   // ES watcher can noly run once in the same event,
   // otherwise it will turn false whatsoever because the condition doesn't change in the second time call.
   if (watcherTkAlign_.check(iSetup)) prepareTkAlign = true;
   if (watcherBS_.check(iSetup)) prepareBS = true;

   if(!prepareTkAlign && !prepareBS) return;

   run_  = iEvent.id().run();
   ls_   = iEvent.id().luminosityBlock();

   if ( prepareTkAlign ) { // check for new IOV for TKAlign

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
     edm::ESHandle<Alignments> globalAlignments;
     iSetup.get<GlobalPositionRcd>().get(globalAlignments);
     std::unique_ptr<const Alignments> globalPositions = std::make_unique<Alignments>(*globalAlignments);
     const AlignTransform& globalCoordinates = align::DetectorGlobalPosition(*globalPositions, DetId(DetId::Tracker));
     TVector3 globalTkPosition(globalCoordinates.translation().x(),
                               globalCoordinates.translation().y(),
                               globalCoordinates.translation().z());


    // loop over bclabels
    for(const auto& label: bcLabels_) {

        // init tree content
        PixelBaryCentreAnalyzer::initBC();

        // Get TkAlign from EventSetup:
        edm::ESHandle<Alignments> alignments;
        iSetup.get<TrackerAlignmentRcd>().get(label, alignments);
        std::vector<AlignTransform> tkAlignments = alignments->m_align;

        TVector3 barycentre_BPIX;
        float nmodules_BPIX(0.);

        TVector3 barycentre_FPIX;
        float nmodules_FPIX(0.);

        // per-ladder barycentre
        std::map<int, std::map<int, float>> nmodules;      // layer-ladder
        std::map<int, std::map<int, TVector3>> barycentre; // layer-ladder

        // loop over tracker module
        for (const auto &ali : tkAlignments) {
            //DetId
            const DetId& detId = DetId(ali.rawId());
            // remove bad module
            if(usePixelQuality_ && badPixelInfo->IsModuleBad(detId) ) continue;

            TVector3 ali_translation(ali.translation().x(),ali.translation().y(),ali.translation().z());

            int subid = DetId(detId).subdetId();
            // BPIX
            if (subid == PixelSubdetector::PixelBarrel){
               nmodules_BPIX += 1;
               barycentre_BPIX += ali_translation;

               int layer  = tkTopo->pxbLayer(detId);
               int ladder = tkTopo->pxbLadder(detId);
               nmodules[layer][ladder] += 1;
               barycentre[layer][ladder] += ali_translation;

            } // BPIX

            // FPIX
            if (subid == PixelSubdetector::PixelEndcap){
               nmodules_FPIX += 1;
               barycentre_FPIX += ali_translation;
            } // FPIX

        }// loop over tracker module

        //PIX
        TVector3 barycentre_PIX = barycentre_BPIX + barycentre_FPIX;
        float nmodules_PIX = nmodules_BPIX + nmodules_FPIX;
        PIX_  = (1.0/nmodules_PIX)*barycentre_PIX   + globalTkPosition;
        PIXx0_ = PIX_.X();
        PIXy0_ = PIX_.Y();
        PIXz0_ = PIX_.Z();

        //BPIX
        BPIX_ = (1.0/nmodules_BPIX)*barycentre_BPIX + globalTkPosition;
        //FPIX
        FPIX_ = (1.0/nmodules_FPIX)*barycentre_FPIX + globalTkPosition;

        // BPix barycentre per-ladder per-layer
        // !!! Based on assumption : each ladder has the same number of modules in the same layer
        // inner =  flipped; outer = non-flipped
        //
          // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
          // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
        //

        int nmodules_BPIX_Flipped = 0; int nmodules_BPIX_NonFlipped = 0;
        TVector3 BPIX_Flipped(0.0,0.0,0.0);
        TVector3 BPIX_NonFlipped(0.0,0.0,0.0);

        // loop over layers
        for (unsigned int i=0; i<barycentre.size(); i++){

             int layer = i+1;

             int nmodulesLayer = 0;
             int nmodulesLayer_Flipped = 0;
             int nmodulesLayer_NonFlipped = 0;
             TVector3 BPIXLayer(0.0,0.0,0.0);
             TVector3 BPIXLayer_Flipped(0.0,0.0,0.0);
             TVector3 BPIXLayer_NonFlipped(0.0,0.0,0.0);

             // loop over ladder
             std::map<int, TVector3> barycentreLayer = barycentre[layer];
             for (std::map<int, TVector3>::iterator it = barycentreLayer.begin(); it != barycentreLayer.end(); ++it) {

                 int ladder = it->first;
                 //BPIXLayerLadder_[layer][ladder] = (1.0/nmodules[layer][ladder])*barycentreLayer[ladder] + globalTkPosition;

                 nmodulesLayer += nmodules[layer][ladder];
                 BPIXLayer     += barycentreLayer[ladder];

                 // Phase-1
                 //
                 // Phase 1: Outer ladders are odd for layer 4 and even for layer 1,2,3
                 if(phase_ == 1) {

                    if(layer!=4){ // layer 1-3

                       if(ladder%2!=0) { // odd ladder = inner = flipped
                          nmodulesLayer_Flipped += nmodules[layer][ladder];
                          BPIXLayer_Flipped     += barycentreLayer[ladder];}
                       else{
                          nmodulesLayer_NonFlipped += nmodules[layer][ladder];
                          BPIXLayer_NonFlipped     += barycentreLayer[ladder];}
                    }
                    else{ // layer-4

                        if(ladder%2==0) { // even ladder = inner = flipped
                           nmodulesLayer_Flipped += nmodules[layer][ladder];
                           BPIXLayer_Flipped     += barycentreLayer[ladder]; }
                        else { // odd ladder = outer = non-flipped
                           nmodulesLayer_NonFlipped += nmodules[layer][ladder];
                           BPIXLayer_NonFlipped     += barycentreLayer[ladder]; }
                    }

                 } // phase-1

                 // Phase-0
                 //
                 // Phase 0: Outer ladders are odd for layer 1,3 and even for layer 2
                 if(phase_ == 0) {

                    if(layer == 2){ // layer-2

                       if(ladder%2!=0) { // odd ladder = inner = flipped
                          nmodulesLayer_Flipped += nmodules[layer][ladder];
                          BPIXLayer_Flipped     += barycentreLayer[ladder]; }
                       else{
                          nmodulesLayer_NonFlipped += nmodules[layer][ladder];
                          BPIXLayer_NonFlipped     += barycentreLayer[ladder];}
                    }
                    else{ // layer-1,3

                        if(ladder%2==0) { // even ladder = inner = flipped
                           nmodulesLayer_Flipped += nmodules[layer][ladder];
                           BPIXLayer_Flipped     += barycentreLayer[ladder]; }
                        else { // odd ladder = outer = non-flipped
                           nmodulesLayer_NonFlipped += nmodules[layer][ladder];
                           BPIXLayer_NonFlipped     += barycentreLayer[ladder]; }
                    }

                 } // phase-0

            }//loop over ladders

            // total BPIX flipped/non-flipped
            BPIX_Flipped += BPIXLayer_Flipped;
            BPIX_NonFlipped += BPIXLayer_NonFlipped;
            nmodules_BPIX_Flipped += nmodulesLayer_Flipped;
            nmodules_BPIX_NonFlipped += nmodulesLayer_NonFlipped;

            //BPIX per-layer
            BPIXLayer            *= (1.0/nmodulesLayer);            BPIXLayer += globalTkPosition;
            BPIXLayer_Flipped    *= (1.0/nmodulesLayer_Flipped);    BPIXLayer_Flipped += globalTkPosition;
            BPIXLayer_NonFlipped *= (1.0/nmodulesLayer_NonFlipped); BPIXLayer_NonFlipped += globalTkPosition;

            BPIXLayer_[i]            = BPIXLayer;
            BPIXLayer_Flipped_[i]    = BPIXLayer_Flipped;
            BPIXLayer_NonFlipped_[i] = BPIXLayer_NonFlipped;

            BPIXLayer_DiffFlippedNonFlipped_[i] = BPIXLayer_Flipped - BPIXLayer_NonFlipped;

        }// loop over layers

        BPIX_Flipped    *= (1.0/nmodules_BPIX_Flipped);     BPIX_Flipped += globalTkPosition;
        BPIX_NonFlipped *= (1.0/nmodules_BPIX_NonFlipped);  BPIX_NonFlipped += globalTkPosition;

        BPIX_Flipped_    = BPIX_Flipped;
        BPIX_NonFlipped_ = BPIX_NonFlipped;

        BPIX_DiffFlippedNonFlipped_ = BPIX_Flipped - BPIX_NonFlipped;

        bcTrees_[label]->Fill();

    } // bcLabels_

   } // check for new IOV for TKAlign

   // beamspot
   if ( prepareBS ) {

     // loop over bsLabels_
     for(const auto& label: bsLabels_) {

        // init bstree content
        PixelBaryCentreAnalyzer::initBS();

        // Get BeamSpot from EventSetup
        edm::ESHandle< BeamSpotObjects > beamhandle;
        iSetup.get<BeamSpotObjectsRcd>().get(label, beamhandle);
        const BeamSpotObjects *mybeamspot = beamhandle.product();

        BSx0_  = mybeamspot->GetX();
        BSy0_  = mybeamspot->GetY();
        BSz0_  = mybeamspot->GetZ();

        BS_ = TVector3(BSx0_,BSy0_,BSz0_);

        bsTrees_[label]->Fill();
     } // bsLabels_

   } // check for new IOV for BS


}


// ------------ method called once each job just before starting event loop  ------------
void
PixelBaryCentreAnalyzer::beginJob()
{

  // init bc bs trees
  for(const auto& label: bsLabels_) {

     std::string treeName = "BeamSpot";
     if(!label.empty()) treeName = "BeamSpot_";
     treeName += label;

     bsTrees_[label] = tFileService->make<TTree>(TString(treeName),"PixelBarycentre analyzer ntuple");

     bsTrees_[label]->Branch("run",&run_,"run/I");
     bsTrees_[label]->Branch("ls",&ls_,"ls/I");
     bsTrees_[label]->Branch("BSx0",&BSx0_,"BSx0/D");
     bsTrees_[label]->Branch("BSy0",&BSy0_,"BSy0/D");
     bsTrees_[label]->Branch("BSz0",&BSz0_,"BSz0/D");

     bsTrees_[label]->Branch("BS",&BS_);

  } // bsLabels_

  for(const auto& label: bcLabels_) {

     std::string treeName = "PixelBarycentre";
     if(!label.empty()) treeName = "PixelBarycentre_";
     treeName += label;
     bcTrees_[label] = tFileService->make<TTree>(TString(treeName),"PixelBarycentre analyzer ntuple");

     bcTrees_[label]->Branch("run",&run_,"run/I");
     bcTrees_[label]->Branch("ls",&ls_,"ls/I");
     bcTrees_[label]->Branch("PIXx0",&PIXx0_);
     bcTrees_[label]->Branch("PIXy0",&PIXy0_);
     bcTrees_[label]->Branch("PIXz0",&PIXz0_);

     bcTrees_[label]->Branch("PIX",&PIX_);
     bcTrees_[label]->Branch("BPIX",&BPIX_);
     bcTrees_[label]->Branch("BPIX_Flipped",&BPIX_Flipped_);
     bcTrees_[label]->Branch("BPIX_NonFlipped",&BPIX_NonFlipped_);
     bcTrees_[label]->Branch("BPIX_DiffFlippedNonFlipped",&BPIX_DiffFlippedNonFlipped_);
     bcTrees_[label]->Branch("FPIX",&FPIX_);

     //per-layer
     for(unsigned int i = 0; i<4; i++){

        TString structure="BPIXLYR";
        int layer = i+1;
        structure+=layer;

        bcTrees_[label]->Branch(structure,&BPIXLayer_[i]);
        bcTrees_[label]->Branch(structure+"_Flipped",&BPIXLayer_Flipped_[i]);
        bcTrees_[label]->Branch(structure+"_NonFlipped",&BPIXLayer_NonFlipped_[i]);
        bcTrees_[label]->Branch(structure+"_DiffFlippedNonFlipped",&BPIXLayer_DiffFlippedNonFlipped_[i]);

     }


  } // bcLabels_

}

// ------------ method called once each job just after ending the event loop  ------------
void
PixelBaryCentreAnalyzer::endJob()
{

   bcLabels_.clear();
   bsLabels_.clear();

   bcTrees_.clear();
   bsTrees_.clear();

}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PixelBaryCentreAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelBaryCentreAnalyzer);
