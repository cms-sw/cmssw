#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"

class HGCalParameterTester : public edm::one::EDAnalyzer<> {

public:
  explicit HGCalParameterTester( const edm::ParameterSet& );
  ~HGCalParameterTester() override {}
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
private:
  const std::string name_;
  const int         mode_;
};

HGCalParameterTester::HGCalParameterTester(const edm::ParameterSet& ic) :
  name_(ic.getUntrackedParameter<std::string>("Name")),
  mode_(ic.getUntrackedParameter<int>("Mode")) { }

void HGCalParameterTester::analyze(const edm::Event& iEvent, 
				   const edm::EventSetup& iSetup) {

  edm::LogVerbatim("HGCalGeomr") << "HGCalParameter::Here I am";
  auto start = std::chrono::high_resolution_clock::now();
  
  edm::ESHandle<HGCalParameters> phgp;
  iSetup.get<IdealGeometryRecord>().get(name_, phgp);

  std::cout << phgp->name_ << "\n";
  unsigned int k(0);
  if (mode_ == 0) {
    std::cout << "CellSize with " << phgp->cellSize_.size() << " elements\n";
    for (auto const& it : phgp->cellSize_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleBls with " << phgp->moduleBlS_.size() << " elements\n";
    for (auto const& it : phgp->moduleBlS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleTls with " << phgp->moduleTlS_.size() << " elements\n";
    for (auto const& it : phgp->moduleTlS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleHS with " << phgp->moduleHS_.size() << " elements\n";
    for (auto const& it : phgp->moduleHS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleDzs with " << phgp->moduleDzS_.size() << " elements\n";
    for (auto const& it : phgp->moduleDzS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleAlphaS with " << phgp->moduleAlphaS_.size() 
	      << " elements\n";
    for (auto const& it : phgp->moduleAlphaS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleCellS with " << phgp->moduleCellS_.size()
	      << " elements\n";
    for (auto const& it : phgp->moduleCellS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleBlR with " << phgp->moduleBlR_.size() << " elements\n";
    for (auto const& it : phgp->moduleBlR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleTlR with " << phgp->moduleTlR_.size() << " elements\n";
    for (auto const& it : phgp->moduleTlR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleHR with " << phgp->moduleHR_.size() << " elements\n";
    for (auto const& it : phgp->moduleHR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleDzR with " << phgp->moduleDzR_.size() << " elements\n";
    for (auto const& it : phgp->moduleDzR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleAlphaR with " << phgp->moduleAlphaR_.size() 
	      << " elements\n";
    for (auto const& it : phgp->moduleAlphaR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleCellR with " << phgp->moduleCellR_.size() 
	      << " elements\n";
    for (auto const& it : phgp->moduleCellR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranX with " << phgp->trformTranX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranX_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranY with " << phgp->trformTranY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranY_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranZ with " << phgp->trformTranZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXX with " << phgp->trformRotXX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYX with " << phgp->trformRotYX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZX with " << phgp->trformRotZX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXY with " << phgp->trformRotXY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotXY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYY with " << phgp->trformRotYY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotYY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZY with " << phgp->trformRotZY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXZ with " << phgp->trformRotXZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYZ with " << phgp->trformRotYZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZZ with " << phgp->trformRotZZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zLayerHex with " << phgp->zLayerHex_.size() << " elements\n";
    for (auto const& it : phgp->zLayerHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMinLayHex with " << phgp->rMinLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMinLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMaxLayHex with " << phgp->rMaxLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMaxLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferPosX with " << phgp->waferPosX_.size() << " elements\n";
    for (auto const& it : phgp->waferPosX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferPosY with " << phgp->waferPosY_.size() << " elements\n";
    for (auto const& it : phgp->waferPosY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "cellFineX with " << phgp->cellFineX_.size() << " elements\n";
    for (auto const& it : phgp->cellFineX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "cellFineY with " << phgp->cellFineY_.size() << " elements\n";
    for (auto const& it : phgp->cellFineY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "cellCoarseX with " << phgp->cellCoarseX_.size()
	      << " elements\n";
    for (auto const& it : phgp->cellCoarseX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "cellCoarseY with " << phgp->cellCoarseY_.size()
	      << " elements\n";
    for (auto const& it : phgp->cellCoarseY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "boundR with " << phgp->boundR_.size() << " elements\n";
    for (auto const& it : phgp->boundR_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleLayS with " << phgp->moduleLayS_.size() 
	      << " elements\n";
    for (auto const& it : phgp->moduleLayS_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "moduleLayR_ with " << phgp->moduleLayR_.size() 
	      << " elements\n";
    for (auto const& it : phgp->moduleLayR_) { ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
      std::cout << it << ", ";
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layer with " << phgp->layer_.size() << " elements\n";
    for (auto const& it : phgp->layer_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerIndex with " << phgp->layerIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->layerIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerGroup with " << phgp->layerGroup_.size() 
	      << " elements\n";
    for (auto const& it : phgp->layerGroup_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "cellFactor with " << phgp->cellFactor_.size()
	      << " elements\n";
    for (auto const& it : phgp->cellFactor_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "depth with " << phgp->depth_.size() << " elements\n";
    for (auto const& it : phgp->depth_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "depthIndex with " << phgp->depthIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->depthIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "depthLayerF with " << phgp->depthLayerF_.size()
	      << " elements\n";
    for (auto const& it : phgp->depthLayerF_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferCopy with " << phgp->waferCopy_.size() << " elements\n";
    for (auto const& it : phgp->waferCopy_) {
      std::cout << it << ", "; ++k; 
      if (k == 8) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
    
    std::cout << "waferTypeL with " << phgp->waferTypeL_.size()
	      << " elements\n";
    for (auto const& it : phgp->waferTypeL_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferTypeT with " << phgp->waferTypeT_.size() 
	      << " elements\n";
    for (auto const& it : phgp->waferTypeT_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerGroupM with " << phgp->layerGroupM_.size()
	      << " elements\n";
    for (auto const& it : phgp->layerGroupM_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerGroupO with " << phgp->layerGroupO_.size() 
	      << " elements\n";
    for (auto const& it : phgp->layerGroupO_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformIndex with " << phgp->trformIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}

    std::cout << "WaferR_: "   << phgp->waferR_   << "\n";
    std::cout << "SlopeMin_: " << phgp->slopeMin_ << "\n";
    std::cout << "nCells_: "   << phgp->nCells_   << "\n";
    std::cout << "nSectors_: " << phgp->nSectors_ << "\n";
    std::cout << "mode_: "     << phgp->mode_     << "\n";
  } else if (mode_ == 1) {

    std::cout << "SlopeMin_: " << phgp->slopeMin_ << "\n";
    std::cout << "Wafer Parameters: " << phgp->waferSize_ << ":"
	      << phgp->waferR_   << ":" << phgp->waferThick_ << ":"
	      << phgp->sensorSeparation_ << ":" << phgp->mouseBite_ << "\n";
    std::cout << "nCells_: "  << phgp->nCells_   << ":" << phgp->nCellsFine_
	      << ":" << phgp->nCellsCoarse_ << "\n";
    std::cout << "nSectors_: " << phgp->nSectors_ << "\n";
    std::cout << "mode_: "     << phgp->mode_     << "\n";
    std::cout << "CellThickness with " << phgp->cellThickness_.size() 
	      << " elements\n";
    for (auto const& it : phgp->cellThickness_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "radius100to200 with " << phgp->radius100to200_.size()
	      << " elements\n";
    for (auto const& it : phgp->radius100to200_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "radius200to300 with " << phgp->radius200to300_.size() 
	      << " elements\n";
    for (auto const& it : phgp->radius200to300_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}

    std::cout << "nCornerCut " << phgp->nCornerCut_ << "  zMinForRad "
	      << phgp->zMinForRad_ << "\n";
  
    std::cout << "radiusMixBoundary with " << phgp->radiusMixBoundary_.size()
	      << " elements\n";
    for (auto const& it : phgp->radiusMixBoundary_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "slopeTop with " << phgp->slopeTop_.size() << " elements\n";
    for (auto const& it : phgp->slopeTop_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zFront with " << phgp->zFront_.size()
	      << " elements\n";
    for (auto const& it : phgp->zFront_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMaxFront with " << phgp->rMaxFront_.size() << " elements\n";
    for (auto const& it : phgp->rMaxFront_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zRanges with " << phgp->zRanges_.size() << " elements\n";
    for (auto const& it : phgp->zRanges_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranX with " << phgp->trformTranX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranX_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranY with " << phgp->trformTranY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranY_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranZ with " << phgp->trformTranZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXX with " << phgp->trformRotXX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYX with " << phgp->trformRotYX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZX with " << phgp->trformRotZX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXY with " << phgp->trformRotXY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotXY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYY with " << phgp->trformRotYY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotYY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZY with " << phgp->trformRotZY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXZ with " << phgp->trformRotXZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYZ with " << phgp->trformRotYZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZZ with " << phgp->trformRotZZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zLayerHex with " << phgp->zLayerHex_.size() << " elements\n";
    for (auto const& it : phgp->zLayerHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMinLayHex with " << phgp->rMinLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMinLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMaxLayHex with " << phgp->rMaxLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMaxLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferPosX with " << phgp->waferPosX_.size() << " elements\n";
    for (auto const& it : phgp->waferPosX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferPosY with " << phgp->waferPosY_.size() << " elements\n";
    for (auto const& it : phgp->waferPosY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layer with " << phgp->layer_.size() << " elements\n";
    for (auto const& it : phgp->layer_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerIndex with " << phgp->layerIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->layerIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "waferCopy with " << phgp->waferCopy_.size() << " elements\n";
    for (auto const& it : phgp->waferCopy_) {
      std::cout << it << ", "; ++k; 
      if (k == 8) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
    
    std::cout << "waferTypeL with " << phgp->waferTypeL_.size()
	      << " elements\n";
    for (auto const& it : phgp->waferTypeL_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformIndex with " << phgp->trformIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}

  } else {

    std::cout << "SlopeMin_: " << phgp->slopeMin_ << "\n";
    std::cout << "EtaMaxBH: "  << phgp->etaMaxBH_ << "\n";
    std::cout << "mode_: "     << phgp->mode_     << "\n";
  
    std::cout << "radiusMixBoundary with " << phgp->radiusMixBoundary_.size()
	      << " elements\n";
    for (auto const& it : phgp->radiusMixBoundary_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "nPhiBinBH with " << phgp->nPhiBinBH_.size() << " elements\n";
    for (auto const& it : phgp->nPhiBinBH_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "dPhiEta with " << phgp->dPhiEta_.size() << " elements\n";
    for (auto const& it : phgp->dPhiEta_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "slopeTop with " << phgp->slopeTop_.size() << " elements\n";
    for (auto const& it : phgp->slopeTop_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zFront with " << phgp->zFront_.size()
	      << " elements\n";
    for (auto const& it : phgp->zFront_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMaxFront with " << phgp->rMaxFront_.size() << " elements\n";
    for (auto const& it : phgp->rMaxFront_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zRanges with " << phgp->zRanges_.size() << " elements\n";
    for (auto const& it : phgp->zRanges_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranX with " << phgp->trformTranX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranX_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranY with " << phgp->trformTranY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranY_) {
      std::cout << it << ", "; ++k; 
      if (k == 20) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformTranZ with " << phgp->trformTranZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformTranZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXX with " << phgp->trformRotXX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYX with " << phgp->trformRotYX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZX with " << phgp->trformRotZX_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZX_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXY with " << phgp->trformRotXY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotXY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYY with " << phgp->trformRotYY_.size()
	      << " elements\n";
    for (auto const& it : phgp->trformRotYY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZY with " << phgp->trformRotZY_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZY_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotXZ with " << phgp->trformRotXZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotXZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotYZ with " << phgp->trformRotYZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotYZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformRotZZ with " << phgp->trformRotZZ_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformRotZZ_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "zLayerHex with " << phgp->zLayerHex_.size() << " elements\n";
    for (auto const& it : phgp->zLayerHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMinLayHex with " << phgp->rMinLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMinLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "rMaxLayHex with " << phgp->rMaxLayHex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->rMaxLayHex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layer with " << phgp->layer_.size() << " elements\n";
    for (auto const& it : phgp->layer_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "layerIndex with " << phgp->layerIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->layerIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 18) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  
    std::cout << "trformIndex with " << phgp->trformIndex_.size() 
	      << " elements\n";
    for (auto const& it : phgp->trformIndex_) {
      std::cout << it << ", "; ++k; 
      if (k == 10) { std::cout << "\n"; k = 0;}
    }
    if (k > 0) {std::cout << "\n"; k = 0;}
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
}

DEFINE_FWK_MODULE(HGCalParameterTester);
