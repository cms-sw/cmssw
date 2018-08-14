#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/PHGCalParametersRcd.h"
#include "CondFormats/GeometryObjects/interface/PHGCalParameters.h"

class HGCalParametersAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit HGCalParametersAnalyzer( const edm::ParameterSet& ) {}
  ~HGCalParametersAnalyzer() override {}
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void
HGCalParametersAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::LogInfo("HGCalParametersAnalyzer") << "Here I am";
  
  edm::ESHandle<PHGCalParameters> phgp;
  iSetup.get<PHGCalParametersRcd>().get( phgp );

  std::cout << phgp->name_ << "\n";
  for (auto it : phgp->cellSize_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleBlS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleTlS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleHS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleDzS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleAlphaS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleCellS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleBlR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleTlR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleHR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleDzR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleAlphaR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleCellR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformTranX_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformTranY_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformTranZ_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotXX_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotYX_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotZX_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotXY_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotYY_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotZY_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotXZ_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotYZ_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformRotZZ_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->zLayerHex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->rMinLayHex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->rMaxLayHex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (unsigned int k=0; k<phgp->waferPosX_.size(); ++k)
    std::cout << "(" << phgp->waferPosX_[k] << ", " << phgp->waferPosY_[k] 
	      << ") ";
  std::cout << "\n";
  
  for (unsigned int k=0; k<phgp->cellFineX_.size(); ++k)
    std::cout << "(" << phgp->cellFineX_[k] << ", " << phgp->cellFineY_[k] 
	      << ") ";
  std::cout << "\n";
  
  for (unsigned int k=0; k<phgp->cellCoarseX_.size(); ++k)
    std::cout << "(" << phgp->cellCoarseX_[k] << ", " << phgp->cellCoarseY_[k]
	      << ") ";
  std::cout << "\n";
  
  for (auto it : phgp->boundR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleLayS_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->moduleLayR_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->layer_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->layerIndex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->layerGroup_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->cellFactor_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->depth_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->depthIndex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->depthLayerF_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->waferCopy_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->waferTypeL_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->waferTypeT_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->layerGroupM_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->layerGroupO_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->trformIndex_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  for (auto it : phgp->slopeMin_)
    std::cout << it << ", ";
  std::cout << "\n";
  
  std::cout << phgp->waferR_   << "\n";
  std::cout << phgp->nCells_   << "\n";
  std::cout << phgp->nSectors_ << "\n";
  std::cout << phgp->mode_     << "\n";
}

DEFINE_FWK_MODULE(HGCalParametersAnalyzer);
