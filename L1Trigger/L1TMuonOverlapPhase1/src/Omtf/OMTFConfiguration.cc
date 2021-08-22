#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "CondFormats/L1TObjects/interface/LUT.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <utility>

///////////////////////////////////////////////
///////////////////////////////////////////////
RefHitDef::RefHitDef(unsigned int aInput, int aPhiMin, int aPhiMax, unsigned int aRegion, unsigned int aRefLayer)
    : iInput(aInput), iRegion(aRegion), iRefLayer(aRefLayer), range(std::pair<int, int>(aPhiMin, aPhiMax)) {}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool RefHitDef::fitsRange(int iPhi) const { return iPhi >= range.first && iPhi <= range.second; }
///////////////////////////////////////////////
///////////////////////////////////////////////
std::ostream &operator<<(std::ostream &out, const RefHitDef &aRefHitDef) {
  out << "iRefLayer: " << aRefHitDef.iRefLayer << " iInput: " << aRefHitDef.iInput << " iRegion: " << aRefHitDef.iRegion
      << " range: (" << aRefHitDef.range.first << ", " << aRefHitDef.range.second << std::endl;

  return out;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::initCounterMatrices() {
  ///Vector of all inputs
  std::vector<int> aLayer1D(nInputs(), 0);

  ///Vector of all layers
  vector2D aLayer2D;
  aLayer2D.assign(nLayers(), aLayer1D);

  ///Vector of all logic cones
  vector3D aLayer3D;
  aLayer3D.assign(nLogicRegions(), aLayer2D);

  ///Vector of all processors
  measurements4D.assign(nProcessors(), aLayer3D);
  measurements4Dref.assign(nProcessors(), aLayer3D);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::configure(const L1TMuonOverlapParams *omtfParams) {
  rawParams = *omtfParams;

  ///Set chamber sectors connections to logic processros.
  barrelMin.resize(nProcessors());
  endcap10DegMin.resize(nProcessors());
  endcap20DegMin.resize(nProcessors());

  barrelMax.resize(nProcessors());
  endcap10DegMax.resize(nProcessors());
  endcap20DegMax.resize(nProcessors());

  const std::vector<int> *connectedSectorsStartVec = omtfParams->connectedSectorsStart();
  const std::vector<int> *connectedSectorsEndVec = omtfParams->connectedSectorsEnd();

  std::copy(connectedSectorsStartVec->begin(), connectedSectorsStartVec->begin() + 6, barrelMin.begin());
  std::copy(connectedSectorsStartVec->begin() + 6, connectedSectorsStartVec->begin() + 12, endcap10DegMin.begin());
  std::copy(connectedSectorsStartVec->begin() + 12, connectedSectorsStartVec->end(), endcap20DegMin.begin());

  std::copy(connectedSectorsEndVec->begin(), connectedSectorsEndVec->begin() + 6, barrelMax.begin());
  std::copy(connectedSectorsEndVec->begin() + 6, connectedSectorsEndVec->begin() + 12, endcap10DegMax.begin());
  std::copy(connectedSectorsEndVec->begin() + 12, connectedSectorsEndVec->end(), endcap20DegMax.begin());

  ///Set connections tables
  const std::vector<L1TMuonOverlapParams::LayerMapNode> *layerMap = omtfParams->layerMap();

  for (unsigned int iLayer = 0; iLayer < nLayers(); ++iLayer) {
    L1TMuonOverlapParams::LayerMapNode aNode = layerMap->at(iLayer);
    hwToLogicLayer[aNode.hwNumber] = aNode.logicNumber;
    logicToHwLayer[aNode.logicNumber] = aNode.hwNumber;
    logicToLogic[aNode.logicNumber] = aNode.connectedToLayer;
    if (aNode.bendingLayer)
      bendingLayers.insert(aNode.logicNumber);
  }
  /////
  refToLogicNumber.resize(nRefLayers());

  const std::vector<L1TMuonOverlapParams::RefLayerMapNode> *refLayerMap = omtfParams->refLayerMap();
  for (unsigned int iRefLayer = 0; iRefLayer < nRefLayers(); ++iRefLayer) {
    L1TMuonOverlapParams::RefLayerMapNode aNode = refLayerMap->at(iRefLayer);
    refToLogicNumber[aNode.refLayer] = aNode.logicNumber;
  }
  /////
  std::vector<int> vector1D(nRefLayers(), nPhiBins());
  processorPhiVsRefLayer.assign(nProcessors(), vector1D);

  ///connections tables for each processor each logic cone
  ///Vector of all layers
  vector1D_pair aLayer1D(nLayers());
  ///Vector of all logic cones
  vector2D_pair aLayer2D;
  aLayer2D.assign(nLogicRegions(), aLayer1D);
  ///Vector of all processors
  connections.assign(nProcessors(), aLayer2D);

  ///Starting phis of each region
  ///Vector of all regions in one processor
  std::vector<std::pair<int, int> > aRefHit1D(nLogicRegions(), std::pair<int, int>(9999, 9999));
  ///Vector of all reflayers
  std::vector<std::vector<std::pair<int, int> > > aRefHit2D;
  aRefHit2D.assign(nRefLayers(), aRefHit1D);
  ///Vector of all inputs
  regionPhisVsRefLayerVsInput.assign(nInputs(), aRefHit2D);

  //Vector of ref hit definitions
  std::vector<RefHitDef> aRefHitsDefs(nRefHits());
  ///Vector of all processros
  refHitsDefs.assign(nProcessors(), aRefHitsDefs);

  const std::vector<int> *phiStartMap = omtfParams->globalPhiStartMap();
  const std::vector<L1TMuonOverlapParams::RefHitNode> *refHitMap = omtfParams->refHitMap();
  const std::vector<L1TMuonOverlapParams::LayerInputNode> *layerInputMap = omtfParams->layerInputMap();
  unsigned int tmpIndex = 0;
  for (unsigned int iProcessor = 0; iProcessor < nProcessors(); ++iProcessor) {
    for (unsigned int iRefLayer = 0; iRefLayer < nRefLayers(); ++iRefLayer) {
      int iPhiStart = phiStartMap->at(iRefLayer + iProcessor * nRefLayers());
      processorPhiVsRefLayer[iProcessor][iRefLayer] = iPhiStart;
    }
    for (unsigned int iRefHit = 0; iRefHit < nRefHits(); ++iRefHit) {
      int iPhiMin = refHitMap->at(iRefHit + iProcessor * nRefHits()).iPhiMin;
      int iPhiMax = refHitMap->at(iRefHit + iProcessor * nRefHits()).iPhiMax;
      unsigned int iInput = refHitMap->at(iRefHit + iProcessor * nRefHits()).iInput;
      unsigned int iRegion = refHitMap->at(iRefHit + iProcessor * nRefHits()).iRegion;
      unsigned int iRefLayer = refHitMap->at(iRefHit + iProcessor * nRefHits()).iRefLayer;
      regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion] = std::pair<int, int>(iPhiMin, iPhiMax);
      refHitsDefs[iProcessor][iRefHit] = RefHitDef(iInput, iPhiMin, iPhiMax, iRegion, iRefLayer);
    }
    for (unsigned int iLogicRegion = 0; iLogicRegion < nLogicRegions(); ++iLogicRegion) {
      for (unsigned int iLayer = 0; iLayer < nLayers(); ++iLayer) {
        tmpIndex = iLayer + iLogicRegion * nLayers() + iProcessor * nLogicRegions() * nLayers();
        unsigned int iFirstInput = layerInputMap->at(tmpIndex).iFirstInput;
        unsigned int nInputsInRegion = layerInputMap->at(tmpIndex).nInputs;
        connections[iProcessor][iLogicRegion][iLayer] =
            std::pair<unsigned int, unsigned int>(iFirstInput, nInputsInRegion);
        ///Symetrize connections. Use th same connections for all processors
        if (iProcessor != 0)
          connections[iProcessor][iLogicRegion][iLayer] = connections[0][iLogicRegion][iLayer];
      }
    }
  }

  initCounterMatrices();

  pdfBins = (1 << rawParams.nPdfAddrBits());
  pdfMaxVal = (1 << rawParams.nPdfValBits()) - 1;

  //configuration based on the firmware version parameter
  //TODO add next entries for the new firmware
  if (fwVersion() <= 4) {
    setMinDtPhiQuality(4);
  } else if (fwVersion() == 5) {
    setMinDtPhiQuality(2);
    setGhostBusterType("GhostBusterPreferRefDt");
  } else if (fwVersion() == 6) {
    setMinDtPhiQuality(2);
    setGhostBusterType("GhostBusterPreferRefDt");
  } else if (fwVersion() == 8) {
    setMinDtPhiQuality(2);
    setMinDtPhiBQuality(2);  //should be 4, but in the fwVersion = 8 was not yet implemented

    setSorterType(1);  //"byLLH"

    setRpcMaxClusterSize(3);
    setRpcMaxClusterCnt(2);
    setRpcDropAllClustersIfMoreThanMax(true);

    setGoldenPatternResultFinalizeFunction(9);

    setNoHitValueInPdf(true);

    setGhostBusterType("GhostBusterPreferRefDt");
  }

  //if (fwVersion() <= 6) {
  //setGoldenPatternResultFinalizeFunction(0); //TODO add other if in the new algorithm it is changed
}

void OMTFConfiguration::configureFromEdmParameterSet(const edm::ParameterSet &edmParameterSet) {
  edm::LogVerbatim("OMTFReconstruction") << "OMTFConfiguration::configureFromEdmParameterSet: setting the params from "
                                            "python config (overwrites the EventSetup (DB) params): "
                                         << std::endl;

  ProcConfigurationBase::configureFromEdmParameterSet(edmParameterSet);

  if (edmParameterSet.exists("goldenPatternResultFinalizeFunction")) {
    int finalizeFunction = edmParameterSet.getParameter<int>("goldenPatternResultFinalizeFunction");
    setGoldenPatternResultFinalizeFunction(finalizeFunction);
    edm::LogVerbatim("OMTFReconstruction")
        << "GoldenPatternResult::setFinalizeFunction: " << finalizeFunction << std::endl;
  }

  if (edmParameterSet.exists("noHitValueInPdf")) {
    setNoHitValueInPdf(edmParameterSet.getParameter<bool>("noHitValueInPdf"));
    edm::LogVerbatim("OMTFReconstruction")
        << "noHitValueInPdf: " << edmParameterSet.getParameter<bool>("noHitValueInPdf") << std::endl;
  }

  if (edmParameterSet.exists("sorterType")) {
    string sorterTypeStr = edmParameterSet.getParameter<std::string>("sorterType");
    if (sorterTypeStr == "byNhitsByLLH")
      sorterType = 0;
    if (sorterTypeStr == "byLLH")
      sorterType = 1;

    edm::LogVerbatim("OMTFReconstruction") << "sorterType: " << sorterType << " = "
                                           << edmParameterSet.getParameter<std::string>("sorterType") << std::endl;
  }

  if (edmParameterSet.exists("ghostBusterType")) {
    setGhostBusterType(edmParameterSet.getParameter<std::string>("ghostBusterType"));

    edm::LogVerbatim("OMTFReconstruction") << "ghostBusterType: " << getGhostBusterType() << std::endl;
  }

  setFixCscGeometryOffset(true);  //for the OMTF by default is true, read from python if needed
}

///////////////////////////////////////////////
///////////////////////////////////////////////
std::ostream &operator<<(std::ostream &out, const OMTFConfiguration &aConfig) {
  out << "nLayers(): " << aConfig.nLayers() << std::endl
      << " nHitsPerLayer(): " << aConfig.nHitsPerLayer() << std::endl
      << " nRefLayers(): " << aConfig.nRefLayers() << std::endl
      << " nPdfAddrBits: " << aConfig.nPdfAddrBits() << std::endl
      << " nPdfValBits: " << aConfig.nPdfValBits() << std::endl
      << " nPhiBins(): " << aConfig.nPhiBins() << std::endl
      << " nPdfAddrBits(): " << aConfig.nPdfAddrBits() << std::endl
      << std::endl;

  for (unsigned int iProcessor = 0; iProcessor < aConfig.nProcessors(); ++iProcessor) {
    out << "Processor: " << iProcessor;
    for (unsigned int iRefLayer = 0; iRefLayer < aConfig.nRefLayers(); ++iRefLayer) {
      out << " " << aConfig.processorPhiVsRefLayer[iProcessor][iRefLayer];
    }
    out << std::endl;
  }

  return out;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool OMTFConfiguration::isInRegionRange(int iPhiStart, unsigned int coneSize, int iPhi) const {
  if (iPhi < 0)
    iPhi += nPhiBins();
  if (iPhiStart < 0)
    iPhiStart += nPhiBins();

  if (iPhiStart + (int)coneSize < (int)nPhiBins()) {
    return iPhiStart <= iPhi && iPhiStart + (int)coneSize > iPhi;
  } else if (iPhi > (int)nPhiBins() / 2) {
    return iPhiStart <= iPhi;
  } else if (iPhi < (int)nPhiBins() / 2) {
    return iPhi < iPhiStart + (int)coneSize - (int)nPhiBins();
  }
  return false;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
unsigned int OMTFConfiguration::getRegionNumberFromMap(unsigned int iInput, unsigned int iRefLayer, int iPhi) const {
  for (unsigned int iRegion = 0; iRegion < nLogicRegions(); ++iRegion) {
    if (iPhi >= regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion].first &&
        iPhi <= regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion].second)
      return iRegion;
  }

  return 99;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
int OMTFConfiguration::globalPhiStart(unsigned int iProcessor) const {
  return *std::min_element(processorPhiVsRefLayer[iProcessor].begin(), processorPhiVsRefLayer[iProcessor].end());
}
///////////////////////////////////////////////
///////////////////////////////////////////////
uint32_t OMTFConfiguration::getLayerNumber(uint32_t rawId) const {
  uint32_t aLayer = 0;

  DetId detId(rawId);
  if (detId.det() != DetId::Muon) {
    std::cout << "PROBLEM: hit in unknown Det, detID: " << detId.det() << std::endl;
    return rawId;
  }

  switch (detId.subdetId()) {
    case MuonSubdetId::RPC: {
      RPCDetId aId(rawId);
      bool isBarrel = (aId.region() == 0);
      if (isBarrel)
        aLayer = aId.station() <= 2 ? 2 * (aId.station() - 1) + aId.layer() : aId.station() + 2;
      else
        aLayer = aId.station();
      aLayer += 10 * (!isBarrel);
      break;
    }
    case MuonSubdetId::DT: {
      DTChamberId dt(rawId);
      aLayer = dt.station();
      break;
    }
    case MuonSubdetId::CSC: {
      CSCDetId csc(rawId);
      aLayer = csc.station();
      if (csc.ring() == 2 && csc.station() == 1)
        aLayer = 1811;  //1811 = 2011 - 200, as we want to get 2011 for this chamber.
      if (csc.station() == 4)
        aLayer = 4;
      break;
    }
  }

  int hwNumber = aLayer + 100 * detId.subdetId();

  return hwNumber;
}
///////////////////////////////////////////////
// phiRad [-pi,pi]
int OMTFConfiguration::getProcScalePhi(unsigned int iProcessor, double phiRad) const {
  double phi15deg =
      M_PI / 3. * (iProcessor) + M_PI / 12.;  // "0" is 15degree moved cyclicaly to each processor, note [0,2pi]

  const double phiUnit = 2 * M_PI / nPhiBins();  //rad/unit

  // adjust [0,2pi] and [-pi,pi] to get deltaPhi difference properly
  switch (iProcessor + 1) {
    case 1:
      break;
    case 6: {
      phi15deg -= 2 * M_PI;
      break;
    }
    default: {
      if (phiRad < 0)
        phiRad += 2 * M_PI;
      break;
    }
  }

  // local angle in CSC halfStrip usnits
  return lround((phiRad - phi15deg) / phiUnit);  //FIXME lround or floor ???
}

double OMTFConfiguration::procHwPhiToGlobalPhi(int procHwPhi, int procHwPhi0) const {
  int globalHwPhi = foldPhi(procHwPhi + procHwPhi0);
  const double phiUnit = 2 * M_PI / nPhiBins();  //rad/unit
  return globalHwPhi * phiUnit;
}
/*int OMTFConfiguration::foldPhi(int phi) const {
  int phiBins = nPhiBins();
  if(phi > phiBins/2)
    return (phi - phiBins );
  else if(phi < -phiBins /2)
    return (phi + phiBins );

  return phi;
}*/

///////////////////////////////////////////////
///////////////////////////////////////////////
OMTFConfiguration::PatternPt OMTFConfiguration::getPatternPtRange(unsigned int patNum) const {
  if (patternPts.empty())
    throw cms::Exception("OMTFConfiguration::getPatternPtRange: patternPts vector not initialized");

  if (patNum > patternPts.size()) {
    throw cms::Exception("OMTFConfiguration::getPatternPtRange: patNum > patternPts.size()");
  }
  return patternPts[patNum];
}

unsigned int OMTFConfiguration::getPatternNum(double pt, int charge) const {
  //in LUT the charge is in convention 0 is -, 1 is + (so it is not the uGMT convention!!!)
  //so we change the charge here
  //if(charge == -1)
  //charge = 0;  //TODO but in the xml (and in GPs) the charge is +1 and -1, so it is important from where the patternPts is loaded FIXME!!!
  for (unsigned int iPat = 0; iPat < patternPts.size(); iPat++) {
    //std::cout<<"iPAt "<<iPat<<" ptFrom "<<getPatternPtRange(iPat).ptFrom<<" "<<getPatternPtRange(iPat).ptTo<<" "<<rawParams.chargeLUT()->data(iPat)<<std::endl;
    PatternPt patternPt = getPatternPtRange(iPat);
    if (pt >= patternPt.ptFrom && pt < patternPt.ptTo && charge == patternPt.charge)
      return iPat;
  }
  return 0;  //FIXME in this way if pt < 4GeV, the pattern = 0 is return , regardless of sign!
}

void OMTFConfiguration::printConfig() const {
  edm::LogVerbatim("OMTFReconstruction") << "OMTFConfiguration: " << std::endl;

  edm::LogVerbatim("OMTFReconstruction") << "rpcMaxClusterSize " << getRpcMaxClusterSize() << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "rpcMaxClusterCnt " << getRpcMaxClusterCnt() << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "rpcDropAllClustersIfMoreThanMax " << getRpcDropAllClustersIfMoreThanMax()
                                         << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "minDtPhiQuality " << getMinDtPhiQuality() << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "minDtPhiBQuality " << getMinDtPhiBQuality() << std::endl;

  edm::LogVerbatim("OMTFReconstruction") << "cscLctCentralBx_ " << cscLctCentralBx() << std::endl;

  edm::LogVerbatim("OMTFReconstruction") << "goldenPatternResultFinalizeFunction "
                                         << goldenPatternResultFinalizeFunction << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "noHitValueInPdf " << noHitValueInPdf << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "sorterType " << sorterType << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "ghostBusterType " << ghostBusterType << std::endl;
}
