#ifndef ECALEBPHASE2TPPARAMPRODUCER_H
#define ECALEBPHASE2TPPARAMPRODUCER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <TGraph.h>
#include <TFile.h>
#include <TMatrix.h>
#include <zlib.h>

/** 
   \class EcalEBPhase2TPParamProducer
   \author L. Lutton, N. Marinelli - Univ. of Notre Dame
   \brief TPG Param Builder for Phase2
   *  
   */

class EcalEBPhase2TPParamProducer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalEBPhase2TPParamProducer(edm::ParameterSet const& pSet);
  ~EcalEBPhase2TPParamProducer() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void beginJob() override;

private:
  std::vector<int> computeWeights(
      int nSamples, bool useBXPlusOne, float phaseShift, unsigned int binOfMaximum, int type);

  void getNumericalDeriv(TGraph graph, TGraph& deriv);
  void fillFMat(std::vector<unsigned int> clockSampleSet,
                bool useThirdPulse,
                std::vector<float> sampleSet,
                std::vector<float> sampleDotSet,
                TMatrix& FMat,
                unsigned int binOfMaximum);
  void getGMatrix(TMatrix FMat, float scaleMatrixBy, TMatrix& GMat);
  void getPulseSampleSet(TGraph pulseGraph, float phaseShift, std::vector<float>& sampleSet);
  //
  bool computeLinearizerParam(double theta, double gainRatio, double calibCoeff, int& shift, int& mult);

  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> theBarrelGeometryToken_;
  std::string inFile_;
  std::string outFile_;
  int nSamplesToUse_;
  bool useBXPlusOne_;
  double phaseShift_;
  unsigned int nWeightGroups_;
  edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> theEcalTPGPedestals_Token_;

  const CaloSubdetectorGeometry* theBarrelGeometry_ = nullptr;
  const EcalLiteDTUPedestalsMap* theEcalTPPedestals_ = nullptr;
  const EcalLiteDTUPedestals* peds_ = nullptr;

  gzFile out_file_;
  TGraph* thePulse_;
  TGraph* pulseDot_;

  UInt_t NCrystals_ = 61200;
  const UInt_t NPoints_ = 1599;  //With the CMSSW pulse
  std::vector<float> sampleSet_;
  std::vector<float> sampleDotSet_;

  float norm_ = 1 / 503.109;  // with the CMSSW pulse shape

  float offset_ = 0.;  // with the CMSSW pulse shape
  int multToInt_ = 0x1000;

  int i2cSub_[2] = {0, 0};

  double Et_sat_;
  double xtal_LSB_;
  unsigned int binOfMaximum_;
  double calibCoeff_;
  double gMatCorr_;

  int mult_;
  int shift_;
  
  static constexpr double gainRatio_[ecalPh2::NGAINS] = {1., 10.};
};

#endif
