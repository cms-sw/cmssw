#ifndef CalibCalorimetry_EBPhase2TPGTools_EcalEBPhase2TPParamProducer_h
#define CalibCalorimetry_EBPhase2TPGTools_EcalEBPhase2TPParamProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

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
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::vector<int> computeWeights(int type);

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

  const edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> theBarrelGeometryToken_;
  const std::string inFile_;
  const std::string outFile_;
  const int nSamplesToUse_;
  const bool useBXPlusOne_;
  const double phaseShift_;
  const unsigned int nWeightGroups_;
  const edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> theEcalTPGPedestals_Token_;

  gzFile out_file_;
  TGraph* thePulse_;
  TGraph* pulseDot_;

  const UInt_t NPoints_ = 1599;  //With the CMSSW pulse

  static constexpr float norm_ = 1 / 503.109;  // with the CMSSW pulse shape
  static constexpr float offset_ = 0.;         // with the CMSSW pulse shape
  int multToInt_ = 0x1000;

  int i2cSub_[2] = {0, 0};

  const double et_sat_;
  const double xtal_LSB_;
  const unsigned int binOfMaximum_;
  static constexpr double gainRatio_[ecalPh2::NGAINS] = {1., 10.};
};

#endif
