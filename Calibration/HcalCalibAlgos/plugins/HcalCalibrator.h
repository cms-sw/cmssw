#ifndef HCALCALIBRATOR_H
#define HCALCALIBRATOR_H

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

//-------------------
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TLorentzVector.h"
//---------------------

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalCalibrator : public edm::EDAnalyzer {
public:
  explicit HcalCalibrator(const edm::ParameterSet&);
  ~HcalCalibrator() override;

  // Added for running the CaloTower creation algorithm

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::string mOutputFile;
  std::string mInputFileList;

  std::string mCalibType;
  std::string mCalibMethod;
  double mMinTargetE;
  double mMaxTargetE;
  double mMinCellE;
  double mMinEOverP;
  double mMaxEOverP;
  double mMaxTrkEmE;

  double mMaxEtThirdJet;
  double mMinDPhiDiJets;
  bool mSumDepths;
  bool mSumSmallDepths;
  bool mCombinePhi;
  int mHbClusterSize;
  int mHeClusterSize;

  bool mUseConeClustering;
  double mMaxConeDist;

  int mCalibAbsIEtaMax;
  int mCalibAbsIEtaMin;

  double mMaxProbeJetEmFrac;
  double mMaxTagJetEmFrac;
  double mMaxTagJetAbsEta;
  double mMinTagJetEt;
  double mMinProbeJetAbsEta;

  std::string mPhiSymCorFileName;
  bool mApplyPhiSymCorFlag;

  std::string mOutputCorCoefFileName;
  std::string mHistoFileName;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;

  const CaloGeometry* mTheCaloGeometry;
  const HcalTopology* mTheHcalTopology;

  bool allowMissingInputs_;
};

#endif
