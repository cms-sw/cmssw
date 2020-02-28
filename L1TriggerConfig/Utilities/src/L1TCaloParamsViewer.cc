#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"
#include <iomanip>

class L1TCaloParamsViewer : public edm::EDAnalyzer {
private:
  bool printPUSParams;
  bool printTauCalibLUT;
  bool printTauCompressLUT;
  bool printJetCalibLUT;
  bool printJetCalibPar;
  bool printJetPUSPar;
  bool printJetCompressPtLUT;
  bool printJetCompressEtaLUT;
  bool printEgCalibLUT;
  bool printEgIsoLUT;
  bool printEtSumMetPUSLUT;
  bool printHfSF;
  bool printHcalSF;
  bool printEcalSF;
  bool printEtSumEttPUSLUT;
  bool printEtSumEcalSumPUSLUT;
  bool printMetCalibrationLUT;
  bool printMetHFCalibrationLUT;
  bool printMetPhiCalibrationLUT;
  bool printMetHFPhiCalibrationLUT;
  bool printEtSumEttCalibrationLUT;
  bool printEtSumEcalSumCalibrationLUT;

  bool useStage2Rcd;

  std::string hash(void* buf, size_t len) const;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TCaloParamsViewer(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    printPUSParams = pset.getUntrackedParameter<bool>("printPUSParams", false);
    printTauCalibLUT = pset.getUntrackedParameter<bool>("printTauCalibLUT", false);
    printTauCompressLUT = pset.getUntrackedParameter<bool>("printTauCompressLUT", false);
    printJetCalibLUT = pset.getUntrackedParameter<bool>("printJetCalibLUT", false);
    printJetCalibPar = pset.getUntrackedParameter<bool>("printJetCalibParams", false);
    printJetPUSPar = pset.getUntrackedParameter<bool>("printJetPUSPar", false);
    printJetCompressPtLUT = pset.getUntrackedParameter<bool>("printJetCompressPtLUT", false);
    printJetCompressEtaLUT = pset.getUntrackedParameter<bool>("printJetCompressEtaLUT", false);
    printEgCalibLUT = pset.getUntrackedParameter<bool>("printEgCalibLUT", false);
    printEgIsoLUT = pset.getUntrackedParameter<bool>("printEgIsoLUT", false);
    printEtSumMetPUSLUT = pset.getUntrackedParameter<bool>("printEtSumMetPUSLUT", false);
    printHfSF = pset.getUntrackedParameter<bool>("printHfSF", false);
    printHcalSF = pset.getUntrackedParameter<bool>("printHcalSF", false);
    printEcalSF = pset.getUntrackedParameter<bool>("printEcalSF", false);
    printEtSumEttPUSLUT = pset.getUntrackedParameter<bool>("printEtSumEttPUSLUT", false);
    printEtSumEcalSumPUSLUT = pset.getUntrackedParameter<bool>("printEtSumEcalSumPUSLUT", false);
    printMetCalibrationLUT = pset.getUntrackedParameter<bool>("printMetCalibrationLUT", false);
    printMetHFCalibrationLUT = pset.getUntrackedParameter<bool>("printMetHFCalibrationLUT", false);
    printEtSumEttCalibrationLUT = pset.getUntrackedParameter<bool>("printEtSumEttCalibrationLUT", false);
    printEtSumEcalSumCalibrationLUT = pset.getUntrackedParameter<bool>("printEtSumEcalSumCalibrationLUT", false);

    useStage2Rcd = pset.getUntrackedParameter<bool>("useStage2Rcd", false);
  }

  ~L1TCaloParamsViewer(void) override {}
};

#include <openssl/sha.h>
#include <cmath>
#include <iostream>
using namespace std;

std::string L1TCaloParamsViewer::hash(void* buf, size_t len) const {
  char tmp[SHA_DIGEST_LENGTH * 2 + 1];
  bzero(tmp, sizeof(tmp));
  SHA_CTX ctx;
  if (!SHA1_Init(&ctx))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 initialization error";

  if (!SHA1_Update(&ctx, buf, len))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 processing error";

  unsigned char hash[SHA_DIGEST_LENGTH];
  if (!SHA1_Final(hash, &ctx))
    throw cms::Exception("L1TCaloParamsViewer::hash") << "SHA1 finalization error";

  // re-write bytes in hex
  for (unsigned int i = 0; i < 20; i++)
    ::sprintf(&tmp[i * 2], "%02x", hash[i]);

  tmp[20 * 2] = 0;
  return std::string(tmp);
}

void L1TCaloParamsViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<l1t::CaloParams> handle1;
  if (useStage2Rcd)
    evSetup.get<L1TCaloStage2ParamsRcd>().get(handle1);
  else
    evSetup.get<L1TCaloParamsRcd>().get(handle1);

  std::shared_ptr<l1t::CaloParams> ptr(new l1t::CaloParams(*(handle1.product())));

  l1t::CaloParamsHelper* ptr1 = nullptr;
  ptr1 = (l1t::CaloParamsHelper*)(&(*ptr));

  edm::LogInfo("") << "L1TCaloParamsViewer:";

  cout << endl << " Towers: " << endl;
  cout << "  towerLsbH=       " << ptr1->towerLsbH() << endl;
  cout << "  towerLsbE=       " << ptr1->towerLsbE() << endl;
  cout << "  towerLsbSum=     " << ptr1->towerLsbSum() << endl;
  cout << "  towerNBitsH=     " << ptr1->towerNBitsH() << endl;
  cout << "  towerNBitsE=     " << ptr1->towerNBitsE() << endl;
  cout << "  towerNBitsSum=   " << ptr1->towerNBitsSum() << endl;
  cout << "  towerNBitsRatio= " << ptr1->towerNBitsRatio() << endl;
  cout << "  towerMaskE=      " << ptr1->towerMaskE() << endl;
  cout << "  towerMaskH=      " << ptr1->towerMaskH() << endl;
  cout << "  towerMaskSum=    " << ptr1->towerMaskSum() << endl;
  cout << "  towerEncoding=    " << ptr1->doTowerEncoding() << endl;

  cout << endl << " Regions: " << endl;
  cout << "  regionLsb=       " << ptr1->regionLsb() << endl;
  cout << "  regionPUSType=   " << ptr1->regionPUSType() << endl;
  cout << "  regionPUSParams= [" << ptr1->regionPUSParams().size() << "] ";
  float pusParams[ptr1->regionPUSParams().size()];
  for (unsigned int i = 0; i < ptr1->regionPUSParams().size(); i++) {
    pusParams[i] = ceil(2 * ptr1->regionPUSParams()[i]);
    if (printPUSParams)
      cout << "   " << ceil(2 * pusParams[i]) << endl;
  }

  if (!ptr1->regionPUSParams().empty())
    cout << hash(pusParams, sizeof(float) * ptr1->regionPUSParams().size()) << endl;
  else
    cout << endl;

  if (!ptr1->regionPUSLUT()->empty()) {
    cout << "  regionPUSLUT=         [" << ptr1->regionPUSLUT()->maxSize() << "] ";
    int regionPUSLUT[ptr1->regionPUSLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->regionPUSLUT()->maxSize(); i++)
      regionPUSLUT[i] = ptr1->regionPUSLUT()->data(i);
    cout << hash(regionPUSLUT, sizeof(int) * ptr1->regionPUSLUT()->maxSize()) << endl;
  } else {
    cout << "  regionPUSLUT=         [0]" << endl;
  }

  cout << "  pileUpTowerThreshold= " << ptr1->pileUpTowerThreshold() << endl;

  cout << endl << " EG: " << endl;
  cout << "  egLsb=                  " << ptr1->egLsb() << endl;
  cout << "  egSeedThreshold=        " << ptr1->egSeedThreshold() << endl;
  cout << "  egNeighbourThreshold=   " << ptr1->egNeighbourThreshold() << endl;
  cout << "  egHcalThreshold=        " << ptr1->egHcalThreshold() << endl;

  if (!ptr1->egTrimmingLUT()->empty()) {
    cout << "  egTrimmingLUT=          [" << ptr1->egTrimmingLUT()->maxSize() << "] " << flush;
    int egTrimming[ptr1->egTrimmingLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egTrimmingLUT()->maxSize(); i++)
      egTrimming[i] = ptr1->egTrimmingLUT()->data(i);
    cout << hash(egTrimming, sizeof(int) * ptr1->egTrimmingLUT()->maxSize()) << endl;
  } else {
    cout << "  egTrimmingLUT=          [0] " << endl;
  }

  cout << "  egMaxHcalEt=            " << ptr1->egMaxHcalEt() << endl;
  cout << "  egMaxPtHOverE=          " << ptr1->egMaxPtHOverE() << endl;
  cout << "  egMinPtJetIsolation=    " << ptr1->egMinPtJetIsolation() << endl;
  cout << "  egMaxPtJetIsolation=    " << ptr1->egMaxPtJetIsolation() << endl;
  cout << "  egMinPtHOverEIsolation= " << ptr1->egMinPtHOverEIsolation() << endl;
  cout << "  egMaxPtHOverEIsolation= " << ptr1->egMaxPtHOverEIsolation() << endl;

  if (!ptr1->egMaxHOverELUT()->empty()) {
    cout << "  egMaxHOverELUT=         [" << ptr1->egMaxHOverELUT()->maxSize() << "] ";
    int egMaxHOverE[ptr1->egMaxHOverELUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egMaxHOverELUT()->maxSize(); i++)
      egMaxHOverE[i] = ptr1->egMaxHOverELUT()->data(i);
    cout << hash(egMaxHOverE, sizeof(int) * ptr1->egMaxHOverELUT()->maxSize()) << endl;
  } else {
    cout << "  egMaxHOverELUT=         [0]" << endl;
  }

  if (!ptr1->egCompressShapesLUT()->empty()) {
    cout << "  egCompressShapesLUT=    [" << ptr1->egCompressShapesLUT()->maxSize() << "] ";
    int egCompressShapes[ptr1->egCompressShapesLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egCompressShapesLUT()->maxSize(); i++)
      egCompressShapes[i] = ptr1->egCompressShapesLUT()->data(i);
    cout << hash(egCompressShapes, sizeof(int) * ptr1->egCompressShapesLUT()->maxSize()) << endl;
  } else {
    cout << "  egCompressShapesLUT=    [0]" << endl;
  }

  cout << "  egShapeIdType=          " << ptr1->egShapeIdType() << endl;
  cout << "  egShapeIdVersion=       " << ptr1->egShapeIdVersion() << endl;
  if (!ptr1->egShapeIdLUT()->empty()) {
    cout << "  egShapeIdLUT=           [" << ptr1->egShapeIdLUT()->maxSize() << "] " << flush;
    int egShapeId[ptr1->egShapeIdLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egShapeIdLUT()->maxSize(); i++)
      egShapeId[i] = ptr1->egShapeIdLUT()->data(i);
    cout << hash(egShapeId, sizeof(int) * ptr1->egShapeIdLUT()->maxSize()) << endl;
  } else {
    cout << "  egShapeIdLUT=           [0]" << endl;
  }

  cout << "  egBypassEGVetos=        " << ptr1->egBypassEGVetos() << endl;
  cout << "  egBypassShape=          " << ptr1->egBypassShape() << endl;
  cout << "  egBypassExtHoverE=      " << ptr1->egBypassExtHOverE() << endl;
  cout << "  egBypassECALFG=         " << ptr1->egBypassECALFG() << endl;
  cout << "  egHOverEcutBarrel=      " << ptr1->egHOverEcutBarrel() << endl;
  cout << "  egHOverEcutEndcap=      " << ptr1->egHOverEcutEndcap() << endl;

  cout << "  egPUSType=              " << ptr1->egPUSType() << endl;

  cout << "  egIsolationType=        " << ptr1->egIsolationType() << endl;
  if (!ptr1->egIsolationLUT()->empty()) {
    cout << "  egIsoLUT=               [" << ptr1->egIsolationLUT()->maxSize() << "] " << flush;
    int egIsolation[ptr1->egIsolationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egIsolationLUT()->maxSize(); i++)
      egIsolation[i] = ptr1->egIsolationLUT()->data(i);
    cout << hash(egIsolation, sizeof(int) * ptr1->egIsolationLUT()->maxSize()) << endl;
    if (printEgIsoLUT)
      for (unsigned int i = 0; i < ptr1->egIsolationLUT()->maxSize(); i++)
        cout << i << " " << egIsolation[i] << endl;
  } else {
    cout << "  egIsoLUT=               [0]" << endl;
  }
  if (!ptr1->egIsolationLUT2()->empty()) {
    cout << "  egIsoLUT2=              [" << ptr1->egIsolationLUT2()->maxSize() << "] " << flush;
    int egIsolation2[ptr1->egIsolationLUT2()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egIsolationLUT2()->maxSize(); i++)
      egIsolation2[i] = ptr1->egIsolationLUT2()->data(i);
    cout << hash(egIsolation2, sizeof(int) * ptr1->egIsolationLUT2()->maxSize()) << endl;
    if (printEgIsoLUT)
      for (unsigned int i = 0; i < ptr1->egIsolationLUT2()->maxSize(); i++)
        cout << i << " " << egIsolation2[i] << endl;
  } else {
    cout << "  egIsoLUT2=              [0]" << endl;
  }

  cout << "  egIsoAreaNrTowersEta=   " << ptr1->egIsoAreaNrTowersEta() << endl;
  cout << "  egIsoAreaNrTowersPhi=   " << ptr1->egIsoAreaNrTowersPhi() << endl;
  cout << "  egIsoVetoNrTowersPhi=   " << ptr1->egIsoVetoNrTowersPhi() << endl;
  cout << "  egPUSParams=            [" << ptr1->egPUSParams().size() << "] " << flush;
  float egPUSParams[ptr1->egPUSParams().size()];
  for (unsigned int i = 0; i < ptr1->egPUSParams().size(); i++)
    egPUSParams[i] = ptr1->egPUSParams()[i];

  if (!ptr1->egPUSParams().empty())
    cout << hash(egPUSParams, sizeof(float) * ptr1->egPUSParams().size()) << endl;
  else
    cout << endl;

  cout << "  egCalibrationParams=    [" << ptr1->egCalibrationParams().size() << "] " << flush;
  double egCalibrationParams[ptr1->egCalibrationParams().size()];
  for (unsigned int i = 0; i < ptr1->egCalibrationParams().size(); i++)
    egCalibrationParams[i] = ptr1->egCalibrationParams()[i];

  if (!ptr1->egCalibrationParams().empty())
    cout << hash(egCalibrationParams, sizeof(double) * ptr1->egCalibrationParams().size()) << endl;
  else
    cout << endl;

  cout << "  egCalibrationType=      " << ptr1->egCalibrationType() << endl;
  cout << "  egCalibrationVersion=   " << ptr1->egCalibrationVersion() << endl;
  if (!ptr1->egCalibrationLUT()->empty()) {
    cout << "  egCalibrationLUT=       [" << ptr1->egCalibrationLUT()->maxSize() << "] " << flush;
    int egCalibration[ptr1->egCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->egCalibrationLUT()->maxSize(); i++)
      egCalibration[i] = ptr1->egCalibrationLUT()->data(i);
    cout << hash(egCalibration, sizeof(int) * ptr1->egCalibrationLUT()->maxSize()) << endl;
    if (printEgCalibLUT)
      for (unsigned int i = 0; i < ptr1->egCalibrationLUT()->maxSize(); i++)
        cout << i << " " << egCalibration[i] << endl;
  } else {
    cout << "  egCalibrationLUT=       [0]" << endl;
  }

  cout << endl << " Tau: " << endl;
  cout << "  tauLsb=                 " << ptr1->tauLsb() << endl;
  //cout<<"  tauSeedThreshold=       "<<ptr1->tauSeedThreshold()<<endl;
  //cout<<"  tauNeighbourThreshold=  "<<ptr1->tauNeighbourThreshold()<<endl;
  cout << "  tauMaxPtTauVeto=        " << ptr1->tauMaxPtTauVeto() << endl;
  cout << "  tauMinPtJetIsolationB=  " << ptr1->tauMinPtJetIsolationB() << endl;
  cout << "  tauPUSType=             " << ptr1->tauPUSType() << endl;
  cout << "  tauMaxJetIsolationB=    " << ptr1->tauMaxJetIsolationB() << endl;
  cout << "  tauMaxJetIsolationA=    " << ptr1->tauMaxJetIsolationA() << endl;
  cout << "  tauIsoAreaNrTowersEta=  " << ptr1->tauIsoAreaNrTowersEta() << endl;
  cout << "  tauIsoAreaNrTowersPhi=  " << ptr1->tauIsoAreaNrTowersPhi() << endl;
  cout << "  tauIsoVetoNrTowersPhi=  " << ptr1->tauIsoVetoNrTowersPhi() << endl;
  if (!ptr1->tauIsolationLUT()->empty()) {
    cout << "  tauIsoLUT=              [" << ptr1->tauIsolationLUT()->maxSize() << "] " << flush;
    int tauIsolation[ptr1->tauIsolationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->tauIsolationLUT()->maxSize(); i++)
      tauIsolation[i] = ptr1->tauIsolationLUT()->data(i);
    cout << hash(tauIsolation, sizeof(int) * ptr1->tauIsolationLUT()->maxSize()) << endl;
  } else {
    cout << "  tauIsoLUT=              [0]" << endl;
  }
  if (!ptr1->tauIsolationLUT2()->empty()) {
    cout << "  tauIsoLUT2=             [" << ptr1->tauIsolationLUT2()->maxSize() << "] " << flush;
    int tauIsolation2[ptr1->tauIsolationLUT2()->maxSize()];
    for (unsigned int i = 0; i < ptr1->tauIsolationLUT2()->maxSize(); i++)
      tauIsolation2[i] = ptr1->tauIsolationLUT2()->data(i);
    cout << hash(tauIsolation2, sizeof(int) * ptr1->tauIsolationLUT2()->maxSize()) << endl;
  } else {
    cout << "  tauIsoLUT2=             [0]" << endl;
  }
  if (!ptr1->tauTrimmingShapeVetoLUT()->empty()) {
    cout << "  tauTrimmingShapeVetoLUT=[" << ptr1->tauTrimmingShapeVetoLUT()->maxSize() << "] " << flush;
    int tauTrimmingShapeVetoLUT[ptr1->tauTrimmingShapeVetoLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->tauTrimmingShapeVetoLUT()->maxSize(); i++)
      tauTrimmingShapeVetoLUT[i] = ptr1->tauTrimmingShapeVetoLUT()->data(i);
    cout << hash(tauTrimmingShapeVetoLUT, sizeof(int) * ptr1->tauTrimmingShapeVetoLUT()->maxSize()) << endl;
  } else {
    cout << "  tauTrimmingShapeVetoLUT=[0]" << endl;
  }

  if (!ptr1->tauCalibrationLUT()->empty()) {
    cout << "  tauCalibrationLUT=      [" << ptr1->tauCalibrationLUT()->maxSize() << "] " << flush;
    int tauCalibration[512];                //ptr1->tauCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < 512; i++)  //ptr1->tauCalibrationLUT()->maxSize(); i++)
      tauCalibration[i] = ptr1->tauCalibrationLUT()->data(i);
    cout << hash(tauCalibration, sizeof(int) * 512 /*ptr1->tauCalibrationLUT()->maxSize() */) << endl;

    if (printTauCalibLUT)
      for (unsigned int i = 0; i < 512 /*ptr1->tauCalibrationLUT()->maxSize()*/; i++)
        cout << i << " " << tauCalibration[i] << endl;

  } else {
    cout << "  tauCalibrationLUT=      [0]" << endl;
  }

  cout << "  tauCalibrationType=     " << ptr1->tauCalibrationType() << endl;

  cout << "  tauCalibrationParams=   [" << ptr1->tauCalibrationParams().size() << "] " << flush;
  double tauCalibrationParams[ptr1->tauCalibrationParams().size()];
  for (unsigned int i = 0; i < ptr1->tauCalibrationParams().size(); i++)
    tauCalibrationParams[i] = ptr1->tauCalibrationParams()[i];

  if (!ptr1->tauCalibrationParams().empty())
    cout << hash(tauCalibrationParams, sizeof(double) * ptr1->tauCalibrationParams().size()) << endl;
  else
    cout << endl;

  if (!ptr1->tauCompressLUT()->empty()) {
    cout << "  tauCompressLUT=         [" << ptr1->tauCompressLUT()->maxSize() << "] " << flush;
    int tauCompress[ptr1->tauCompressLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->tauCompressLUT()->maxSize(); i++)
      tauCompress[i] = ptr1->tauCompressLUT()->data(i);
    cout << hash(tauCompress, sizeof(int) * ptr1->tauCompressLUT()->maxSize()) << endl;

    if (printTauCompressLUT)
      for (unsigned int i = 0; i < ptr1->tauCompressLUT()->maxSize(); i++)
        cout << i << " " << tauCompress[i] << endl;

  } else {
    cout << "  tauCompressLUT=         [0]" << endl;
  }

  if (!ptr1->tauEtToHFRingEtLUT()->empty()) {
    cout << "  tauEtToHFRingEtLUT=     [" << ptr1->tauEtToHFRingEtLUT()->maxSize() << "] " << flush;
    int tauEtToHFRingEt[ptr1->tauEtToHFRingEtLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->tauEtToHFRingEtLUT()->maxSize(); i++)
      tauEtToHFRingEt[i] = ptr1->tauEtToHFRingEtLUT()->data(i);

    cout << hash(tauEtToHFRingEt, sizeof(int) * ptr1->tauEtToHFRingEtLUT()->maxSize()) << endl;
  } else {
    cout << "  tauEtToHFRingEtLUT=     [0]" << endl;
  }

  cout << "  isoTauEtaMin=           " << ptr1->isoTauEtaMin() << endl;
  cout << "  isoTauEtaMax=           " << ptr1->isoTauEtaMax() << endl;
  cout << "  tauPUSParams=           [" << ptr1->tauPUSParams().size() << "] " << flush;
  float tauPUSParams[ptr1->tauPUSParams().size()];
  for (unsigned int i = 0; i < ptr1->tauPUSParams().size(); i++)
    tauPUSParams[i] = ptr1->tauPUSParams()[i];

  if (!ptr1->tauPUSParams().empty())
    cout << hash(tauPUSParams, sizeof(float) * ptr1->tauPUSParams().size()) << endl;
  else
    cout << endl;

  cout << endl << " Jets: " << endl;
  cout << "  jetLsb=                 " << ptr1->jetLsb() << endl;
  cout << "  jetSeedThreshold=       " << ptr1->jetSeedThreshold() << endl;
  cout << "  jetNeighbourThreshold=  " << ptr1->jetNeighbourThreshold() << endl;
  cout << "  jetRegionMask=          " << ptr1->jetRegionMask() << endl;
  cout << "  jetBypassPUS=           " << ptr1->jetBypassPUS() << endl;
  //cout<<"  jetPUSType=             "<<ptr1->jetPUSType()<<endl;
  cout << "  jetPUSUsePhiRing=       " << ptr1->jetPUSUsePhiRing() << endl;
  cout << "  jetCalibrationType=     " << ptr1->jetCalibrationType() << endl;
  //cout<<"  jetCalibrationParams=   ["<<ptr1->jetCalibrationParams().size()<<"] "<<flush;
  //float jetCalibrationParams[ptr1->jetCalibrationParams().size()]; // deliberately drop double precision
  //for(unsigned int i=0; i<ptr1->jetCalibrationParams().size(); i++) jetCalibrationParams[i] = ptr1->jetCalibrationParams()[i];

  /*if( !ptr1->jetCalibrationParams().empty() ){
        cout << hash( jetCalibrationParams, sizeof(float)*ptr1->jetCalibrationParams().size() ) << endl;
        if( printJetCalibPar )
            for(unsigned int i=0; i<ptr1->jetCalibrationParams().size(); i++)
                cout<<i<<" " << std::setprecision(14) << jetCalibrationParams[i]<<endl;

    } else cout<<endl;

    cout<<"  jetPUSParams=           ["<<ptr1->jetPUSParams().size()<<"] "<<flush;
    float jetPUSParams[ptr1->jetPUSParams().size()]; // deliberately drop double precision
    for(unsigned int i=0; i<ptr1->jetPUSParams().size(); i++) jetPUSParams[i] = ptr1->jetPUSParams()[i];
    if( !ptr1->jetPUSParams().empty() ){
        cout << hash( jetPUSParams, sizeof(float)*ptr1->jetPUSParams().size() ) << endl;
        if( printJetPUSPar )
            for(unsigned int i=0; i<ptr1->jetPUSParams().size(); i++)
                cout<<i<<" " << std::setprecision(14) << jetPUSParams[i]<<endl;

    } else cout<<endl;
    */

  if (!ptr1->jetCalibrationLUT()->empty()) {
    cout << "  jetCalibrationLUT=      [" << ptr1->jetCalibrationLUT()->maxSize() << "] " << flush;
    int jetCalibration[ptr1->jetCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->jetCalibrationLUT()->maxSize(); i++)
      jetCalibration[i] = ptr1->jetCalibrationLUT()->data(i);

    cout << hash(jetCalibration, sizeof(int) * ptr1->jetCalibrationLUT()->maxSize()) << endl;

    if (printJetCalibLUT)
      for (unsigned int i = 0; i < ptr1->jetCalibrationLUT()->maxSize(); i++)
        cout << i << " " << jetCalibration[i] << endl;

  } else {
    cout << "  jetCalibrationLUT=      [0]" << endl;
  }

  if (!ptr1->jetCompressPtLUT()->empty()) {
    cout << "  jetCompressPtLUT=       [" << ptr1->jetCompressPtLUT()->maxSize() << "] " << flush;
    int jetCompressPt[ptr1->jetCompressPtLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->jetCompressPtLUT()->maxSize(); i++)
      jetCompressPt[i] = ptr1->jetCompressPtLUT()->data(i);

    cout << hash(jetCompressPt, sizeof(int) * ptr1->jetCompressPtLUT()->maxSize()) << endl;

    if (printJetCompressPtLUT)
      for (unsigned int i = 0; i < ptr1->jetCompressPtLUT()->maxSize(); i++)
        cout << i << " " << jetCompressPt[i] << endl;

  } else {
    cout << "  jetCompressPtLUT=       [0]" << endl;
  }

  if (!ptr1->jetCompressEtaLUT()->empty()) {
    cout << "  jetCompressEtaLUT=      [" << ptr1->jetCompressEtaLUT()->maxSize() << "] " << flush;
    int jetCompressEta[ptr1->jetCompressEtaLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->jetCompressEtaLUT()->maxSize(); i++)
      jetCompressEta[i] = ptr1->jetCompressEtaLUT()->data(i);

    cout << hash(jetCompressEta, sizeof(int) * ptr1->jetCompressEtaLUT()->maxSize()) << endl;

    if (printJetCompressEtaLUT)
      for (unsigned int i = 0; i < ptr1->jetCompressEtaLUT()->maxSize(); i++)
        cout << i << " " << jetCompressEta[i] << endl;

  } else {
    cout << "  jetCompressEtaLUT=      [0]" << endl;
  }

  cout << endl << " Sums: " << endl;
  unsigned int nEntities = 0;
  cout << "  etSumLsb=               " << ptr1->etSumLsb() << endl;
  cout << "  etSumEtaMin=            [";
  for (unsigned int i = 0; ptr1->etSumEtaMin(i) > 0.001; i++)
    cout << (i == 0 ? "" : ",") << ptr1->etSumEtaMin(i);
  cout << "]" << endl;
  cout << "  etSumEtaMax=            [";
  for (unsigned int i = 0; ptr1->etSumEtaMax(i) > 0.001; i++, nEntities++)
    cout << (i == 0 ? "" : ",") << ptr1->etSumEtaMax(i);
  cout << "]" << endl;
  cout << "  etSumEtThreshold=       [";
  for (unsigned int i = 0; i < nEntities; i++)
    cout << (i == 0 ? "" : ",") << ptr1->etSumEtThreshold(i);
  cout << "]" << endl;

  cout << "  etSumBypassMetPUS=      " << ptr1->etSumBypassMetPUS() << endl;
  cout << "  etSumBypassEttPUS=      " << ptr1->etSumBypassEttPUS() << endl;
  cout << "  etSumBypassEcalSumPUS   " << ptr1->etSumBypassEcalSumPUS() << endl;

  cout << "  etSumMetPUSType=        " << ptr1->etSumMetPUSType() << endl;
  cout << "  etSumEttPUSType=        " << ptr1->etSumEttPUSType() << endl;
  cout << "  etSumEcalSumPUSType=    " << ptr1->etSumEcalSumPUSType() << endl;

  cout << "  etSumCentralityUpper=   [";
  for (unsigned int i = 0; ptr1->etSumCentUpper(i) > 0.001; i++)
    cout << (i == 0 ? "" : ",") << ptr1->etSumCentUpper(i);
  cout << "]" << endl;
  cout << "  etSumCentralityLower=   [";
  for (unsigned int i = 0; ptr1->etSumCentLower(i) > 0.001; i++)
    cout << (i == 0 ? "" : ",") << ptr1->etSumCentLower(i);
  cout << "]" << endl;

  cout << "  metCalibrationType=  " << ptr1->metCalibrationType() << endl;
  cout << "  metHFCalibrationType=  " << ptr1->metHFCalibrationType() << endl;
  cout << "  etSumEttCalibrationType=" << ptr1->etSumEttCalibrationType() << endl;
  cout << "  etSumEcalSumCalibrationType=" << ptr1->etSumEcalSumCalibrationType() << endl;

  if (!ptr1->etSumMetPUSLUT()->empty()) {
    cout << "  etSumMetPUSLUT=         [" << ptr1->etSumMetPUSLUT()->maxSize() << "] " << flush;
    int etSumMetPUSLUT[ptr1->etSumMetPUSLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->etSumMetPUSLUT()->maxSize(); i++)
      etSumMetPUSLUT[i] = ptr1->etSumMetPUSLUT()->data(i);

    cout << hash(etSumMetPUSLUT, sizeof(int) * ptr1->etSumMetPUSLUT()->maxSize()) << endl;

    if (printEtSumMetPUSLUT)
      for (unsigned int i = 0; i < ptr1->etSumMetPUSLUT()->maxSize(); i++)
        cout << i << " " << etSumMetPUSLUT[i] << endl;

  } else {
    cout << "  etSumMetPUSLUT=         [0]" << endl;
  }

  if (!ptr1->etSumEttPUSLUT()->empty()) {
    cout << "  etSumEttPUSLUT=         [" << ptr1->etSumEttPUSLUT()->maxSize() << "] " << flush;
    int etSumEttPUSLUT[ptr1->etSumEttPUSLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->etSumEttPUSLUT()->maxSize(); i++)
      etSumEttPUSLUT[i] = ptr1->etSumEttPUSLUT()->data(i);

    cout << hash(etSumEttPUSLUT, sizeof(int) * ptr1->etSumEttPUSLUT()->maxSize()) << endl;

    if (printEtSumEttPUSLUT)
      for (unsigned int i = 0; i < ptr1->etSumEttPUSLUT()->maxSize(); i++)
        cout << i << " " << etSumEttPUSLUT[i] << endl;

  } else {
    cout << "  etSumEttPUSLUT=         [0]" << endl;
  }

  if (!ptr1->etSumEcalSumPUSLUT()->empty()) {
    cout << "  etSumEcalSumPUSLUT=     [" << ptr1->etSumEcalSumPUSLUT()->maxSize() << "] " << flush;
    int etSumEcalSumPUSLUT[ptr1->etSumEcalSumPUSLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->etSumEcalSumPUSLUT()->maxSize(); i++)
      etSumEcalSumPUSLUT[i] = ptr1->etSumEcalSumPUSLUT()->data(i);

    cout << hash(etSumEcalSumPUSLUT, sizeof(int) * ptr1->etSumEcalSumPUSLUT()->maxSize()) << endl;

    if (printEtSumEcalSumPUSLUT)
      for (unsigned int i = 0; i < ptr1->etSumEcalSumPUSLUT()->maxSize(); i++)
        cout << i << " " << etSumEcalSumPUSLUT[i] << endl;

  } else {
    cout << "  etSumEcalSumPUSLUT=     [0]" << endl;
  }

  if (!ptr1->metCalibrationLUT()->empty()) {
    cout << "  metCalibrationLUT=   [" << ptr1->metCalibrationLUT()->maxSize() << "] " << flush;
    int metCalibrationLUT[ptr1->metCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->metCalibrationLUT()->maxSize(); i++)
      metCalibrationLUT[i] = ptr1->metCalibrationLUT()->data(i);

    cout << hash(metCalibrationLUT, sizeof(int) * ptr1->metCalibrationLUT()->maxSize()) << endl;

    if (printMetCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->metCalibrationLUT()->maxSize(); i++)
        cout << i << " " << metCalibrationLUT[i] << endl;

  } else {
    cout << "  metCalibrationLUT=   [0]" << endl;
  }

  if (!ptr1->metHFCalibrationLUT()->empty()) {
    cout << "  metHFCalibrationLUT=   [" << ptr1->metHFCalibrationLUT()->maxSize() << "] " << flush;
    int metHFCalibrationLUT[ptr1->metHFCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->metHFCalibrationLUT()->maxSize(); i++)
      metHFCalibrationLUT[i] = ptr1->metHFCalibrationLUT()->data(i);

    cout << hash(metHFCalibrationLUT, sizeof(int) * ptr1->metHFCalibrationLUT()->maxSize()) << endl;

    if (printMetHFCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->metHFCalibrationLUT()->maxSize(); i++)
        cout << i << " " << metHFCalibrationLUT[i] << endl;

  } else {
    cout << "  metHFCalibrationLUT=   [0]" << endl;
  }

  if (!ptr1->metPhiCalibrationLUT()->empty()) {
    cout << "  metPhiCalibrationLUT=   [" << ptr1->metPhiCalibrationLUT()->maxSize() << "] " << flush;
    int metPhiCalibrationLUT[ptr1->metPhiCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->metPhiCalibrationLUT()->maxSize(); i++)
      metPhiCalibrationLUT[i] = ptr1->metPhiCalibrationLUT()->data(i);

    cout << hash(metPhiCalibrationLUT, sizeof(int) * ptr1->metPhiCalibrationLUT()->maxSize()) << endl;

    if (printMetPhiCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->metPhiCalibrationLUT()->maxSize(); i++)
        cout << i << " " << metPhiCalibrationLUT[i] << endl;

  } else {
    cout << "  metPhiCalibrationLUT=   [0]" << endl;
  }

  if (!ptr1->metHFPhiCalibrationLUT()->empty()) {
    cout << "  metHFPhiCalibrationLUT=   [" << ptr1->metHFPhiCalibrationLUT()->maxSize() << "] " << flush;
    int metHFPhiCalibrationLUT[ptr1->metHFPhiCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->metHFPhiCalibrationLUT()->maxSize(); i++)
      metHFPhiCalibrationLUT[i] = ptr1->metHFPhiCalibrationLUT()->data(i);

    cout << hash(metHFPhiCalibrationLUT, sizeof(int) * ptr1->metHFPhiCalibrationLUT()->maxSize()) << endl;

    if (printMetHFCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->metHFPhiCalibrationLUT()->maxSize(); i++)
        cout << i << " " << metHFPhiCalibrationLUT[i] << endl;

  } else {
    cout << "  metHFPhiCalibrationLUT=   [0]" << endl;
  }

  if (!ptr1->etSumEttCalibrationLUT()->empty()) {
    cout << "  etSumEttCalibrationLUT= [" << ptr1->etSumEttCalibrationLUT()->maxSize() << "] " << flush;
    int etSumEttCalibrationLUT[ptr1->etSumEttCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->etSumEttCalibrationLUT()->maxSize(); i++)
      etSumEttCalibrationLUT[i] = ptr1->etSumEttCalibrationLUT()->data(i);

    cout << hash(etSumEttCalibrationLUT, sizeof(int) * ptr1->etSumEttCalibrationLUT()->maxSize()) << endl;

    if (printEtSumEttCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->etSumEttCalibrationLUT()->maxSize(); i++)
        cout << i << " " << etSumEttCalibrationLUT[i] << endl;

  } else {
    cout << "  etSumEttCalibrationLUT= [0]" << endl;
  }

  if (!ptr1->etSumEcalSumCalibrationLUT()->empty()) {
    cout << "  etSumEcalSumCalibrationLUT=[" << ptr1->etSumEttCalibrationLUT()->maxSize() << "] " << flush;
    int etSumEcalSumCalibrationLUT[ptr1->etSumEcalSumCalibrationLUT()->maxSize()];
    for (unsigned int i = 0; i < ptr1->etSumEcalSumCalibrationLUT()->maxSize(); i++)
      etSumEcalSumCalibrationLUT[i] = ptr1->etSumEcalSumCalibrationLUT()->data(i);

    cout << hash(etSumEcalSumCalibrationLUT, sizeof(int) * ptr1->etSumEcalSumCalibrationLUT()->maxSize()) << endl;

    if (printEtSumEcalSumCalibrationLUT)
      for (unsigned int i = 0; i < ptr1->etSumEcalSumCalibrationLUT()->maxSize(); i++)
        cout << i << " " << etSumEcalSumCalibrationLUT[i] << endl;

  } else {
    cout << "  etSumEcalSumCalibrationLUT=[0]" << endl;
  }

  cout << endl << " HI centrality trigger: " << endl;
  cout << "  centralityLUT=          [";
  for (unsigned int i = 0; i < ptr1->centralityLUT()->maxSize(); i++)
    cout << (i == 0 ? "" : ",") << ptr1->centralityLUT()->data(i);
  cout << "]" << endl;

  std::vector<int> mbt = ptr1->minimumBiasThresholds();
  cout << "  minimumBiasThresholds=  [";
  for (unsigned int i = 0; i < mbt.size(); i++)
    cout << mbt[i];
  cout << "]" << endl;

  cout << endl << "centralityRegionMask() = " << ptr1->centralityRegionMask() << endl;
  cout << endl << "jetRegionMask() = " << ptr1->jetRegionMask() << endl;
  cout << endl << "tauRegionMask() = " << ptr1->tauRegionMask() << endl;

  cout << endl << " HI Q2 trigger: " << endl;
  cout << "  q2LUT=                  [";
  for (unsigned int i = 0; i < ptr1->q2LUT()->maxSize(); i++)
    cout << (i == 0 ? "" : ",") << ptr1->q2LUT()->data(i);
  cout << "]" << endl;

  cout << endl << " Layer1: " << endl;
  std::vector<double> ecalSF = ptr1->layer1ECalScaleFactors();
  cout << "  layer1ECalScaleFactors= [" << ecalSF.size() << "] " << flush;
  int _ecalSF[ecalSF.size()];
  for (unsigned int i = 0; i < ecalSF.size(); i++)
    _ecalSF[i] = int(ecalSF[i] * 100000.);
  cout << hash(_ecalSF, sizeof(int) * ecalSF.size()) << endl;
  if (printEcalSF) {
    cout << endl << "    [" << endl;
    for (unsigned int i = 0; i < ecalSF.size(); i++)
      cout << (i == 0 ? "" : ",") << int(ecalSF[i] * 1000.) / 1000.;
    cout << "]" << endl;
  }
  std::vector<double> hcalSF = ptr1->layer1HCalScaleFactors();
  cout << "  layer1HCalScaleFactors= [" << hcalSF.size() << "] " << flush;
  int _hcalSF[hcalSF.size()];
  for (unsigned int i = 0; i < hcalSF.size(); i++) {
    // round false precision
    //        double significand;
    //        int    exponent;
    //        significand = frexp( hcalSF[i],  &exponent );
    //         _hcalSF[i] = ldexp( int(significand*10000)/10000., exponent );
    _hcalSF[i] = int(hcalSF[i] * 100000.);
  }
  cout << hash(_hcalSF, sizeof(int) * hcalSF.size()) << endl;
  if (printHcalSF) {
    cout << endl << "    [" << endl;
    for (unsigned int i = 0; i < hcalSF.size(); i++)
      cout << (i == 0 ? "" : ",") << int(hcalSF[i] * 1000.) / 1000.;
    cout << "]" << endl;
  }
  std::vector<double> hfSF = ptr1->layer1HFScaleFactors();
  cout << "  layer1HFScaleFactors=   [" << hfSF.size() << "] " << flush;
  int _hfSF[hfSF.size()];
  for (unsigned int i = 0; i < hfSF.size(); i++)
    _hfSF[i] = int(hfSF[i] * 100000.);
  cout << hash(_hfSF, sizeof(int) * hfSF.size()) << endl;
  if (printHfSF) {
    cout << endl << "    [" << endl;
    for (unsigned int i = 0; i < hfSF.size(); i++)
      cout << (i == 0 ? "" : ",") << int(hfSF[i] * 1000.) / 1000.;
    cout << "]" << endl;
  }

  std::vector<int> ecalScaleET = ptr1->layer1ECalScaleETBins();
  cout << "  layer1ECalScaleETBins=  [";
  for (unsigned int i = 0; i < ecalScaleET.size(); i++)
    cout << (i == 0 ? "" : ",") << ecalScaleET[i];
  cout << "]" << endl;
  std::vector<int> hcalScaleET = ptr1->layer1HCalScaleETBins();
  cout << "  layer1HCalScaleETBins=  [";
  for (unsigned int i = 0; i < hcalScaleET.size(); i++)
    cout << (i == 0 ? "" : ",") << hcalScaleET[i];
  cout << "]" << endl;
  std::vector<int> hfScaleET = ptr1->layer1HFScaleETBins();
  cout << "  layer1HFScaleETBins=    [";
  for (unsigned int i = 0; i < hfScaleET.size(); i++)
    cout << (i == 0 ? "" : ",") << hfScaleET[i];
  cout << "]" << endl;

  std::vector<unsigned> layer1ECalScalePhi = ptr1->layer1ECalScalePhiBins();
  cout << "  layer1ECalScalePhi=     [";
  for (unsigned int i = 0; i < layer1ECalScalePhi.size(); i++)
    cout << (i == 0 ? "" : ",") << layer1ECalScalePhi[i];
  cout << "]" << endl;
  std::vector<unsigned> layer1HCalScalePhi = ptr1->layer1HCalScalePhiBins();
  cout << "  layer1HCalScalePhi=     [";
  for (unsigned int i = 0; i < layer1HCalScalePhi.size(); i++)
    cout << (i == 0 ? "" : ",") << layer1HCalScalePhi[i];
  cout << "]" << endl;
  std::vector<unsigned> layer1HFScalePhiBins = ptr1->layer1HFScalePhiBins();
  cout << "  layer1HFScalePhiBins=   [";
  for (unsigned int i = 0; i < layer1HFScalePhiBins.size(); i++)
    cout << (i == 0 ? "" : ",") << layer1HFScalePhiBins[i];
  cout << "]" << endl;

  //    std::vector<unsigned> layer1SecondStageLUT = ptr1->layer1SecondStageLUT();
  //    cout<<"  layer1HFScalePhiBins=   ["; for(unsigned int i=0; i<layer1SecondStageLUT.size(); i++) cout<<(i==0?"":",")<<layer1SecondStageLUT[i]; cout<<"]"<<endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TCaloParamsViewer);
