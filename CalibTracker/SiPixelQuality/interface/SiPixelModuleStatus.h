#ifndef SIPIXELMODULESTATUS_h
#define SIPIXELMODULESTATUS_h

#include "CalibTracker/SiPixelQuality/interface/SiPixelRocStatus.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

#include <vector>

// ----------------------------------------------------------------------
class SiPixelModuleStatus {
public:
  SiPixelModuleStatus(int det = 0, int nrocs = 16);  // default for Phase-1
  ~SiPixelModuleStatus();

  /// fill digi
  void fillDIGI(int iroc);
  /// fill FEDerror25
  void fillFEDerror25(PixelFEDChannel ch);

  /// update digi (nhit > 1)
  void updateDIGI(int iroc, unsigned int nhit);
  /// update FEDerror25
  void updateFEDerror25(int iroc, bool FEDerror25);

  /// return ROC status (= hits on ROC iroc)
  unsigned int digiOccROC(int iroc);

  /// return ROC FEDerror25
  bool fedError25(int iroc);

  /// return module status (= hits on module)
  unsigned int digiOccMOD();

  /// get a ROC
  SiPixelRocStatus* getRoc(int i);

  /// accessors and setters
  int detid();
  int nrocs();
  void setDetId(int detid);
  void setNrocs(int iroc);

  /// calculate (averaged over this module's ROCs) mean hit number and its sigma
  double perRocDigiOcc();
  double perRocDigiOccVar();

  /// combine new data to update(topup) module status
  void updateModuleDIGI(int roc, unsigned int nhits);
  void updateModuleStatus(SiPixelModuleStatus newData);

private:
  int fDetid_, fNrocs_;
  std::vector<SiPixelRocStatus> fRocs_;
};

#endif
