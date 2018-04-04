#ifndef SIPIXELMODULESTATUS_h
#define SIPIXELMODULESTATUS_h

#include "CalibTracker/SiPixelQuality/interface/SiPixelRocStatus.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

#include <vector>


// ----------------------------------------------------------------------
class SiPixelModuleStatus {

public:

  SiPixelModuleStatus(int det = 0, int nrocs = 16);
  ~SiPixelModuleStatus();

  /// fill with online coordinates
  void fillDIGI(int iroc);

  /// fill with online coordinates (nhit > 1)
  void updateDIGI(int iroc, unsigned int nhit);

  /// fill FEDerror25
  void fillFEDerror25( PixelFEDChannel ch );

  /// return ROC status (= hits on ROC iroc)
  unsigned int digiOccROC(int iroc);

  /// return module status (= hits on module)
  unsigned int digiOccMOD();

  /// get a ROC
  SiPixelRocStatus* getRoc(int i);

  /// accessors and setters
  int    detid();
  int    nrocs();
  void   setNrocs(int iroc);

  /// calculate (averaged over this module's ROCs) mean hit number and its sigma
  double perRocDigiOcc();
  double perRocDigiOccVar();

  /// combine new data to update(topup) module status
  void updateModuleDIGI(int roc, unsigned int nhits);
  void updateModuleStatus(SiPixelModuleStatus newData);

private:

  int fDetid, fNrocs;
  std::vector<SiPixelRocStatus> fRocs;

};

#endif
