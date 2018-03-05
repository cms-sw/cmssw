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
  void fillDIGI(int iroc, int idc);

  /// fill with online coordinates (nhit > 1)
  void updateDIGI(int iroc, int idc, unsigned long nhit);

  /// fill stuck TBM
  void fillStuckTBM( PixelFEDChannel ch, std::time_t time );

  /// return DC status of a ROC (=hits on DC idc on ROC iroc)
  unsigned long int digiOccDC(int iroc, int idc);

  /// return ROC status (= hits on ROC iroc)
  unsigned long int digiOccROC(int iroc);

  /// return module status (= hits on module)
  unsigned long int digiOccMOD();

  /// get a ROC
  SiPixelRocStatus* getRoc(int i);

  /// accessors and setters
  int    detid();
  int    nrocs();
  void   setNrocs(int iroc);

  /// calculate (averaged over this module's ROCs) mean hit number and its sigma
  void digiOccupancy();
  double perRocDigiOcc();
  double perRocDigiOccVar();

  /// combine new data to update(topup) module status
  void updateModuleDIGI(int roc, int dc, unsigned long int nhits);
  void updateModuleStatus(SiPixelModuleStatus newData);

private:

  int fDetid, fNrocs;
  double fModAverage, fModSigma;
  std::vector<SiPixelRocStatus> fRocs;

};

#endif
