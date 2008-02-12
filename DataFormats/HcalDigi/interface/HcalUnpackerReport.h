#ifndef DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H
#define DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H 1

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

/** \class HcalUnpackerReport
  *  
  * $Date: 2006/10/09 14:43:25 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalUnpackerReport {
public:
  HcalUnpackerReport();
  const std::vector<int>& getFedsUnpacked() const { return FEDsUnpacked_; }
  const std::vector<int>& getFedsError() const { return FEDsError_; }
  bool errorFree() const;
  bool anyValidHCAL() const;
  int unmappedDigis() const { return unmappedDigis_; }
  int unmappedTPDigis() const { return unmappedTPDigis_; }
  int spigotFormatErrors() const { return spigotFormatErrors_; }
  int badQualityDigis() const { return badqualityDigis_; }
  int totalDigis() const { return totalDigis_; }
  int totalTPDigis() const { return totalTPDigis_; }
  int totalHOTPDigis() const { return totalHOTPDigis_; }


  typedef std::vector<DetId> DetIdVector;
  typedef std::vector<HcalElectronicsId> ElectronicsIdVector;

  DetIdVector::const_iterator bad_quality_begin() const { return badqualityIds_.begin(); }
  DetIdVector::const_iterator bad_quality_end() const { return badqualityIds_.end(); }
  ElectronicsIdVector::const_iterator unmapped_begin() const { return unmappedIds_.begin(); }
  ElectronicsIdVector::const_iterator unmapped_end() const { return unmappedIds_.end(); }

  // setters
  void addUnpacked(int fed);
  void addError(int fed);
  void countDigi();
  void countTPDigi(bool ho=false);
  void countUnmappedDigi();
  void countUnmappedTPDigi();
  void countSpigotFormatError();
  void countBadQualityDigi();
  void countUnmappedDigi(const HcalElectronicsId& eid);
  void countUnmappedTPDigi(const HcalElectronicsId& eid);
  void countBadQualityDigi(const DetId& did);
private:
  std::vector<int> FEDsUnpacked_;
  std::vector<int> FEDsError_;
  int unmappedDigis_, unmappedTPDigis_;
  int spigotFormatErrors_, badqualityDigis_;
  int totalDigis_, totalTPDigis_, totalHOTPDigis_;
  DetIdVector badqualityIds_;
  ElectronicsIdVector unmappedIds_;
};

#endif
