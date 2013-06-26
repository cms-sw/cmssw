#ifndef DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H
#define DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H 1

#include <vector>
#include <map>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"

/** \class HcalUnpackerReport
  *  
  * $Date: 2012/06/04 10:47:17 $
  * $Revision: 1.6 $
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
  int emptyEventSpigots() const { return emptyEventSpigots_; }
  int OFWSpigots() const { return ofwSpigots_; }
  int busySpigots() const { return busySpigots_; }

  bool unsuppressedChannels() const { return unsuppressed_; }

  bool hasFedWithCalib() const { return !fedInfo_.empty(); }
  HcalCalibrationEventType fedCalibType(uint16_t fed) const;

  void setFedCalibInfo(uint16_t fed, HcalCalibrationEventType ctype);

  typedef std::vector<DetId> DetIdVector;
  typedef std::vector<HcalElectronicsId> ElectronicsIdVector;

  DetIdVector::const_iterator bad_quality_begin() const { return badqualityIds_.begin(); }
  DetIdVector::const_iterator bad_quality_end() const { return badqualityIds_.end(); }
  ElectronicsIdVector::const_iterator unmapped_begin() const { return unmappedIds_.begin(); }
  ElectronicsIdVector::const_iterator unmapped_end() const { return unmappedIds_.end(); }

  bool hasReportInfo(const std::string& name) const;
  std::string getReportInfo(const std::string& name) const;
  std::vector<std::string> getReportKeys() const;
  
  // setters
  void addUnpacked(int fed);
  void addError(int fed);
  void countDigi();
  void countTPDigi(bool ho=false);
  void countUnmappedDigi();
  void countUnmappedTPDigi();
  void countSpigotFormatError();
  void countBadQualityDigi();
  void countEmptyEventSpigot();
  void countOFWSpigot();
  void countBusySpigot();
  void countUnmappedDigi(const HcalElectronicsId& eid);
  void countUnmappedTPDigi(const HcalElectronicsId& eid);
  void countBadQualityDigi(const DetId& did);
  void setUnsuppressed(bool isSup);
  void setReportInfo(const std::string& name, const std::string& value);
private:
  std::vector<int> FEDsUnpacked_;
  std::vector<int> FEDsError_;
  int unmappedDigis_, unmappedTPDigis_;
  int spigotFormatErrors_, badqualityDigis_;
  int totalDigis_, totalTPDigis_, totalHOTPDigis_;
  DetIdVector badqualityIds_;
  ElectronicsIdVector unmappedIds_;
  bool unsuppressed_;

  std::vector<std::string> reportInfo_;
  std::vector<uint16_t> fedInfo_; // first is fed, second is type

  int emptyEventSpigots_,ofwSpigots_,busySpigots_;
};

#endif
