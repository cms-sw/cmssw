#ifndef DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H
#define DATAFORMATS_HCALDIGI_HCALUNPACKERREPORT_H 1

#include <vector>

/** \class HcalUnpackerReport
  *  
  * $Date: $
  * $Revision: $
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

  // setters
  void addUnpacked(int fed);
  void addError(int fed);
  void countUnmappedDigi();
  void countUnmappedTPDigi();
  void countSpigotFormatError();
  void countBadQualityDigi();
private:
  std::vector<int> FEDsUnpacked_;
  std::vector<int> FEDsError_;
  int unmappedDigis_, unmappedTPDigis_;
  int spigotFormatErrors_, badqualityDigis_;
};

#endif
