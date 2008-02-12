#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

HcalUnpackerReport::HcalUnpackerReport() :
  unmappedDigis_(0), unmappedTPDigis_(0),
  spigotFormatErrors_(0), badqualityDigis_(0),
  totalDigis_(0),totalTPDigis_(0),totalHOTPDigis_(0)
{
}

bool HcalUnpackerReport::errorFree() const {
  return FEDsError_.empty() && spigotFormatErrors_==0;
}

bool HcalUnpackerReport::anyValidHCAL() const {
  return !FEDsUnpacked_.empty();
}

void HcalUnpackerReport::addUnpacked(int fed) {
  FEDsUnpacked_.push_back(fed);
}

void HcalUnpackerReport::addError(int fed) {
  FEDsError_.push_back(fed);
}

void HcalUnpackerReport::countDigi() {
  totalDigis_++;
}
void HcalUnpackerReport::countTPDigi(bool ho) {
  if (ho) totalHOTPDigis_++;
  else totalTPDigis_++;
}

void HcalUnpackerReport::countUnmappedDigi() {
  unmappedDigis_++;
}
void HcalUnpackerReport::countUnmappedTPDigi() {
  unmappedTPDigis_++;
}
void HcalUnpackerReport::countSpigotFormatError() {
  spigotFormatErrors_++;
}
void HcalUnpackerReport::countBadQualityDigi() {
  badqualityDigis_++;
}
void HcalUnpackerReport::countBadQualityDigi(const DetId& id) {
  badqualityDigis_++;
  badqualityIds_.push_back(id);
}
void HcalUnpackerReport::countUnmappedDigi(const HcalElectronicsId& eid) {
  unmappedDigis_++;
  unmappedIds_.push_back(eid);
}
void HcalUnpackerReport::countUnmappedTPDigi(const HcalElectronicsId& eid) {
  unmappedTPDigis_++;
  unmappedIds_.push_back(eid);
}
