#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

HcalUnpackerReport::HcalUnpackerReport() :
  unmappedDigis_(0), unmappedTPDigis_(0),
  spigotFormatErrors_(0), badqualityDigis_(0)
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
