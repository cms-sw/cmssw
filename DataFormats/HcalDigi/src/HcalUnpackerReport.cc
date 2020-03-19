#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

HcalUnpackerReport::HcalUnpackerReport()
    : unmappedDigis_(0),
      unmappedTPDigis_(0),
      spigotFormatErrors_(0),
      badqualityDigis_(0),
      totalDigis_(0),
      totalTPDigis_(0),
      totalHOTPDigis_(0),
      unsuppressed_(false),
      emptyEventSpigots_(0),
      ofwSpigots_(0),
      busySpigots_(0) {}

bool HcalUnpackerReport::errorFree() const { return FEDsError_.empty() && spigotFormatErrors_ == 0; }

bool HcalUnpackerReport::anyValidHCAL() const { return !FEDsUnpacked_.empty(); }

void HcalUnpackerReport::addUnpacked(int fed) { FEDsUnpacked_.push_back(fed); }

void HcalUnpackerReport::addError(int fed) { FEDsError_.push_back(fed); }

void HcalUnpackerReport::countDigi() { totalDigis_++; }
void HcalUnpackerReport::countTPDigi(bool ho) {
  if (ho)
    totalHOTPDigis_++;
  else
    totalTPDigis_++;
}

void HcalUnpackerReport::countUnmappedDigi() { unmappedDigis_++; }
void HcalUnpackerReport::countUnmappedTPDigi() { unmappedTPDigis_++; }
void HcalUnpackerReport::countSpigotFormatError() { spigotFormatErrors_++; }
void HcalUnpackerReport::countEmptyEventSpigot() { emptyEventSpigots_++; }
void HcalUnpackerReport::countOFWSpigot() { ofwSpigots_++; }
void HcalUnpackerReport::countBusySpigot() { busySpigots_++; }

void HcalUnpackerReport::countBadQualityDigi() { badqualityDigis_++; }
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

HcalCalibrationEventType HcalUnpackerReport::fedCalibType(uint16_t fed) const {
  std::vector<uint16_t>::size_type i;
  uint16_t retval = 0;
  for (i = 0; i < fedInfo_.size(); i += 2)
    if (fedInfo_[i] == fed) {
      retval = fedInfo_[i + 1];
      break;
    }
  return HcalCalibrationEventType(retval);
}

void HcalUnpackerReport::setFedCalibInfo(uint16_t fed, HcalCalibrationEventType ctype) {
  std::vector<uint16_t>::size_type i;
  for (i = 0; i < fedInfo_.size(); i += 2)
    if (fedInfo_[i] == fed) {
      fedInfo_[i + 1] = uint16_t(ctype);
      break;
    }
  if (i >= fedInfo_.size()) {
    fedInfo_.push_back(fed);
    fedInfo_.push_back(uint16_t(ctype));
  }
}

void HcalUnpackerReport::setUnsuppressed(bool isSup) { unsuppressed_ = isSup; }

static const std::string ReportSeparator("==>");

void HcalUnpackerReport::setReportInfo(const std::string& name, const std::string& value) {
  reportInfo_.push_back(name + "==>" + value);
}

bool HcalUnpackerReport::hasReportInfo(const std::string& name) const {
  std::string searchFor = name + ReportSeparator;
  std::vector<std::string>::const_iterator i;
  for (i = reportInfo_.begin(); i != reportInfo_.end(); i++)
    if (i->find(searchFor) == 0)
      break;
  return (i != reportInfo_.end());
}
std::string HcalUnpackerReport::getReportInfo(const std::string& name) const {
  std::string searchFor = name + ReportSeparator;
  std::vector<std::string>::const_iterator i;
  for (i = reportInfo_.begin(); i != reportInfo_.end(); i++)
    if (i->find(searchFor) == 0)
      break;
  std::string retval;
  if (i != reportInfo_.end()) {
    retval = i->substr(searchFor.length());
  }
  return retval;
}

std::vector<std::string> HcalUnpackerReport::getReportKeys() const {
  std::vector<std::string> retval;
  std::vector<std::string>::const_iterator i;
  for (i = reportInfo_.begin(); i != reportInfo_.end(); i++)
    retval.push_back(i->substr(0, i->find(ReportSeparator)));
  return retval;
}
