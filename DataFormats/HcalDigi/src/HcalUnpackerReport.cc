#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"

HcalUnpackerReport::HcalUnpackerReport() :
  unmappedDigis_(0), unmappedTPDigis_(0),
  spigotFormatErrors_(0), badqualityDigis_(0),
  totalDigis_(0),totalTPDigis_(0),totalHOTPDigis_(0), unsuppressed_(false)
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

HcalCalibrationEventType HcalUnpackerReport::fedCalibType(uint16_t fed) const {
  std::vector<FedCalibInfo>::const_iterator i;
  uint16_t retval=0;
  for (i=fedInfo_.begin(); i!=fedInfo_.end(); i++)
    if (i->fed_==fed) {
      retval=i->type_;
      break;
    }
  return HcalCalibrationEventType(retval);
}

void HcalUnpackerReport::setFedCalibInfo(uint16_t fed, HcalCalibrationEventType ctype) {
  std::vector<FedCalibInfo>::iterator i;
  for (i=fedInfo_.begin(); i!=fedInfo_.end(); i++)
    if (i->fed_==fed) {
      i->type_=uint16_t(ctype);
      break;
    }
  if (i==fedInfo_.end()) {
    fedInfo_.push_back(FedCalibInfo());
    fedInfo_.back().fed_=fed;
    fedInfo_.back().type_=uint16_t(ctype);
  }
}

void HcalUnpackerReport::setUnsuppressed(bool isSup) {
  unsuppressed_=isSup;
}
void HcalUnpackerReport::setReportInfo(const std::string& name, const std::string& value) {
  reportInfo_[name]=value;
}

bool HcalUnpackerReport::hasReportInfo(const std::string& name) const {
  std::map<std::string,std::string>::const_iterator i=reportInfo_.find(name);
  return (i!=reportInfo_.end());
}
std::string HcalUnpackerReport::getReportInfo(const std::string& name) const {
  std::string retval;
  std::map<std::string,std::string>::const_iterator i=reportInfo_.find(name);
  if (i!=reportInfo_.end()) retval=i->second;
  return retval;
}
