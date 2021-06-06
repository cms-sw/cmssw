#ifndef EventFilter_EcalRawToDigi_interface_EcalRegionCabling_h
#define EventFilter_EcalRawToDigi_interface_EcalRegionCabling_h

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/EcalMapping/interface/ESElectronicsMapper.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

class EcalRegionCabling {
public:
  EcalRegionCabling(edm::ParameterSet& conf, const EcalElectronicsMapping* m) : mapping_(m) {
    const edm::ParameterSet esMap = conf.getParameter<edm::ParameterSet>("esMapping");
    es_mapping_ = new ESElectronicsMapper(esMap);
  }

  ~EcalRegionCabling() {
    // this pointer is own by this object.
    delete es_mapping_;
  }
  const EcalElectronicsMapping* mapping() const { return mapping_; }
  const ESElectronicsMapper* es_mapping() const { return es_mapping_; }

  static uint32_t maxElementIndex() { return (FEDNumbering::MAXECALFEDID - FEDNumbering::MINECALFEDID + 1); }
  static uint32_t maxESElementIndex() {
    return (FEDNumbering::MAXPreShowerFEDID - FEDNumbering::MINPreShowerFEDID + 1);
  }

  static uint32_t elementIndex(const int FEDindex) {
    //do a test for the time being
    if (FEDindex > FEDNumbering::MAXECALFEDID || FEDindex < FEDNumbering::MINECALFEDID) {
      edm::LogError("IncorrectMapping") << "FEDindex: " << FEDindex
                                        << " is not between: " << (int)FEDNumbering::MINECALFEDID << " and "
                                        << (int)FEDNumbering::MAXECALFEDID;
      return 0;
    }
    uint32_t eI = FEDindex - FEDNumbering::MINECALFEDID;
    return eI;
  }

  static uint32_t esElementIndex(const int FEDindex) {
    //do a test for the time being
    if (FEDindex > FEDNumbering::MAXPreShowerFEDID || FEDindex < FEDNumbering::MINPreShowerFEDID) {
      edm::LogError("IncorrectMapping") << "FEDindex: " << FEDindex
                                        << " is not between: " << (int)FEDNumbering::MINPreShowerFEDID << " and "
                                        << (int)FEDNumbering::MAXPreShowerFEDID;
      return 0;
    }
    uint32_t eI = FEDindex - FEDNumbering::MINPreShowerFEDID;
    return eI;
  }

  static int fedIndex(const uint32_t index) {
    int fI = index + FEDNumbering::MINECALFEDID;
    return fI;
  }

  static int esFedIndex(const uint32_t index) {
    int fI = index + FEDNumbering::MINPreShowerFEDID;
    return fI;
  }

  uint32_t elementIndex(const double eta, const double phi) const {
    int FEDindex = mapping()->GetFED(eta, phi);
    return elementIndex(FEDindex);
  }

private:
  const EcalElectronicsMapping* mapping_;
  const ESElectronicsMapper* es_mapping_;
};

#endif  // EventFilter_EcalRawToDigi_interface_EcalRegionCabling_h
