#ifndef CaloOnlineTools_HcalOnlineDb_HcalLutManager_h
#define CaloOnlineTools_HcalOnlineDb_HcalLutManager_h

#include "CalibCalorimetry/CaloTPG/interface/CaloTPGTranscoderULUT.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLDOMBlock.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct HcalLutSet {
  std::string label;
  std::vector<std::string> subdet;
  std::vector<int> eta_min, eta_max, phi_min, phi_max, depth_min, depth_max;
  std::vector<std::vector<unsigned int> > lut;
};

class HcalLutManager {
public:
  HcalLutManager(const HcalDbService* conditions,
                 const HcalChannelQuality* _cq = nullptr,
                 uint32_t _status_word_to_mask = 0x8000);

  ~HcalLutManager();

  std::map<int, std::shared_ptr<LutXml> > getLinearizationLutXmlFromAsciiMasterEmap(const std::string& _filename,
                                                                                    const std::string& _tag,
                                                                                    int _crate);

  std::map<int, std::shared_ptr<LutXml> > getCompressionLutXmlFromAsciiMaster(const std::string& _filename,
                                                                              const std::string& _tag,
                                                                              int _crate = -1);

  std::map<int, std::shared_ptr<LutXml> > getMasks(int var, const std::string& _tag);

  std::map<int, std::shared_ptr<LutXml> > getLinearizationLutXmlFromCoderEmap(const HcalTPGCoder& _coder,
                                                                              const std::string& _tag);

  std::map<int, std::shared_ptr<LutXml> > getCompressionLutXmlFromCoder(const CaloTPGTranscoderULUT& _coder,
                                                                        const std::string& _tag);

  std::map<int, std::shared_ptr<LutXml> > getZdcLutXml(const HcalTPGCoder& _coder,
                                                       const std::string& _tag,
                                                       bool ootpu_lut = false);

  std::map<int, std::shared_ptr<LutXml> > getHEFineGrainLUTs(const std::string& _tag);

  // add two std::map<s with LUTs. Designed mainly for joining compression LUTs to linearization ones.
  void addLutMap(std::map<int, std::shared_ptr<LutXml> >& result, const std::map<int, std::shared_ptr<LutXml> >& other);

  // read LUTs from ASCII master file.
  // _type = 1 - linearization, 2 - compression
  HcalLutSet getLutSetFromFile(const std::string& _filename, int _type = 1);

  int writeLutXmlFiles(std::map<int, std::shared_ptr<LutXml> >& _xml, const std::string& _tag = "default_tag");

  int createLutXmlFiles_HBEFFromCoder_HOFromAscii_ZDC(const std::string& _tag,
                                                      const HcalTPGCoder& _coder,
                                                      const CaloTPGTranscoderULUT& _transcoder,
                                                      const std::string& _lin_file);

  static int getInt(const std::string& number);
  static std::string get_time_stamp();
  static std::vector<std::string> splitString(const std::string& fLine);

protected:
  XMLDOMBlock* lut_checksums_xml;
  const HcalElectronicsMap* emap;
  const HcalChannelQuality* cq;
  const HcalDbService* conditions;
  uint32_t status_word_to_mask;
};

#endif
