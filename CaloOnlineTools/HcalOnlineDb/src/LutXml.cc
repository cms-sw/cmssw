#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "md5.h"

#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

LutXml::Config::_Config() {
  infotype = "LUT";
  ieta = -1000;
  iphi = -1000;
  crate = -1;
  slot = -1;
  topbottom = -1;
  fiber = -1;
  fiberchan = -1;
  lut_type = -1;
  creationtag = "default_tag";
  creationstamp =
      std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
  formatrevision = "default_revision";
  targetfirmware = "default_revision";
  generalizedindex = -1;
  weight = -1;
  // Default to keeping veto disabled
  codedvetothreshold = 0;
}

LutXml::LutXml() : XMLDOMBlock("CFGBrickSet", true) { init(); }

LutXml::LutXml(const std::string& filename) : XMLDOMBlock(filename) { init(); }

LutXml::~LutXml() {
  xercesc::XMLString::release(&root);
  xercesc::XMLString::release(&brick);
}

void LutXml::init(void) {
  root = xercesc::XMLString::transcode("CFGBrickSet");
  brick = xercesc::XMLString::transcode("CFGBrick");
  brickElem = nullptr;
}

// checksums_xml is 0 by default
void LutXml::addLut(LutXml::Config& _config, XMLDOMBlock* checksums_xml) {
  xercesc::DOMElement* rootElem = document->getDocumentElement();

  brickElem = document->createElement(XMLProcessor::_toXMLCh("CFGBrick").get());
  rootElem->appendChild(brickElem);

  addParameter("INFOTYPE", "string", _config.infotype);
  addParameter("CREATIONTAG", "string", _config.creationtag);
  addParameter("CREATIONSTAMP", "string", _config.creationstamp);
  addParameter("FORMATREVISION", "string", _config.formatrevision);
  addParameter("TARGETFIRMWARE", "string", _config.targetfirmware);
  addParameter("GENERALIZEDINDEX", "int", _config.generalizedindex);
  addParameter("CRATE", "int", _config.crate);
  addParameter("SLOT", "int", _config.slot);

  if (checksums_xml) {
    addParameter("CHECKSUM", "string", get_checksum(_config.lut));
  }

  if (_config.lut_type == 1) {  // linearizer LUT
    addParameter("IETA", "int", _config.ieta);
    addParameter("IPHI", "int", _config.iphi);
    addParameter("TOPBOTTOM", "int", _config.topbottom);
    addParameter("LUT_TYPE", "int", _config.lut_type);
    addParameter("FIBER", "int", _config.fiber);
    addParameter("FIBERCHAN", "int", _config.fiberchan);
    addParameter("DEPTH", "int", _config.depth);
    addData(std::to_string(_config.lut.size()), "hex", _config.lut);
  } else if (_config.lut_type == 2 || _config.lut_type == 4) {  // compression LUT or HE feature bit LUT
    addParameter("IETA", "int", _config.ieta);
    addParameter("IPHI", "int", _config.iphi);
    addParameter("TOPBOTTOM", "int", _config.topbottom);
    addParameter("LUT_TYPE", "int", _config.lut_type);
    addParameter("SLB", "int", _config.fiber);
    addParameter("SLBCHAN", "int", _config.fiberchan);
    addParameter("WEIGHT", "int", _config.weight);
    // Special coded veto threshold value of zero disables vetoing in PFA1'
    if (_config.codedvetothreshold > 0) {
      // A valid coded value here is in the range (1, 2048) inclusive
      if (_config.codedvetothreshold <= 2048) {
        // The coded value of 2048 means to do vetoing with no threshold
        int actualvetothreshold = _config.codedvetothreshold == 2048 ? 0 : _config.codedvetothreshold;
        addParameter("PREFIRE_VETO_THRESHOLD", "int", actualvetothreshold);
      } else {
        edm::LogWarning("LutXml") << "Positive veto threshold of " << _config.codedvetothreshold
                                  << " is not in range (1, 2048) ! Vetoing will not be done in PFA1' !";
      }
    }
    addData(std::to_string(_config.lut.size()), "hex", _config.lut);
  } else if (_config.lut_type == 5) {  // channel masks
    addParameter("MASK_TYPE", "string", "TRIGGERCHANMASK");
    addData(std::to_string(_config.mask.size()), "hex", _config.mask);
  } else if (_config.lut_type == 6) {  // adc threshold for tdc mask
    addParameter("THRESH_TYPE", "string", "TRIGINTIME");
    addData(std::to_string(_config.mask.size()), "hex", _config.mask);
  } else if (_config.lut_type == 7) {  // tdc mask
    addParameter("TDCMAP_TYPE", "string", "TRIGINTIME");
    addData(std::to_string(_config.mask.size()), "hex", _config.mask);
  } else {
    edm::LogError("LutXml") << "Unknown LUT type...produced XML will be incorrect";
  }

  if (checksums_xml) {
    add_checksum(checksums_xml->getDocument(), _config);
  }
}

template <typename T>
xercesc::DOMElement* LutXml::addData(const std::string& _elements, const std::string& _encoding, const T& _lut) {
  xercesc::DOMElement* child = document->createElement(XMLProcessor::_toXMLCh("Data").get());
  child->setAttribute(XMLProcessor::_toXMLCh("elements").get(), XMLProcessor::_toXMLCh(_elements).get());
  child->setAttribute(XMLProcessor::_toXMLCh("encoding").get(), XMLProcessor::_toXMLCh(_encoding).get());

  std::ostringstream buf;
  for (const auto& item : _lut) {
    buf << std::hex << static_cast<std::uint64_t>(item) << ' ';
  }

  xercesc::DOMText* data_value = document->createTextNode(XMLProcessor::_toXMLCh(buf.str()).get());
  child->appendChild(data_value);

  brickElem->appendChild(child);

  return child;
}

xercesc::DOMElement* LutXml::add_checksum(xercesc::DOMDocument* parent, const Config& config) {
  xercesc::DOMElement* child = parent->createElement(XMLProcessor::_toXMLCh("Data").get());
  child->setAttribute(XMLProcessor::_toXMLCh("crate").get(), XMLProcessor::_toXMLCh(config.crate).get());
  child->setAttribute(XMLProcessor::_toXMLCh("slot").get(), XMLProcessor::_toXMLCh(config.slot).get());
  child->setAttribute(XMLProcessor::_toXMLCh("fpga").get(), XMLProcessor::_toXMLCh(config.topbottom).get());
  child->setAttribute(XMLProcessor::_toXMLCh("fiber").get(), XMLProcessor::_toXMLCh(config.fiber).get());
  child->setAttribute(XMLProcessor::_toXMLCh("fiberchan").get(), XMLProcessor::_toXMLCh(config.fiberchan).get());
  child->setAttribute(XMLProcessor::_toXMLCh("luttype").get(), XMLProcessor::_toXMLCh(config.lut_type).get());
  child->setAttribute(XMLProcessor::_toXMLCh("elements").get(), XMLProcessor::_toXMLCh("1").get());
  child->setAttribute(XMLProcessor::_toXMLCh("encoding").get(), XMLProcessor::_toXMLCh("hex").get());
  xercesc::DOMText* checksum_value = parent->createTextNode(XMLProcessor::_toXMLCh(get_checksum(config.lut)).get());
  child->appendChild(checksum_value);

  parent->getDocumentElement()->appendChild(child);

  return child;
}

xercesc::DOMElement* LutXml::addParameter(const std::string& _name,
                                          const std::string& _type,
                                          const std::string& _value) {
  xercesc::DOMElement* child = document->createElement(XMLProcessor::_toXMLCh("Parameter").get());
  child->setAttribute(XMLProcessor::_toXMLCh("name").get(), XMLProcessor::_toXMLCh(_name).get());
  child->setAttribute(XMLProcessor::_toXMLCh("type").get(), XMLProcessor::_toXMLCh(_type).get());
  xercesc::DOMText* parameter_value = document->createTextNode(XMLProcessor::_toXMLCh(_value).get());
  child->appendChild(parameter_value);

  brickElem->appendChild(child);

  return child;
}

xercesc::DOMElement* LutXml::addParameter(const std::string& _name, const std::string& _type, int _value) {
  return addParameter(_name, _type, std::to_string(_value));
}

// do MD5 checksum
std::string LutXml::get_checksum(const std::vector<unsigned int>& lut) {
  std::stringstream result;
  md5_state_t md5er;
  md5_byte_t digest[16];
  md5_init(&md5er);
  // linearizer LUT:
  if (lut.size() == 128) {
    unsigned char tool[2];
    for (int i = 0; i < 128; i++) {
      tool[0] = lut[i] & 0xFF;
      tool[1] = (lut[i] >> 8) & 0xFF;
      md5_append(&md5er, tool, 2);
    }
  } else if (lut.size() == 256) {
    unsigned char tool[2];
    for (int i = 0; i < 256; i++) {
      tool[0] = lut[i] & 0xFF;
      tool[1] = (lut[i] >> 8) & 0xFF;
      md5_append(&md5er, tool, 2);
    }
  }
  // compression LUT:
  else if (lut.size() == 1024) {
    unsigned char tool;
    for (int i = 0; i < 1024; i++) {
      tool = lut[i] & 0xFF;
      md5_append(&md5er, &tool, 1);
    }
  } else if (lut.size() == 2048) {
    unsigned char tool;
    for (int i = 0; i < 2048; i++) {
      tool = lut[i] & 0xFF;
      md5_append(&md5er, &tool, 1);
    }
  }
  // HE fine grain LUT
  else if (lut.size() == 4096) {
    unsigned char tool;
    for (int i = 0; i < 4096; i++) {
      tool = lut[i] & 0xFF;
      md5_append(&md5er, &tool, 1);
    }
  } else {
    edm::LogError("LutXml") << "Irregular LUT size, " << lut.size()
                            << " , do not know how to compute checksum, exiting...";
    exit(-1);
  }
  md5_finish(&md5er, digest);
  for (int i = 0; i < 16; i++)
    result << std::hex << (((int)(digest[i])) & 0xFF);

  return result.str();
}

DetId LutXml::detid_from_crate(
    int crate, int slot, int fiber, int fiberch, bool isTrigger, const HcalElectronicsMap* emap) {
  HcalElectronicsId electronicsId = HcalElectronicsId(crate, slot, fiber, fiberch, isTrigger);

  DetId detId = emap->lookup(electronicsId);
  if (detId.null()) {
    edm::LogWarning("LutXml") << "Invalid electronics ID or no mapping found for crate: " << crate << " slot: " << slot
                              << " fiber: " << fiber << " fiberch: " << fiberch << std::endl;
    return 0;
  } else {
    return detId;
  }
}

// organize all LUTs in XML into a map for fast access
int LutXml::create_lut_map(const HcalElectronicsMap* emap) {
  lut_map.clear();

  if (document) {
    xercesc::DOMNodeList* brick_list = document->getDocumentElement()->getElementsByTagName(brick);
    int n_of_bricks = brick_list->getLength();
    for (int i = 0; i != n_of_bricks; i++) {
      xercesc::DOMElement* aBrick = (xercesc::DOMElement*)(brick_list->item(i));
      xercesc::DOMNodeList* par_list = aBrick->getElementsByTagName(xercesc::XMLString::transcode("Parameter"));
      int n_of_par = par_list->getLength();
      int ieta = -99;
      int iphi = -99;
      int crate = -99;
      int slot = -99;
      int fiber = -99;
      int fiberch = -99;
      int slb = -99;
      int lut_type = -99;
      for (int j = 0; j != n_of_par; j++) {
        xercesc::DOMElement* aPar = (xercesc::DOMElement*)(par_list->item(j));
        char* aName = xercesc::XMLString::transcode(aPar->getAttribute(XMLProcessor::_toXMLCh("name").get()));
        std::string paramName = std::string(aName);
        xercesc::XMLString::release(&aName);
        if (paramName == "IETA")
          ieta = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "IPHI")
          iphi = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "CRATE")
          crate = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "SLOT")
          slot = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "FIBERCHAN")
          fiberch = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "FIBER")
          fiber = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "SLB")
          slb = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
        if (paramName == "LUT_TYPE")
          lut_type = xercesc::XMLString::parseInt(aPar->getFirstChild()->getNodeValue());
      }

      xercesc::DOMElement* _data =
          (xercesc::DOMElement*)(aBrick->getElementsByTagName(xercesc::XMLString::transcode("Data"))->item(0));
      char* _str = xercesc::XMLString::transcode(_data->getFirstChild()->getNodeValue());

      // get the LUT vector
      int _string_length = strlen(_str);
      std::vector<unsigned int> _lut;
      unsigned int _base = 16;
      unsigned int _item = 0;
      for (int i = 0; i != _string_length; i++) {
        bool _range = false;
        char ch_cur = _str[i];
        if (_base == 16)
          _range = (ch_cur >= '0' and ch_cur <= '9') || (ch_cur >= 'a' and ch_cur <= 'f') ||
                   (ch_cur >= 'A' and ch_cur <= 'F');
        else if (_base == 10)
          _range = (ch_cur >= '0' and ch_cur <= '9');
        if (_range) {
          if (ch_cur >= 'a' and ch_cur <= 'f')
            ch_cur += 10 - 'a';
          else if (ch_cur >= 'A' and ch_cur <= 'F')
            ch_cur += 10 - 'A';
          else if (ch_cur >= '0' and ch_cur <= '9')
            ch_cur += -'0';
          _item = _item * _base;
          _item += ch_cur;
          bool last_digit = false;
          if ((i + 1) == _string_length)
            last_digit = true;
          else {
            char ch_next = _str[i + 1];
            bool _range_next = false;
            if (_base == 16)
              _range_next = (ch_next >= '0' and ch_next <= '9') || (ch_next >= 'a' and ch_next <= 'f') ||
                            (ch_next >= 'A' and ch_next <= 'F');
            else if (_base == 10)
              _range_next = (ch_next >= '0' and ch_next <= '9');
            if (!_range_next)
              last_digit = true;
          }
          if (last_digit) {
            _lut.push_back(_item);
            _item = 0;
          }
        }
      }

      // filling the map
      uint32_t _key = 0;
      if (lut_type == 1) {
        DetId detId = detid_from_crate(crate, slot, fiber, fiberch, false, emap);
        if (detId.det() == DetId::Hcal) {
          HcalDetId _id(detId);
          _key = _id.rawId();
        } else if (detId.det() == DetId::Calo && detId.subdetId() == HcalZDCDetId::SubdetectorId) {
          HcalZDCDetId _id(detId);
          _key = _id.rawId();
        }
      } else if (lut_type == 2) {
        int version = (abs(ieta) > 29 && slb != 12 && crate > 20) ? 1 : 0;
        HcalTrigTowerDetId _id(ieta, iphi, 10 * version);
        _key = _id.rawId();
      } else
        continue;
      lut_map.insert(std::pair<uint32_t, std::vector<unsigned int> >(_key, _lut));
    }
  } else {
    edm::LogError("LutXml") << "XML file with LUTs is not loaded, cannot create map!";
  }

  return 0;
}

LutXml::const_iterator LutXml::begin() const { return lut_map.begin(); }

LutXml::const_iterator LutXml::end() const { return lut_map.end(); }

LutXml::const_iterator LutXml::find(uint32_t id) const { return lut_map.find(id); }
