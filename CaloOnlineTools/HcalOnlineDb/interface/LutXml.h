#ifndef CaloOnlineTools_HcalOnlineDb_LutXml_h
#define CaloOnlineTools_HcalOnlineDb_LutXml_h

// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     LutXml
//
/**\class LutXml LutXml.h CaloOnlineTools/HcalOnlineDb/interface/LutXml.h

 Description: Defines payload for HCAL LUT XML bricks

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Mar 18 14:30:33 CDT 2008
//

#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <xercesc/dom/DOM.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

class LutXml : public XMLDOMBlock {
public:
  typedef struct _Config {
    _Config();
    std::string infotype;
    int ieta, iphi, depth, crate, slot, topbottom, fiber, fiberchan, lut_type;
    std::string creationtag;
    std::string creationstamp;
    std::string formatrevision;
    std::string targetfirmware;
    int generalizedindex;
    int weight;
    int codedvetothreshold;
    std::vector<unsigned int> lut;
    std::vector<uint64_t> mask;
  } Config;

  LutXml();
  LutXml(const std::string& filename);
  ~LutXml() override;

  void init(void);
  void addLut(Config& _config, XMLDOMBlock* checksums_xml = nullptr);

  DetId detid_from_crate(int crate, int slot, int fiber, int fiberch, bool isTrigger, const HcalElectronicsMap* emap);
  int create_lut_map(const HcalElectronicsMap* emap);

  static std::string get_checksum(const std::vector<unsigned int>& lut);

  typedef std::map<uint32_t, std::vector<unsigned int> >::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator find(uint32_t) const;

protected:
  XMLCh* root;
  XMLCh* brick;
  xercesc::DOMElement* addParameter(const std::string& _name, const std::string& _type, const std::string& _value);
  xercesc::DOMElement* addParameter(const std::string& _name, const std::string& _type, int _value);

  template <typename T>
  xercesc::DOMElement* addData(const std::string& _elements, const std::string& _encoding, const T& _lut);
  xercesc::DOMElement* add_checksum(xercesc::DOMDocument* parent, const Config& config);
  xercesc::DOMElement* brickElem;
  std::map<uint32_t, std::vector<unsigned int> > lut_map;
};

#endif
