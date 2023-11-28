/*  =====================================================================================
 *
 *       Filename:  CSCDQM_Collection.cc
 *
 *    Description:  Histogram booking code
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 *  =====================================================================================
 */

#include "CSCDQM_Collection.h"
#include "Utilities/Xerces/interface/Xerces.h"
#include <cstdio>
#include <string>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/TransService.hpp>

namespace cscdqm {

  /**
   * @brief  Constructor
   * @param  p_config Pointer to Global configuration object
   */
  Collection::Collection(Configuration* const p_config) { config = p_config; }

  /**
   * @brief  Load XML file and fill definition map(s)
   * @return 
   */
  void Collection::load() {
    LOG_INFO << "Reading histograms from " << config->getBOOKING_XML_FILE();

    if (config->getBOOKING_XML_FILE().empty()) {
      return;
    }

    try {
      cms::concurrency::xercesInitialize();
      {
        XercesDOMParser parser;

        parser.setValidationScheme(XercesDOMParser::Val_Always);
        parser.setDoNamespaces(true);
        parser.setDoSchema(true);
        parser.setExitOnFirstFatalError(true);
        parser.setValidationConstraintFatal(true);
        XMLFileErrorHandler eh;
        parser.setErrorHandler(&eh);
        parser.parse(config->getBOOKING_XML_FILE().c_str());

        DOMDocument* doc = parser.getDocument();
        DOMElement* docNode = doc->getDocumentElement();
        DOMNodeList* itemList = docNode->getChildNodes();

        CoHisto definitions;
        for (XMLSize_t i = 0; i < itemList->getLength(); i++) {
          DOMNode* node = itemList->item(i);
          if (node->getNodeType() != DOMNode::ELEMENT_NODE) {
            continue;
          }

          std::string nodeName = XMLString::transcode(node->getNodeName());

          ///
          /// Load histogram definition
          ///
          if (nodeName == XML_BOOK_DEFINITION) {
            CoHistoProps dp;
            getNodeProperties(node, dp);

            DOMElement* el = dynamic_cast<DOMElement*>(node);
            std::string id(XMLString::transcode(el->getAttribute(XMLString::transcode(XML_BOOK_DEFINITION_ID))));
            definitions.insert(make_pair(id, dp));

          } else

            ///
            /// Load histogram
            ///
            if (nodeName == XML_BOOK_HISTOGRAM) {
              CoHistoProps hp;

              DOMElement* el = dynamic_cast<DOMElement*>(node);
              if (el->hasAttribute(XMLString::transcode(XML_BOOK_DEFINITION_REF))) {
                std::string id(XMLString::transcode(el->getAttribute(XMLString::transcode(XML_BOOK_DEFINITION_REF))));

                CoHistoProps d = definitions[id];
                for (CoHistoProps::iterator it = d.begin(); it != d.end(); it++) {
                  hp[it->first] = it->second;
                }
              }

              getNodeProperties(node, hp);

              std::string name = hp[XML_BOOK_HISTO_NAME];
              std::string prefix = hp[XML_BOOK_HISTO_PREFIX];

              // Check if this histogram is an ON DEMAND histogram?
              hp[XML_BOOK_ONDEMAND] =
                  (Utility::regexMatch(REGEXP_ONDEMAND, name) ? XML_BOOK_ONDEMAND_TRUE : XML_BOOK_ONDEMAND_FALSE);

              LOG_DEBUG << "[Collection::load] loading " << prefix << "::" << name
                        << " XML_BOOK_ONDEMAND = " << hp[XML_BOOK_ONDEMAND];

              CoHistoMap::iterator it = collection.find(prefix);
              if (it == collection.end()) {
                CoHisto h;
                h[name] = hp;
                collection[prefix] = h;
              } else {
                it->second.insert(make_pair(name, hp));
              }
            }
        }
      }

      cms::concurrency::xercesTerminate();

    } catch (XMLException& e) {
      char* message = XMLString::transcode(e.getMessage());
      throw Exception(message);
    }

    for (CoHistoMap::const_iterator i = collection.begin(); i != collection.end(); i++) {
      LOG_INFO << i->second.size() << " " << i->first << " histograms defined";
    }
  }

  /**
   * @brief  Extract and write single histogram properties from XML node to
   * map.
   * @param  node XML node
   * @param  p List of properties to fill
   * @return 
   */

  void Collection::getNodeProperties(DOMNode*& node, CoHistoProps& p) {
    DOMNodeList* props = node->getChildNodes();

    for (XMLSize_t j = 0; j < props->getLength(); j++) {
      DOMNode* node = props->item(j);
      if (node->getNodeType() != DOMNode::ELEMENT_NODE) {
        continue;
      }
      DOMElement* element = dynamic_cast<DOMElement*>(node);
      std::string name = XMLString::transcode(element->getNodeName());

      const XMLCh* content = element->getTextContent();
      XERCES_CPP_NAMESPACE_QUALIFIER TranscodeToStr tc(content, "UTF-8");
      std::istringstream buffer((const char*)tc.str());
      std::string value = buffer.str();

      DOMNamedNodeMap* attributes = node->getAttributes();
      if (attributes) {
        for (XMLSize_t i = 0; i < attributes->getLength(); i++) {
          DOMNode* attribute = attributes->item(i);
          std::string aname = XMLString::transcode(attribute->getNodeName());
          std::string avalue = XMLString::transcode(attribute->getNodeValue());
          p[name + "_" + aname] = avalue;
        }
      }
      p[name] = value;
    }
  }

  /**
   * @brief  Find string histogram value in map
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @return true if parameter found and filled, false - otherwise
   */
  const bool Collection::checkHistoValue(const CoHistoProps& h, const std::string& name, std::string& value) {
    CoHistoProps::const_iterator i = h.find(name);
    if (i == h.end()) {
      return false;
    }
    value = i->second;
    return true;
  }

  /**
   * @brief  get Histogram int value out of the map and return boolean result
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @return true if parameter found and filled, false - otherwise
   */
  const bool Collection::checkHistoValue(const CoHistoProps& h, const std::string& name, int& value) {
    CoHistoProps::const_iterator i = h.find(name);
    if (i == h.end()) {
      return false;
    }
    if (EOF == std::sscanf(i->second.c_str(), "%d", &value)) {
      return false;
    }
    return true;
  }

  /**
   * @brief  get Histogram double value out of the map and return boolean
   * result
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @return true if parameter found and filled, false - otherwise
   */
  const bool Collection::checkHistoValue(const CoHistoProps& h, const std::string name, double& value) {
    CoHistoProps::const_iterator i = h.find(name);
    if (i == h.end()) {
      return false;
    }
    if (EOF == std::sscanf(i->second.c_str(), "%lf", &value)) {
      return false;
    }
    return true;
  }

  /**
   * @brief  Find string histogram value in map
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @param  def_value default value if parameter not found 
   * @return pointer to value
   */
  std::string& Collection::getHistoValue(const CoHistoProps& h,
                                         const std::string& name,
                                         std::string& value,
                                         const std::string& def_value) {
    if (!checkHistoValue(h, name, value)) {
      value = def_value;
    }
    return value;
  }

  /**
   * @brief  get Histogram int value out of the map and 
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @param  def_value default value if parameter not found 
   * @return pointer to value
   */
  int& Collection::getHistoValue(const CoHistoProps& h, const std::string& name, int& value, const int& def_value) {
    if (!checkHistoValue(h, name, value)) {
      value = def_value;
    }
    return value;
  }

  /**
   * @brief  get Histogram double value out of the map and 
   * @param  h Histogram map
   * @param  name parameter name
   * @param  value handler for parameter value
   * @param  def_value default value if parameter not found 
   * @return pointer to value
   */
  double& Collection::getHistoValue(const CoHistoProps& h, const std::string name, double& value, const int def_value) {
    if (!checkHistoValue(h, name, value)) {
      value = def_value;
    }
    return value;
  }

  /**
   * @brief  Parse Axis label string and return values in vector
   * @param  s source string to parse
   * @param  labels pointer to result vector
   * @return number of labels found
   */
  const int Collection::ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels) {
    std::string tmp = s;
    std::string::size_type pos = tmp.find('|');
    char* stopstring = nullptr;

    while (pos != std::string::npos) {
      std::string label_pair = tmp.substr(0, pos);
      tmp.replace(0, pos + 1, "");
      if (label_pair.find('=') != std::string::npos) {
        int nbin = strtol(label_pair.substr(0, label_pair.find('=')).c_str(), &stopstring, 10);
        std::string label = label_pair.substr(label_pair.find('=') + 1, label_pair.length());
        while (label.find('\'') != std::string::npos) {
          label.erase(label.find('\''), 1);
        }
        labels[nbin] = label;
      }
      pos = tmp.find('|');
    }
    return labels.size();
  }

  /**
   * @brief  Book EMU histograms
   * @return 
   */
  void Collection::bookEMUHistos() const {
    CoHistoMap::const_iterator i = collection.find("EMU");
    if (i != collection.end()) {
      const CoHisto hs = i->second;
      for (CoHisto::const_iterator j = hs.begin(); j != hs.end(); j++) {
        std::string s = "";
        if (getHistoValue(j->second, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_FALSE) {
          HistoId hid = 0;
          if (HistoDef::getHistoIdByName(j->first, hid)) {
            EMUHistoDef hdef(hid);
            book(hdef, j->second, config->getFOLDER_EMU());
          }
        }
      }
    }
  }

  /**
   * @brief  Book FED histograms
   * @param  fedId FED Id
   * @return 
   */
  void Collection::bookFEDHistos(const HwId fedId) const {
    CoHistoMap::const_iterator i = collection.find("FED");
    if (i != collection.end()) {
      const CoHisto hs = i->second;
      for (CoHisto::const_iterator j = hs.begin(); j != hs.end(); j++) {
        std::string s = "";
        if (getHistoValue(j->second, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_FALSE) {
          HistoId hid = 0;
          if (HistoDef::getHistoIdByName(j->first, hid)) {
            FEDHistoDef hdef(hid, fedId);
            book(hdef, j->second, config->getFOLDER_FED());
          }
        }
      }
    }
  }

  /**
   * @brief  Book DDU histograms
   * @param  dduId DDU Id
   * @return 
   */
  void Collection::bookDDUHistos(const HwId dduId) const {
    CoHistoMap::const_iterator i = collection.find("DDU");
    if (i != collection.end()) {
      const CoHisto hs = i->second;
      for (CoHisto::const_iterator j = hs.begin(); j != hs.end(); j++) {
        std::string s = "";
        if (getHistoValue(j->second, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_FALSE) {
          HistoId hid = 0;
          if (HistoDef::getHistoIdByName(j->first, hid)) {
            DDUHistoDef hdef(hid, dduId);
            book(hdef, j->second, config->getFOLDER_DDU());
          }
        }
      }
    }
  }

  /**
   * @brief  Book Chamber Histograms
   * @param  crateId CSC Crate Id
   * @param  dmbId CSC DMB Id
   * @return 
   */
  void Collection::bookCSCHistos(const HwId crateId, const HwId dmbId) const {
    CoHistoMap::const_iterator i = collection.find("CSC");
    if (i != collection.end()) {
      const CoHisto hs = i->second;
      for (CoHisto::const_iterator j = hs.begin(); j != hs.end(); j++) {
        std::string s = "";
        HistoId hid = 0;
        if (HistoDef::getHistoIdByName(j->first, hid)) {
          if (getHistoValue(j->second, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_FALSE) {
            CSCHistoDef hdef(hid, crateId, dmbId);
            book(hdef, j->second, config->getFOLDER_CSC());
          } else {
            int from = 0, to = 0;
            if (checkHistoValue(j->second, XML_BOOK_NAME_FROM, from) &&
                checkHistoValue(j->second, XML_BOOK_NAME_TO, to)) {
              for (int k = from; k <= to; k++) {
                CSCHistoDef hdef(hid, crateId, dmbId, k);
                book(hdef, j->second, config->getFOLDER_CSC());
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief  Book Chamber Histogram with additional identifier (On Demand)
   * @param  hid Histogram Identifier
   * @param  crateId CSC Crate Id
   * @param  dmbId CSC DMB Id
   * @param  addId CSC Additional identifier, ex. Layer Id, ALCT Id, etc.
   * @return 
   */
  void Collection::bookCSCHistos(const HistoId hid, const HwId crateId, const HwId dmbId, const HwId addId) const {
    CoHistoMap::const_iterator i = collection.find("CSC");
    if (i != collection.end()) {
      CoHisto::const_iterator j = i->second.find(h::names[hid]);
      if (j != i->second.end()) {
        CSCHistoDef hdef(hid, crateId, dmbId, addId);
        book(hdef, j->second, config->getFOLDER_CSC());
      }
    }
  }

  /**
   * @brief  Book histogram
   * @param  h Histogram definition to book
   * @param  p Map of Histogram properties
   * @param  folder folder to book histograms to
   * @return 
   */
  void Collection::book(const HistoDef& h, const CoHistoProps& p, const std::string& folder) const {
    MonitorObject* me = nullptr;
    std::string name = h.getName(), type, title, s;

    /** Check if this histogram is included in booking by filters */
    if (!config->needBookMO(h.getFullPath())) {
      LOG_INFO << "MOFilter excluded " << name << " from booking";
      config->fnPutHisto(h, me);
      return;
    }

    int i1, i2, i3;
    double d1, d2, d3, d4, d5, d6;
    bool ondemand =
        (getHistoValue(p, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_TRUE ? true : false);

    if (!checkHistoValue(p, XML_BOOK_HISTO_TYPE, type)) {
      throw Exception("Histogram does not have type!");
    }
    checkHistoValue(p, XML_BOOK_HISTO_TITLE, title);

    if (ondemand) {
      title = h.processTitle(title);
    }

    if (type == "h1") {
      me = config->fnBook(HistoBookRequest(h,
                                           H1D,
                                           type,
                                           folder,
                                           title,
                                           getHistoValue(p, "XBins", i1, 1),
                                           getHistoValue(p, "XMin", d1, 0),
                                           getHistoValue(p, "XMax", d2, 1)));
    } else if (type == "h2") {
      me = config->fnBook(HistoBookRequest(h,
                                           H2D,
                                           type,
                                           folder,
                                           title,
                                           getHistoValue(p, "XBins", i1, 1),
                                           getHistoValue(p, "XMin", d1, 0),
                                           getHistoValue(p, "XMax", d2, 1),
                                           getHistoValue(p, "YBins", i2, 1),
                                           getHistoValue(p, "YMin", d3, 0),
                                           getHistoValue(p, "YMax", d4, 1)));
    } else if (type == "h3") {
      me = config->fnBook(HistoBookRequest(h,
                                           H3D,
                                           type,
                                           folder,
                                           title,
                                           getHistoValue(p, "XBins", i1, 1),
                                           getHistoValue(p, "XMin", d1, 0),
                                           getHistoValue(p, "XMax", d2, 1),
                                           getHistoValue(p, "YBins", i2, 1),
                                           getHistoValue(p, "YMin", d3, 0),
                                           getHistoValue(p, "YMax", d4, 1),
                                           getHistoValue(p, "ZBins", i3, 1),
                                           getHistoValue(p, "ZMin", d5, 0),
                                           getHistoValue(p, "ZMax", d6, 1)));
    } else if (type == "hp") {
      me = config->fnBook(HistoBookRequest(h,
                                           PROFILE,
                                           type,
                                           folder,
                                           title,
                                           getHistoValue(p, "XBins", i1, 1),
                                           getHistoValue(p, "XMin", d1, 0),
                                           getHistoValue(p, "XMax", d2, 1)));
      /*
        HistoBookRequest(h, PROFILE, type, folder, title,
          getHistoValue(p, "XBins", i1, 1),
          getHistoValue(p, "XMin",  d1, 0),
          getHistoValue(p, "XMax",  d2, 1),
          getHistoValue(p, "YBins", i2, 1),
          getHistoValue(p, "YMin",  d3, 0),
          getHistoValue(p, "YMax",  d4, 1)));
	*/
    } else if (type == "hp2") {
      me = config->fnBook(HistoBookRequest(h,
                                           PROFILE2D,
                                           type,
                                           folder,
                                           title,
                                           getHistoValue(p, "XBins", i1, 1),
                                           getHistoValue(p, "XMin", d1, 0),
                                           getHistoValue(p, "XMax", d2, 1),
                                           getHistoValue(p, "YBins", i2, 1),
                                           getHistoValue(p, "YMin", d3, 0),
                                           getHistoValue(p, "YMax", d4, 1),
                                           getHistoValue(p, "ZBins", i3, 1),
                                           getHistoValue(p, "ZMin", d5, 0),
                                           getHistoValue(p, "ZMax", d6, 1)));
    } else {
      throw Exception("Can not book histogram with type: " + type);
    }

    if (me != nullptr) {
      LockType lock(me->mutex);
      TH1* th = me->getTH1Lock();

      if (checkHistoValue(p, "XTitle", s)) {
        if (ondemand) {
          s = h.processTitle(s);
        }
        me->setAxisTitle(s, 1);
      }

      if (checkHistoValue(p, "YTitle", s)) {
        if (ondemand) {
          s = h.processTitle(s);
        }
        me->setAxisTitle(s, 2);
      }

      if (checkHistoValue(p, "ZTitle", s)) {
        if (ondemand) {
          s = h.processTitle(s);
        }
        me->setAxisTitle(s, 3);
      }

      if (checkHistoValue(p, "SetOption", s))
        th->SetOption(s.c_str());
      if (checkHistoValue(p, "SetStats", i1))
        th->SetStats(i1);
      th->SetFillColor(getHistoValue(p, "SetFillColor", i1, DEF_HISTO_COLOR));
      if (checkHistoValue(p, "SetXLabels", s)) {
        std::map<int, std::string> labels;
        ParseAxisLabels(s, labels);
        th->GetXaxis()->SetNoAlphanumeric();  // For ROOT6 to prevent getting zero means values
        for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr) {
          th->GetXaxis()->SetBinLabel(l_itr->first, l_itr->second.c_str());
        }
      }
      if (checkHistoValue(p, "SetYLabels", s)) {
        std::map<int, std::string> labels;
        ParseAxisLabels(s, labels);
        th->GetYaxis()->SetNoAlphanumeric();  // For ROOT6 to prevent getting zero means values
        for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr) {
          th->GetYaxis()->SetBinLabel(l_itr->first, l_itr->second.c_str());
        }
      }
      if (checkHistoValue(p, "LabelOption", s)) {
        std::vector<std::string> v;
        if (2 == Utility::tokenize(s, v, ",")) {
          th->LabelsOption(v[0].c_str(), v[1].c_str());
        }
      }
      if (checkHistoValue(p, "SetLabelSize", s)) {
        std::vector<std::string> v;
        if (2 == Utility::tokenize(s, v, ",")) {
          th->SetLabelSize((double)atof(v[0].c_str()), v[1].c_str());
        }
      }
      if (checkHistoValue(p, "SetTitleOffset", s)) {
        std::vector<std::string> v;
        if (2 == Utility::tokenize(s, v, ",")) {
          th->SetTitleOffset((double)atof(v[0].c_str()), v[1].c_str());
        }
      }
      if (checkHistoValue(p, "SetMinimum", d1))
        th->SetMinimum(d1);
      if (checkHistoValue(p, "SetMaximum", d1))
        me->SetMaximum(d1);
      if (checkHistoValue(p, "SetNdivisionsX", i1)) {
        th->SetNdivisions(i1, "X");
        th->GetXaxis()->CenterLabels(true);
      }
      if (checkHistoValue(p, "SetNdivisionsY", i1)) {
        th->SetNdivisions(i1, "Y");
        th->GetYaxis()->CenterLabels(true);
      }
      if (checkHistoValue(p, "SetTickLengthX", d1))
        th->SetTickLength(d1, "X");
      if (checkHistoValue(p, "SetTickLengthY", d1))
        th->SetTickLength(d1, "Y");
      if (checkHistoValue(p, "SetLabelSizeX", d1))
        th->SetLabelSize(d1, "X");
      if (checkHistoValue(p, "SetLabelSizeY", d1))
        th->SetLabelSize(d1, "Y");
      if (checkHistoValue(p, "SetLabelSizeZ", d1))
        th->SetLabelSize(d1, "Z");
      if (checkHistoValue(p, "SetErrorOption", s))
        reinterpret_cast<TProfile*>(th)->SetErrorOption(s.c_str());

      lock.unlock();
    }

    LOG_DEBUG << "[Collection::book] booked " << h.getFullPath() << " (" << me << ")";

    /** Put histogram into cache */
    config->fnPutHisto(h, me);
  }

  /**
   * @brief  Check if the histogram is on demand (by histogram name)
   * @param  name name of the histogram
   * @return true if this histogram is on demand, false - otherwise
   */
  const bool Collection::isOnDemand(const HistoName& name) const {
    CoHistoMap::const_iterator i = collection.find("CSC");
    if (i != collection.end()) {
      CoHisto hs = i->second;
      CoHisto::const_iterator j = hs.find(name);
      if (j != hs.end()) {
        std::string s;
        return (getHistoValue(j->second, XML_BOOK_ONDEMAND, s, XML_BOOK_ONDEMAND_FALSE) == XML_BOOK_ONDEMAND_TRUE);
      }
    }
    return false;
  }

  /**
  * @brief  Print collection of available histograms and their parameters
  * @return 
  */
  void Collection::printCollection() const {
    std::ostringstream buffer;
    for (CoHistoMap::const_iterator hdmi = collection.begin(); hdmi != collection.end(); hdmi++) {
      buffer << hdmi->first << " [" << std::endl;
      for (CoHisto::const_iterator hdi = hdmi->second.begin(); hdi != hdmi->second.end(); hdi++) {
        buffer << "   " << hdi->first << " [" << std::endl;
        for (CoHistoProps::const_iterator hi = hdi->second.begin(); hi != hdi->second.end(); hi++) {
          buffer << "     " << hi->first << " = " << hi->second << std::endl;
        }
        buffer << "   ]" << std::endl;
      }
      buffer << " ]" << std::endl;
    }
    LOG_INFO << buffer.str();
  }

}  // namespace cscdqm
