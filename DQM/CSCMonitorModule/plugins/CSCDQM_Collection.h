/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Collection.h
 *
 *    Description:  Histogram Booking Collection Management Class
 *
 *        Version:  1.0
 *        Created:  10/30/2008 04:40:38 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Collection_H
#define CSCDQM_Collection_H

#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <TProfile.h>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include "CSCDQM_Exception.h"
#include "CSCDQM_Logger.h"
#include "CSCDQM_Utility.h"
#include "CSCDQM_Configuration.h"

namespace cscdqm {

  /**
   * Constants used for element recognition in Booking XML.
   */
  static const char XML_BOOK_DEFINITION[] = "Definition";
  static const char XML_BOOK_DEFINITION_ID[] = "id";
  static const char XML_BOOK_HISTOGRAM[] = "Histogram";
  static const char XML_BOOK_DEFINITION_REF[] = "ref";
  static const char XML_BOOK_HISTO_NAME[] = "Name";
  static const char XML_BOOK_HISTO_PREFIX[] = "Prefix";
  static const char XML_BOOK_HISTO_TYPE[] = "Type";
  static const char XML_BOOK_HISTO_TITLE[] = "Title";
  static const char XML_BOOK_ONDEMAND[] = "OnDemand";
  static const char XML_BOOK_ONDEMAND_TRUE[] = "1";
  static const char XML_BOOK_ONDEMAND_FALSE[] = "0";
  static const char XML_BOOK_NAME_FROM[] = "Name_from";
  static const char XML_BOOK_NAME_TO[] = "Name_to";

  /** Default histogram color */
  static const int DEF_HISTO_COLOR = 48;

  /** List of Single Histogram properties */
  typedef std::map<std::string, std::string> CoHistoProps;
  /** List of Histograms */
  typedef std::map<std::string, CoHistoProps> CoHisto;
  /** List of Histogram Types */
  typedef std::map<std::string, CoHisto> CoHistoMap;

  /**
   * @class Collection
   * @brief Manage collection of histograms, load histogram definitions from
   * XML file and book histograms by calling MonitorObjectProvider routines.  
   */
  class Collection {
  public:
    typedef xercesc::DOMDocument DOMDocument;
    typedef xercesc::DOMElement DOMElement;
    typedef xercesc::DOMNode DOMNode;
    typedef xercesc::DOMNodeList DOMNodeList;
    typedef xercesc::DOMNamedNodeMap DOMNamedNodeMap;
    typedef xercesc::XMLException XMLException;
    typedef xercesc::XMLString XMLString;
    typedef xercesc::XMLPlatformUtils XMLPlatformUtils;
    typedef xercesc::XercesDOMParser XercesDOMParser;
    Collection(Configuration* const p_config);
    void load();

    void bookEMUHistos() const;
    void bookFEDHistos(const HwId fedId) const;
    void bookDDUHistos(const HwId dduId) const;
    void bookCSCHistos(const HwId crateId, const HwId dmbId) const;
    void bookCSCHistos(const HistoId hid, const HwId crateId, const HwId dmbId, const HwId addId) const;

    const bool isOnDemand(const HistoName& name) const;

    void printCollection() const;

  private:
    static const bool checkHistoValue(const CoHistoProps& h, const std::string& name, std::string& value);
    static const bool checkHistoValue(const CoHistoProps& h, const std::string& name, int& value);
    static const bool checkHistoValue(const CoHistoProps& h, const std::string name, double& value);

    static std::string& getHistoValue(const CoHistoProps& h,
                                      const std::string& name,
                                      std::string& value,
                                      const std::string& def_value = "");
    static int& getHistoValue(const CoHistoProps& h, const std::string& name, int& value, const int& def_value = 0);
    static double& getHistoValue(const CoHistoProps& h,
                                 const std::string name,
                                 double& value,
                                 const int def_value = 0.0);

    void book(const HistoDef& h, const CoHistoProps& p, const std::string& folder) const;
    static const int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels);
    static void getNodeProperties(DOMNode*& node, CoHistoProps& hp);

    Configuration* config;
    CoHistoMap collection;
  };

}  // namespace cscdqm

#endif
