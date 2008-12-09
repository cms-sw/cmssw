/*
 * =====================================================================================
 *
 *       Filename:  Configuration.h
 *
 *    Description:  CSCDQM Configuration parameter storage
 *
 *        Version:  1.0
 *        Created:  10/03/2008 10:26:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius, valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Configuration_H
#define CSCDQM_Configuration_H

#include <string>
#include <sstream>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>

#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comparison/equal.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#ifdef DQMGLOBAL

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#endif

#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectProvider.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Exception.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"

#define CONFIG_PARAMETERS_SEQ \
  \
  \
  (( bool, BINCHECKER_CRC_ALCT, false )) \
  (( bool, BINCHECKER_CRC_CLCT, false )) \
  (( bool, BINCHECKER_CRC_CFEB, false )) \
  (( bool, BINCHECKER_OUTPUT,   false )) \
  (( bool, FRAEFF_AUTO_UPDATE,  false )) \
  (( bool, FRAEFF_SEPARATE_THREAD,  false )) \
  (( std::string, BOOKING_XML_FILE, "" )) \
  (( std::string, FOLDER_EMU, "" )) \
  (( std::string, FOLDER_DDU, "" )) \
  (( std::string, FOLDER_CSC, "" )) \
  (( std::string, FOLDER_PAR, "" )) \
  (( unsigned int, DDU_CHECK_MASK,    0xFFFFFFFF )) \
  (( unsigned int, DDU_BINCHECK_MASK, 0x02080016 )) \
  (( unsigned int, BINCHECK_MASK,     0xFFFFFFFF )) \
  (( unsigned int, FRAEFF_AUTO_UPDATE_START, 5 )) \
  (( unsigned int, FRAEFF_AUTO_UPDATE_FREQ,  1 )) \
  (( double, EFF_COLD_THRESHOLD,   0.1 )) \
  (( double, EFF_COLD_SIGFAIL,     5.0 )) \
  (( double, EFF_HOT_THRESHOLD,    0.1 )) \
  (( double, EFF_HOT_SIGFAIL,      5.0 )) \
  (( double, EFF_ERR_THRESHOLD,    0.1 )) \
  (( double, EFF_ERR_SIGFAIL,      5.0 )) \
  (( double, EFF_NODATA_THRESHOLD, 0.1 )) \
  (( double, EFF_NODATA_SIGFAIL,   5.0 )) \
  \
  \
  /* */

#define CONFIG_PARAMETER_DEFINE_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(3, 0, elem) BOOST_PP_TUPLE_ELEM(3, 1, elem);

#define CONFIG_PARAMETER_DEFAULT_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(3, 1, elem) = BOOST_PP_TUPLE_ELEM(3, 2, elem);

#define CONFIG_PARAMETER_GETTER_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(3, 0, elem) BOOST_PP_CAT(get, BOOST_PP_TUPLE_ELEM(3, 1, elem))() { \
    return BOOST_PP_TUPLE_ELEM(3, 1, elem); \
  } \

#define CONFIG_PARAMETER_SETTER_MACRO(r, data, i, elem) \
  void BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(3, 1, elem))(BOOST_PP_TUPLE_ELEM(3, 0, elem) p) { \
    BOOST_PP_TUPLE_ELEM(3, 1, elem) = p; \
  } \

#define CONFIG_PARAMETER_LOADPS_MACRO(r, data, i, elem) \
  BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(3, 1, elem))(ps.getUntrackedParameter<BOOST_PP_TUPLE_ELEM(3, 0, elem)>(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 1, elem)), BOOST_PP_TUPLE_ELEM(3, 2, elem)));

#define CONFIG_PARAMETER_LOADXML_MACRO(r, data, i, elem) \
  if (nodeName.compare(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 1, elem))) == 0) { \
    stm >> BOOST_PP_TUPLE_ELEM(3, 1, elem); \
    continue; \
  } \

namespace cscdqm {

  using namespace XERCES_CPP_NAMESPACE;

  /**
   * @class Configuration
   * @brief Framework configuration
   */
  class Configuration {

    private:

      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFINE_MACRO, _, CONFIG_PARAMETERS_SEQ)

    public:
      
      boost::function< bool (const HistoDef& histoT, MonitorObject*&) > fnGetHisto;
      boost::function< void (const HistoDef& histoT, MonitorObject*&) > fnPutHisto;
      boost::function< MonitorObject* (const HistoBookRequest&) > fnBook;
      boost::function< CSCDetId (const unsigned int, const unsigned int) > fnGetCSCDetId;
      boost::function< bool (unsigned int&, unsigned int&, unsigned int&) > fnNextBookedCSC;

      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_GETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)
      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_SETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)

      Configuration() {

        BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFAULT_MACRO, _, CONFIG_PARAMETERS_SEQ)

        reset();

      }

      void load(const std::string configFile) {
        XMLPlatformUtils::Initialize();
        boost::shared_ptr<XercesDOMParser> parser(new XercesDOMParser());
        parser->setValidationScheme(XercesDOMParser::Val_Always);
        parser->setDoNamespaces(true);
        parser->setDoSchema(true);
        parser->setExitOnFirstFatalError(true);
        parser->setValidationConstraintFatal(true);
        XMLFileErrorHandler eh;
        parser->setErrorHandler(&eh);
        parser->parse(configFile.c_str());
        DOMDocument *doc = parser->getDocument();
        DOMNode *docNode = (DOMNode*) doc->getDocumentElement();

        DOMNodeList *itemList = docNode->getChildNodes();
        for(uint32_t i = 0; i < itemList->getLength(); i++) {
          DOMNode* node = itemList->item(i);
          if (node->getNodeType() != DOMNode::ELEMENT_NODE) { continue; }
          std::string nodeName = XMLString::transcode(node->getNodeName());
          std::string value = XMLString::transcode(node->getTextContent());
          std::istringstream stm(value);

          BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_LOADXML_MACRO, _, CONFIG_PARAMETERS_SEQ)

        }
      }

#ifdef DQMGLOBAL

      void load(const edm::ParameterSet& ps) {
        BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_LOADPS_MACRO, _, CONFIG_PARAMETERS_SEQ)
      }

#endif

    /**
      * Counters
      */

    public:

      void reset() {
        nEvents = 0;
        nEventsBad = 0;
        nEventsGood = 0;
        nEventsCSC = 0;
        nUnpackedDMB = 0;
      }

      const unsigned long getNEvents() const      { return nEvents; }
      const unsigned long getNEventsBad() const   { return nEventsBad; }
      const unsigned long getNEventsGood() const  { return nEventsGood; }
      const unsigned long getNEventsCSC() const   { return nEventsCSC; }
      const unsigned long getNUnpackedDMB() const { return nUnpackedDMB; }

      void incNEvents()      { nEvents++; }
      void incNEventsBad()   { nEventsBad++; }
      void incNEventsGood()  { nEventsGood++; }
      void incNEventsCSC()   { nEventsCSC++; }
      void incNUnpackedDMB() { nUnpackedDMB++; }

    private:

      unsigned long nEvents;
      unsigned long nEventsBad;
      unsigned long nEventsGood;
      unsigned long nEventsCSC;
      unsigned long nUnpackedDMB; 

  };

}

#undef CONFIG_PARAMETERS_SEQ
#undef CONFIG_PARAMETER_DEFINE_MACRO
#undef CONFIG_PARAMETER_DEFAULT_MACRO
#undef CONFIG_PARAMETER_GETTER_MACRO
#undef CONFIG_PARAMETER_SETTER_MACRO
#undef CONFIG_PARAMETER_LOADPS_MACRO
#undef CONFIG_PARAMETER_LOADXML_MACRO

#endif
