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
#include <xercesc/dom/DOMComment.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/dom/DOM.hpp>

#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comparison/equal.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <boost/timer.hpp>

#ifdef DQMGLOBAL

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#endif

#include "DQM/CSCMonitorModule/interface/CSCDQM_MonitorObjectProvider.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Exception.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Utility.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"

#define CONFIG_PARAMETERS_SEQ_LEN 4

#define CONFIG_PARAMETERS_SEQ \
  \
  \
  (( bool, PROCESS_DDU, true, "enter DDU (and latter Chamber) sections (EventProcessor flag)" )) \
  (( bool, PROCESS_CSC, true, "enter Chamber section (EventProcessor flag)" )) \
  (( bool, PROCESS_EFF_HISTOS, true, "calculate efficiency histograms (Dispatcher flag)" )) \
  (( bool, PROCESS_EFF_PARAMETERS, true, "calculate efficiency parameters (EventProcessor flag)" )) \
  (( bool, BINCHECKER_CRC_ALCT, false , "check ALCT CRC (CSCDCCExaminer flag)" )) \
  (( bool, BINCHECKER_CRC_CLCT, false , "check CLCT CRC (CSCDCCExaminer flag)" )) \
  (( bool, BINCHECKER_CRC_CFEB, false , "check CFEB CRC (CSCDCCExaminer flag)" )) \
  (( bool, BINCHECKER_MODE_DDU, true , "set DDU mode (CSCDCCExaminer flag)" )) \
  (( bool, BINCHECKER_OUTPUT,   false , "print 1 and 2 output (CSCDCCExaminer flag)" )) \
  (( bool, FRAEFF_AUTO_UPDATE,  false , "start fractional and efficiency histogram update automatically (Dispatcher flag)" )) \
  (( bool, FRAEFF_SEPARATE_THREAD,  false , "start fractional and efficiency histogram update on separate thread (EventProcessor flag)" )) \
  (( std::string, BOOKING_XML_FILE, "" , "histogram description (booking) file in XML format (Collection)" )) \
  (( std::string, FOLDER_EMU, "" , "root file folder name to be used for EMU histograms (EventProcessor)" )) \
  (( std::string, FOLDER_DDU, "" , "root file folder name to be used for DDU histograms (EventProcessor)" )) \
  (( std::string, FOLDER_CSC, "" , "root file folder name to be used for CSC histograms (EventProcessor)" )) \
  (( std::string, FOLDER_PAR, "" , "root file folder name to be used for parameters (EventProcessor)" )) \
  (( unsigned int, DDU_CHECK_MASK,    0xFFFFFFFF , "mask for cumulative EmuFileReader DDU error flags (EventProcessor)" )) \
  (( unsigned int, DDU_BINCHECK_MASK, 0x02080016 , "mask for DDU level examiner errors (CSCDCCExaminer)" )) \
  (( unsigned int, BINCHECK_MASK,     0xFFFFFFFF , "mask for chamber level examiner errors (CSCDCCExaminer)" )) \
  (( unsigned int, FRAEFF_AUTO_UPDATE_START, 5 , "event number to start automatic fractional and efficiency histogram updates from (Dispatcer)" )) \
  (( unsigned int, FRAEFF_AUTO_UPDATE_FREQ,  1 , "frequency in events to perform automatic fractional and efficiency histogram updates (Dispatcher)" )) \
  (( double, EFF_COLD_THRESHOLD,   0.1 , "threshold in fraction to check for cold (not reporting) HW (EventProcessor)" )) \
  (( double, EFF_COLD_SIGFAIL,     5.0 , "statistical significance for cold (not reporting) HW (EventProcessor)" )) \
  (( double, EFF_HOT_THRESHOLD,    0.1 , "threshold in fraction to check for hot HW (EventProcessor)" )) \
  (( double, EFF_HOT_SIGFAIL,      5.0 , "statistical significance for hot HW (EventProcessor)" )) \
  (( double, EFF_ERR_THRESHOLD,    0.1 , "threshold in fraction to check for errors in HW (EventProcessor)" )) \
  (( double, EFF_ERR_SIGFAIL,      5.0 , "statistical significance for errors in HW (EventProcessor)" )) \
  (( double, EFF_NODATA_THRESHOLD, 0.1 , "threshold in fraction to check for not reporting elements in HW (EventProcessor)" )) \
  (( double, EFF_NODATA_SIGFAIL,   5.0 , "statistical significance for not reportingelements in HW (EventProcessor)" )) \
  (( unsigned int, EVENTS_ECHO, 1000, "frequency in events to print echo message (EventProcessor)" )) \
  \
  \
  /* */

#define CONFIG_PARAMETER_DEFINE_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem) BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem);

#define CONFIG_PARAMETER_DEFAULT_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem) = BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 2, elem);

#define CONFIG_PARAMETER_GETTER_MACRO(r, data, i, elem) \
  const BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem) BOOST_PP_CAT(get, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))() const { \
    return BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem); \
  } \

#define CONFIG_PARAMETER_SETTER_MACRO(r, data, i, elem) \
  void BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem) p) { \
    BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem) = p; \
  } \

#define CONFIG_PARAMETER_LOADPS_MACRO(r, data, i, elem) \
  BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))(ps.getUntrackedParameter<BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem)>(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem)), BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 2, elem)));

#define CONFIG_PARAMETER_LOADXML_MACRO(r, data, i, elem) \
  if (nodeName.compare(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))) == 0) { \
    stm >> BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem); \
    continue; \
  } \

#define CONFIG_PARAMETER_PRINTXML_MACRO(r, data, i, elem) \
  { \
    DOMComment* comment = doc->createComment(XERCES_TRANSCODE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 3, elem))); \
    DOMElement* el = doc->createElement(XERCES_TRANSCODE(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem)))); \
    std::string value = toString(config.BOOST_PP_CAT(get, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))()); \
    DOMText* tdata = doc->createTextNode(XERCES_TRANSCODE(value.c_str())); \
    el->appendChild(tdata); \
    rootElem->appendChild(comment); \
    rootElem->appendChild(el); \
  } \

namespace cscdqm {

  using namespace XERCES_CPP_NAMESPACE;

  typedef struct MOFilterItem {

    TPRegexp pattern;
    bool include;

    MOFilterItem(const std::string pattern_, const bool include_) :
      pattern(pattern_.c_str()), include(include_) { }

  };

  /**
   * @class Configuration
   * @brief Framework configuration
   */
  class Configuration {

    private:

      bool printStatsOnExit;
      std::vector<MOFilterItem> MOFilterItems;
      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFINE_MACRO, _, CONFIG_PARAMETERS_SEQ)

    public:
      
      boost::function< bool (const HistoDef& histoT, MonitorObject*&) > fnGetHisto;
      boost::function< bool (const HistoId id, MonitorObject*& mo) > fnGetCacheEMUHisto;
      boost::function< bool (const HistoId id, const HwId& id1, MonitorObject*& mo) > fnGetCacheDDUHisto;
      boost::function< bool (const HistoId id, const HwId& id1, const HwId& id2, const HwId& id3, MonitorObject*& mo) > fnGetCacheCSCHisto;
      boost::function< bool (const HistoId id, MonitorObject*& mo) > fnGetCacheParHisto;
      boost::function< void (const HistoDef& histoT, MonitorObject*&) > fnPutHisto;
      boost::function< MonitorObject* (const HistoBookRequest&) > fnBook;
      boost::function< CSCDetId (const unsigned int, const unsigned int) > fnGetCSCDetId;
      boost::function< bool (unsigned int&, unsigned int&, unsigned int&) > fnNextBookedCSC;
      boost::function< bool (unsigned int&, unsigned int&) > fnIsBookedCSC;
      boost::function< bool (unsigned int&) > fnIsBookedDDU;

      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_GETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)
      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_SETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)

      Configuration(const bool p_printStatsOnExit = true) {
        printStatsOnExit = p_printStatsOnExit;
        BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFAULT_MACRO, _, CONFIG_PARAMETERS_SEQ)
        reset();
      }

      ~Configuration() {
        if (printStatsOnExit) {
          printStats();
        }
      }

      void load(const std::string& configFile) {
        XMLPlatformUtils::Initialize();
        boost::shared_ptr<XercesDOMParser> parser(new XercesDOMParser());

        /*
        parser->setValidationScheme(XercesDOMParser::Val_Always);
        parser->setDoNamespaces(true);
        parser->setDoSchema(false);
        parser->setExitOnFirstFatalError(true);
        parser->setValidationConstraintFatal(true);
        */

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

          if (nodeName.compare("MO_FILTER") == 0) {
            DOMNodeList *filterList = node->getChildNodes();
            for(uint32_t j = 0; j < filterList->getLength(); j++) {
              DOMNode* filter = filterList->item(j);
              if (filter->getNodeType() != DOMNode::ELEMENT_NODE) { continue; }
              std::string filterName = XMLString::transcode(filter->getNodeName());
              std::string filterValue = XMLString::transcode(filter->getTextContent());
              MOFilterItems.insert(MOFilterItems.end(), MOFilterItem(filterValue, (filterName.compare("INCLUDE") == 0)));
            }
          }

        }

        //doc->release();
        //XMLPlatformUtils::Terminate();

      }

      static void printXML(const Configuration& config) {
        XMLPlatformUtils::Initialize();

        DOMImplementation* domImpl = DOMImplementationRegistry::getDOMImplementation(XERCES_TRANSCODE("core"));
        DOMDocument *doc = domImpl->createDocument(0, XERCES_TRANSCODE("processor_configuration"), 0);
        DOMElement* rootElem = doc->getDocumentElement();

        BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_PRINTXML_MACRO, _, CONFIG_PARAMETERS_SEQ)

        DOMWriter *ser = domImpl->createDOMWriter();
        if (ser->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true)) {
          ser->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
        }
        XMLFileErrorHandler eh;
        ser->setErrorHandler((DOMErrorHandler*) &eh);
        ser->writeNode(new StdOutFormatTarget(), *doc);

        doc->release();
        XMLPlatformUtils::Terminate();
      }

#ifdef DQMGLOBAL

      void load(const edm::ParameterSet& ps) {
        BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_LOADPS_MACRO, _, CONFIG_PARAMETERS_SEQ)
      }

#endif

    /**
      * Statistics
      */

    private:

      boost::timer globalTimer;
      boost::timer eventTimer;
      boost::timer fraTimer;
      boost::timer effTimer;
      double eventTimeSum;
      double fraTimeSum;
      double effTimeSum;

    public:

#define STATFIELD(caption, value, units) \
      logger << std::setfill(' '); \
      logger << std::setiosflags(std::ios::right) << std::setw(25) << caption << ": "; \
      logger << std::setiosflags(std::ios::right) << std::setw(12); \
      if (value < 0) { \
        logger << "NA"; \
      } else { \
        logger << value; \
      } \
      logger << std::setiosflags(std::ios::left) << std::setw(2) << units; \
      logger << std::endl;
#define SEPFIELD \
      logger << std::setfill('-'); \
      logger << std::setw(25) << ""; \
      logger << std::setw(10) << ""; \
      logger << std::setw(2)  << ""; \
      logger << std::endl;

      void printStats() {

        double allTime = globalTimer.elapsed();
        LogInfo logger;
        logger << std::endl;

        STATFIELD("Events processed", nEvents, "")
        STATFIELD("Bad events: ", nEventsBad, "")
        STATFIELD("Good events: ", nEventsGood, "")
        STATFIELD("CSC events: ", nEventsCSC, "")
        STATFIELD("Unpacked DMBs: ", nUnpackedDMB, "")

        SEPFIELD

        STATFIELD("All event time: ", eventTimeSum, "s")
        double eventTimeAverage = (nEvents > 0 ? eventTimeSum / nEvents : -1.0);
        STATFIELD("Avg. event time: ", eventTimeAverage, "s")
        double eventRateAverage = (eventTimeSum > 0 ? nEvents / eventTimeSum : -1.0);
        STATFIELD("Avg. event rate: ", eventRateAverage, "Hz")

        SEPFIELD

        STATFIELD("All fra update time: ", fraTimeSum, "s")
        double fraTimeAverage = (fraCount > 0 ? fraTimeSum / fraCount : -1.0);
        STATFIELD("Avg. fra update time: ", fraTimeAverage, "s")

        SEPFIELD

        STATFIELD("All eff update time: ", effTimeSum, "s")
        double effTimeAverage = (effCount > 0 ? effTimeSum / effCount : -1.0);
        STATFIELD("Avg. eff update time: ", effTimeAverage, "s")

        SEPFIELD

        STATFIELD("All time: ", allTime, "s")
        double allTimeAverage = (nEvents > 0 ? allTime / nEvents : -1.0);
        STATFIELD("Avg. event all time: ", allTimeAverage, "s")
        double allRateAverage = (allTime > 0 ? nEvents / allTime : -1.0);
        STATFIELD("Avg. event all rate: ", allRateAverage, "Hz")

      }

#undef STATFIELD
#undef SEPFIELD

      const bool needBookMO(const std::string name) const {
        bool result = true;
        for (unsigned int i = 0; i < MOFilterItems.size(); i++) {
          const MOFilterItem* filter = &MOFilterItems.at(i);
          if (Utility::regexMatch(filter->pattern, name)) result = filter->include;
        }
        return result;
      }

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
        fraCount = 0;
        effCount = 0;
        eventTimeSum = 0.0;
        fraTimeSum = 0.0;
        effTimeSum = 0.0;
      }

      const unsigned long getNEvents() const      { return nEvents; }
      const unsigned long getNEventsBad() const   { return nEventsBad; }
      const unsigned long getNEventsGood() const  { return nEventsGood; }
      const unsigned long getNEventsCSC() const   { return nEventsCSC; }
      const unsigned long getNUnpackedDMB() const { return nUnpackedDMB; }

      void eventProcessTimer(const bool start) {
        if (start) {
          eventTimer.restart();
        } else {
          eventTimeSum += eventTimer.elapsed();
        }
      }

      void updateFraTimer(const bool start) {
        if (start) {
          fraTimer.restart();
        } else {
          fraTimeSum += fraTimer.elapsed();
          fraCount++;
        }
      }

      void updateEffTimer(const bool start) {
        if (start) {
          effTimer.restart();
        } else {
          effTimeSum += effTimer.elapsed();
          effCount++;
        }
      }

      void incNEvents()      { 
        nEvents++; 
        if (getEVENTS_ECHO() > 0) {
          if (getNEvents() % getEVENTS_ECHO() == 0) {
            LOG_INFO << "(echo) Events processed: " << std::setw(12) << getNEvents();
          }
        }
      }
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
      unsigned long fraCount; 
      unsigned long effCount;



  };

}

#undef CONFIG_PARAMETERS_SEQ_LEN
#undef CONFIG_PARAMETERS_SEQ
#undef CONFIG_PARAMETER_DEFINE_MACRO
#undef CONFIG_PARAMETER_DEFAULT_MACRO
#undef CONFIG_PARAMETER_GETTER_MACRO
#undef CONFIG_PARAMETER_SETTER_MACRO
#undef CONFIG_PARAMETER_LOADPS_MACRO
#undef CONFIG_PARAMETER_LOADXML_MACRO
#undef CONFIG_PARAMETER_PRINTXML_MACRO

#endif
