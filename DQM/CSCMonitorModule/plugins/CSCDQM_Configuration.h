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
#include <functional>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMComment.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/dom/DOM.hpp>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include "boost/tuple/tuple.hpp"

#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comparison/equal.hpp>

#include <boost/timer.hpp>

/** Headers for Global DQM Only */
#ifdef DQMGLOBAL

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#endif

#include "CSCDQM_MonitorObjectProvider.h"
#include "CSCDQM_Exception.h"
#include "CSCDQM_Utility.h"
#include "CSCDQM_Logger.h"

/** Length of Global Parameter sequence */
#define CONFIG_PARAMETERS_SEQ_LEN 4

/** 
 * Sequence of Global Parameters. Add new line or edit existing in form:
 * (( type (C++), name (upper case), default value, "description" )) \
 */
#define CONFIG_PARAMETERS_SEQ                                                                                           \
                                                                                                                        \
  ((bool, PROCESS_DDU, true, "enter DDU (and latter Chamber) sections (EventProcessor flag)"))(                         \
      (bool, PROCESS_CSC, true, "enter Chamber section (EventProcessor flag)"))(                                        \
      (bool, PROCESS_EFF_HISTOS, true, "calculate efficiency histograms (Dispatcher flag)"))(                           \
      (bool, PROCESS_EFF_PARAMETERS, true, "calculate efficiency parameters (EventProcessor flag)"))(                   \
      (bool, BINCHECKER_CRC_ALCT, false, "check ALCT CRC (CSCDCCExaminer flag)"))(                                      \
      (bool, BINCHECKER_CRC_CLCT, false, "check CLCT CRC (CSCDCCExaminer flag)"))(                                      \
      (bool, BINCHECKER_CRC_CFEB, false, "check CFEB CRC (CSCDCCExaminer flag)"))(                                      \
      (bool, BINCHECKER_MODE_DDU, true, "set DDU mode (CSCDCCExaminer flag)"))(                                         \
      (bool, BINCHECKER_OUTPUT, false, "print 1 and 2 output (CSCDCCExaminer flag)"))(                                  \
      (bool,                                                                                                            \
       FRAEFF_AUTO_UPDATE,                                                                                              \
       false,                                                                                                           \
       "start fractional and efficiency histogram update automatically (Dispatcher flag)"))(                            \
      (bool,                                                                                                            \
       FRAEFF_SEPARATE_THREAD,                                                                                          \
       false,                                                                                                           \
       "start fractional and efficiency histogram update on separate thread (EventProcessor flag)"))(                   \
      (bool, PRINT_STATS_ON_EXIT, true, "print statistics on exit (destruction)"))(                                     \
      (bool, IN_FULL_STANDBY, true, "full detector is in standby mode from the beginning of the run"))(                 \
      (std::string, BOOKING_XML_FILE, "", "histogram description (booking) file in XML format (Collection)"))(          \
      (std::string, FOLDER_EMU, "", "root file folder name to be used for EMU histograms (EventProcessor)"))(           \
      (std::string, FOLDER_DDU, "", "root file folder name to be used for DDU histograms (EventProcessor)"))(           \
      (std::string, FOLDER_CSC, "", "root file folder name to be used for CSC histograms (EventProcessor)"))(           \
      (std::string, FOLDER_PAR, "", "root file folder name to be used for parameters (EventProcessor)"))((              \
      unsigned int, DDU_CHECK_MASK, 0xFFFFFFFF, "mask for cumulative EmuFileReader DDU error flags (EventProcessor)"))( \
      (unsigned int, DDU_BINCHECK_MASK, 0x02080016, "mask for DDU level examiner errors (CSCDCCExaminer)"))(            \
      (unsigned int, BINCHECK_MASK, 0xFFFFFFFF, "mask for chamber level examiner errors (CSCDCCExaminer)"))(            \
      (unsigned int,                                                                                                    \
       FRAEFF_AUTO_UPDATE_START,                                                                                        \
       5,                                                                                                               \
       "event number to start automatic fractional and efficiency histogram updates from (Dispatcer)"))(                \
      (unsigned int,                                                                                                    \
       FRAEFF_AUTO_UPDATE_FREQ,                                                                                         \
       1,                                                                                                               \
       "frequency in events to perform automatic fractional and efficiency histogram updates (Dispatcher)"))((          \
      double, EFF_COLD_THRESHOLD, 0.1, "threshold in fraction to check for cold (not reporting) HW (EventProcessor)"))( \
      (double, EFF_COLD_SIGFAIL, 5.0, "statistical significance for cold (not reporting) HW (EventProcessor)"))(        \
      (double, EFF_HOT_THRESHOLD, 0.1, "threshold in fraction to check for hot HW (EventProcessor)"))(                  \
      (double, EFF_HOT_SIGFAIL, 5.0, "statistical significance for hot HW (EventProcessor)"))(                          \
      (double, EFF_ERR_THRESHOLD, 0.1, "threshold in fraction to check for errors in HW (EventProcessor)"))(            \
      (double, EFF_ERR_SIGFAIL, 5.0, "statistical significance for errors in HW (EventProcessor)"))(                    \
      (double,                                                                                                          \
       EFF_NODATA_THRESHOLD,                                                                                            \
       0.1,                                                                                                             \
       "threshold in fraction to check for not reporting elements in HW (EventProcessor)"))(                            \
      (double, EFF_NODATA_SIGFAIL, 5.0, "statistical significance for not reportingelements in HW (EventProcessor)"))(  \
      (unsigned int, EVENTS_ECHO, 1000, "frequency in events to print echo message (EventProcessor)"))(                 \
      (std::string, FOLDER_FED, "", "root file folder name to be used for FED histograms (EventProcessor)"))(           \
      (bool, PREBOOK_ALL_HISTOS, true, "pre-book all FED, DDU, CSC histogragrams before run begins"))

/**
 * Global Parameter Manipulation macros.
 */

/** Parameter as class property definition */
#define CONFIG_PARAMETER_DEFINE_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem) BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem);

/** Parameter Default value definition (in constructor) */
#define CONFIG_PARAMETER_DEFAULT_MACRO(r, data, i, elem) \
  BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem) = BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 2, elem);

/** Parameter Getter method */
#define CONFIG_PARAMETER_GETTER_MACRO(r, data, i, elem)                                    \
  const BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem)                            \
      BOOST_PP_CAT(get, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))() const { \
    return BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem);                        \
  }

/** Parameter Setter method */
#define CONFIG_PARAMETER_SETTER_MACRO(r, data, i, elem)                            \
  void BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))( \
      BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem) p) {                 \
    BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem) = p;                   \
  }

#ifdef DQMGLOBAL

/** Load parameter from parameters set line (Global DQM) */
#define CONFIG_PARAMETER_LOADPS_MACRO(r, data, i, elem)                               \
  BOOST_PP_CAT(set, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))          \
  (ps.getUntrackedParameter<BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 0, elem)>( \
      BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem)),    \
                         BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 2, elem)));

#endif

#ifdef DQMLOCAL

/** Load parameter from XML node line (Local DQM) */
#define CONFIG_PARAMETER_LOADXML_MACRO(r, data, i, elem)                                                    \
  if (nodeName.compare(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))) == 0) { \
    stm >> BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem);                                         \
    continue;                                                                                               \
  }

/** Include parameter into XML stream for printing */
#define CONFIG_PARAMETER_PRINTXML_MACRO(r, data, i, elem)                                                              \
  {                                                                                                                    \
    DOMComment* comment =                                                                                              \
        doc->createComment(XERCES_TRANSCODE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 3, elem)));                 \
    DOMElement* el = doc->createElement(                                                                               \
        XERCES_TRANSCODE(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))));                \
    std::string value = toString(config.BOOST_PP_CAT(get, BOOST_PP_TUPLE_ELEM(CONFIG_PARAMETERS_SEQ_LEN, 1, elem))()); \
    DOMText* tdata = doc->createTextNode(XERCES_TRANSCODE(value.c_str()));                                             \
    el->appendChild(tdata);                                                                                            \
    rootElem->appendChild(comment);                                                                                    \
    rootElem->appendChild(el);                                                                                         \
  }

#endif

namespace cscdqm {

  /** @brief MO filter Item definition (loaded from XML/PSet) */
  struct MOFilterItem {
    /** Regexp filter pattern */
    TPRegexp pattern;
    /** Include filtered item or not */
    bool include;
    /** Constructor */
    MOFilterItem(const std::string pattern_, const bool include_) : pattern(pattern_.c_str()), include(include_) {}
  };

  /** @brief Chamber level counter types */
  enum ChamberCounterType {
    DMB_EVENTS,
    BAD_EVENTS,
    DMB_TRIGGERS,
    ALCT_TRIGGERS,
    CLCT_TRIGGERS,
    CFEB_TRIGGERS,
    EVENT_DISPLAY_PLOT
  };

  /** Single Chamber counters type */
  typedef std::map<ChamberCounterType, uint32_t> ChamberCounterMapType;

  /** @brief Chamber Counters key type */
  struct ChamberCounterKeyType {
    HwId crateId;
    HwId dmbId;
    ChamberCounterMapType counters;
    ChamberCounterKeyType(const HwId& crateId_, const HwId& dmbId_, const ChamberCounterMapType& c_)
        : crateId(crateId_), dmbId(dmbId_), counters(c_) {}
  };

  /** Map of Chamber Counters Type */
  typedef boost::multi_index_container<
      ChamberCounterKeyType,
      boost::multi_index::indexed_by<boost::multi_index::ordered_unique<boost::multi_index::composite_key<
          ChamberCounterKeyType,
          boost::multi_index::member<ChamberCounterKeyType, HwId, &ChamberCounterKeyType::crateId>,
          boost::multi_index::member<ChamberCounterKeyType, HwId, &ChamberCounterKeyType::dmbId> > > > >
      ChamberMapCounterMapType;

  /**
   * @class Configuration
   * @brief CSCDQM Framework Global Configuration
   */
  class Configuration {
  private:
    unsigned short printStatsLocal;

    /** Map of MO Filters */
    std::vector<MOFilterItem> MOFilterItems;

    /** Define parameters */
    BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFINE_MACRO, _, CONFIG_PARAMETERS_SEQ)

    /**
       * @brief  Initialize parameter values and reset counters (used by constructors)
       * @return 
       */
    void init() {
      /** Assign default values to parameters */
      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_DEFAULT_MACRO, _, CONFIG_PARAMETERS_SEQ)
      reset();
    }

  public:
    /**
        * Pointers to Shared functions (created in Dispatcher)
        */

    /** Get MO Globally */
    std::function<bool(const HistoDef& histoT, MonitorObject*&)> fnGetHisto;

    /** Pointers to Cache Functions */
    std::function<bool(const HistoId id, MonitorObject*& mo)> fnGetCacheEMUHisto;
    std::function<bool(const HistoId id, const HwId& id1, MonitorObject*& mo)> fnGetCacheFEDHisto;
    std::function<bool(const HistoId id, const HwId& id1, MonitorObject*& mo)> fnGetCacheDDUHisto;
    std::function<bool(const HistoId id, const HwId& id1, const HwId& id2, const HwId& id3, MonitorObject*& mo)>
        fnGetCacheCSCHisto;
    std::function<bool(const HistoId id, MonitorObject*& mo)> fnGetCacheParHisto;
    std::function<void(const HistoDef& histoT, MonitorObject*&)> fnPutHisto;
    std::function<bool(unsigned int&, unsigned int&, unsigned int&)> fnNextBookedCSC;
    std::function<bool(unsigned int&, unsigned int&)> fnIsBookedCSC;
    std::function<bool(unsigned int&)> fnIsBookedDDU;
    std::function<bool(unsigned int&)> fnIsBookedFED;

    /** Pointer to Collection Book Function */
    std::function<MonitorObject*(const HistoBookRequest&)> fnBook;

    /** Pointer to CSC Det Id function */
    std::function<bool(const unsigned int, const unsigned int, CSCDetId&)> fnGetCSCDetId;

    /** Parameter Getters */
    BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_GETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)

    /** Parameter Setters */
    BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_SETTER_MACRO, _, CONFIG_PARAMETERS_SEQ)

    /**
       * @brief  Constructor
       */
    Configuration() {
      init();
      printStatsLocal = 0;
    }

    /**
       * @brief  Constructor
       * @param printStats Print statistics on exit or not (overrides configuration parameter)
       */
    Configuration(const bool printStats) {
      init();
      if (printStats) {
        printStatsLocal = 1;
      } else {
        printStatsLocal = 2;
      }
    }

    /**
       * @brief  Destructor
       */
    ~Configuration() {
      if ((PRINT_STATS_ON_EXIT && printStatsLocal == 0) || printStatsLocal == 1) {
        printStats();
      }
    }

#ifdef DQMLOCAL

    /**
       * @brief  Load parameters from XML file (Local DQM)
       * @param  configFile Parameters file in XML format
       * @return 
       */
    void load(const std::string& configFile) {
      cms::concurrency::xercesInitialize();

      {
        XercesDOMParser parser;

        XMLFileErrorHandler eh;
        parser.setErrorHandler(&eh);

        parser.parse(configFile.c_str());
        DOMDocument* doc = parser.getDocument();
        DOMNode* docNode = (DOMNode*)doc->getDocumentElement();

        DOMNodeList* itemList = docNode->getChildNodes();
        for (XMLSize_t i = 0; i < itemList->getLength(); i++) {
          DOMNode* node = itemList->item(i);
          if (node->getNodeType() != DOMNode::ELEMENT_NODE) {
            continue;
          }

          std::string nodeName = XMLString::transcode(node->getNodeName());
          std::string value = XMLString::transcode(node->getTextContent());
          std::istringstream stm(value);

          BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_LOADXML_MACRO, _, CONFIG_PARAMETERS_SEQ)

          if (nodeName.compare("MO_FILTER") == 0) {
            DOMNodeList* filterList = node->getChildNodes();
            for (XMLSize_t j = 0; j < filterList->getLength(); j++) {
              DOMNode* filter = filterList->item(j);
              if (filter->getNodeType() != DOMNode::ELEMENT_NODE) {
                continue;
              }
              std::string filterName = XMLString::transcode(filter->getNodeName());
              std::string filterValue = XMLString::transcode(filter->getTextContent());
              MOFilterItems.insert(MOFilterItems.end(),
                                   MOFilterItem(filterValue, (filterName.compare("INCLUDE") == 0)));
            }
          }
        }
      }

      cms::concurrency::xercesTerminate();
    }

    /**
       * @brief  Print configuration in XML format
       * @param  config Configuration object to print
       * @return 
       */
    static void printXML(const Configuration& config) {
      cms::concurrency::xercesInitialize();

      DOMImplementation* domImpl = DOMImplementationRegistry::getDOMImplementation(XERCES_TRANSCODE("core"));
      DOMDocument* doc = domImpl->createDocument(0, XERCES_TRANSCODE("processor_configuration"), 0);
      DOMElement* rootElem = doc->getDocumentElement();

      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_PRINTXML_MACRO, _, CONFIG_PARAMETERS_SEQ)

      DOMLSSerializer* ser = domImpl->createLSSerializer();
      if (ser->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true)) {
        ser->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);
      }
      XMLFileErrorHandler eh;
      ser->setErrorHandler((DOMErrorHandler*)&eh);
      ser->writeNode(new StdOutFormatTarget(), *doc);

      doc->release();
      cms::concurrency::xercesTerminate();
    }

#endif

#ifdef DQMGLOBAL

    /**
       * @brief  Load parameters from ParameterSet (Global DQM)
       * @param  ps ParameterSet to load parameters from
       * @return 
       */
    void load(const edm::ParameterSet& ps) {
      BOOST_PP_SEQ_FOR_EACH_I(CONFIG_PARAMETER_LOADPS_MACRO, _, CONFIG_PARAMETERS_SEQ)
      std::vector<std::string> moFilter = ps.getUntrackedParameter<std::vector<std::string> >("MO_FILTER");
      for (std::vector<std::string>::iterator it = moFilter.begin(); it != moFilter.end(); it++) {
        std::string f = *it;
        if (!Utility::regexMatch("^[-+]/.*/$", f)) {
          LOG_WARN << "MO_FILTER item " << f << " does not recognized to be a valid one. Skipping...";
          continue;
        }
        bool include = Utility::regexMatch("^[+]", f);
        Utility::regexReplace("^./(.*)/$", f, "$1");
        MOFilterItems.insert(MOFilterItems.end(), MOFilterItem(f, include));
      }
    }

#endif

    /**
      * Statistics collection and printing section
      */

  private:
    /** Global Timer */
    boost::timer globalTimer;

    /** Event processing Timer */
    boost::timer eventTimer;

    /** Fractional MO update Timer */
    boost::timer fraTimer;

    /** Efficiency MO update Timer */
    boost::timer effTimer;

    /** Event processing time cummulative */
    double eventTimeSum;

    /** Fractional MO update time cummulative */
    double fraTimeSum;

    /** Efficiency MO update time cummulative */
    double effTimeSum;

  public:
/** Statistics field definition */
#define STATFIELD(caption, value, units)                                            \
  logger << std::setfill(' ');                                                      \
  logger << std::setiosflags(std::ios::right) << std::setw(25) << caption << " : "; \
  logger << std::setiosflags(std::ios::right) << std::setw(12);                     \
  if (((double)value) < 0) {                                                        \
    logger << "NA";                                                                 \
  } else {                                                                          \
    logger << value;                                                                \
  }                                                                                 \
  logger << std::setiosflags(std::ios::left) << std::setw(2) << units;              \
  logger << std::endl;

/** Statistics separator definition */
#define SEPFIELD                 \
  logger << std::setfill('-');   \
  logger << std::setw(25) << ""; \
  logger << std::setw(10) << ""; \
  logger << std::setw(2) << "";  \
  logger << std::endl;

    /**
       * @brief  Print Statistics on Exit (Destruction)
       * @return 
       */
    void printStats() {
      double allTime = globalTimer.elapsed();
      LogInfo logger;
      logger << std::endl;

      STATFIELD("Events processed", nEvents, "")
      STATFIELD("Bad events", nEventsBad, "")
      STATFIELD("Good events", nEventsGood, "")
      STATFIELD("CSC DCC events", nEventsCSC, "")
      STATFIELD("Unpacked CSCs", nUnpackedCSC, "")

      SEPFIELD

      STATFIELD("All event time", eventTimeSum, "s")
      double eventTimeAverage = (nEvents > 0 ? eventTimeSum / nEvents : -1.0);
      STATFIELD("Avg. event time", eventTimeAverage, "s")
      double eventRateAverage = (eventTimeSum > 0 ? nEvents / eventTimeSum : -1.0);
      STATFIELD("Avg. event rate", eventRateAverage, "Hz")
      double chamberRateAverage = (eventTimeSum > 0 ? nUnpackedCSC / eventTimeSum : -1.0);
      STATFIELD("Avg. chamber rate", chamberRateAverage, "Hz")

      SEPFIELD

      STATFIELD("All fra update time", fraTimeSum, "s")
      STATFIELD("All fra update count", fraCount, "")
      double fraTimeAverage = (fraCount > 0 ? fraTimeSum / fraCount : -1.0);
      STATFIELD("Avg. fra update time", fraTimeAverage, "s")

      SEPFIELD

      STATFIELD("All eff update time", effTimeSum, "s")
      STATFIELD("All eff update count", effCount, "")
      double effTimeAverage = (effCount > 0 ? effTimeSum / effCount : -1.0);
      STATFIELD("Avg. eff update time", effTimeAverage, "s")

      SEPFIELD

      STATFIELD("All time", allTime, "s")
      double allTimeAverage = (nEvents > 0 ? allTime / nEvents : -1.0);
      STATFIELD("Avg. event all time", allTimeAverage, "s")
      double allRateAverage = (allTime > 0 ? nEvents / allTime : -1.0);
      STATFIELD("Avg. event all rate", allRateAverage, "Hz")
      double chamberAllRateAverage = (allTime > 0 ? nUnpackedCSC / allTime : -1.0);
      STATFIELD("Avg. chamber all rate", chamberAllRateAverage, "Hz")
    }

#undef STATFIELD
#undef SEPFIELD

    /**
       * @brief  Switch on/off event processing timer
       * @param start timer action (true - start, false - stop) 
       * @return 
       */
    void eventProcessTimer(const bool start) {
      if (start) {
        eventTimer.restart();
      } else {
        eventTimeSum += eventTimer.elapsed();
      }
    }

    /**
       * @brief  Switch on/off fractional MO processing timer
       * @param start timer action (true - start, false - stop) 
       * @return 
       */
    void updateFraTimer(const bool start) {
      if (start) {
        fraTimer.restart();
      } else {
        fraTimeSum += fraTimer.elapsed();
        fraCount++;
      }
    }

    /**
       * @brief  Switch on/off efficiency MO processing timer
       * @param start timer action (true - start, false - stop) 
       * @return 
       */
    void updateEffTimer(const bool start) {
      if (start) {
        effTimer.restart();
      } else {
        effTimeSum += effTimer.elapsed();
        effCount++;
      }
    }

    /**
       * @brief  Check if MO is not excluded by MO Filter
       * @param  name MO name to book
       * @return true if MO is not excluded, false - otherwise
       */
    const bool needBookMO(const std::string name) const {
      bool result = true;
      for (unsigned int i = 0; i < MOFilterItems.size(); i++) {
        const MOFilterItem* filter = &MOFilterItems.at(i);
        if (Utility::regexMatch(filter->pattern, name))
          result = filter->include;
      }
      return result;
    }

    /**
      * Counters section.
      */

  public:
    /**
       * @brief  Reset counters
       * @return 
       */
    void reset() {
      nEvents = 0;
      nEventsBad = 0;
      nEventsGood = 0;
      nEventsCSC = 0;
      nUnpackedCSC = 0;
      fraCount = 0;
      effCount = 0;
      eventTimeSum = 0.0;
      fraTimeSum = 0.0;
      effTimeSum = 0.0;
    }

    /**
       * Getters for Global Counters.
       */

    const unsigned long getNEvents() const { return nEvents; }
    const unsigned long getNEventsBad() const { return nEventsBad; }
    const unsigned long getNEventsGood() const { return nEventsGood; }
    const unsigned long getNEventsCSC() const { return nEventsCSC; }
    const unsigned long getNUnpackedCSC() const { return nUnpackedCSC; }

    /**
       * Increments (by 1) for Global Counters.
       */

    void incNEvents() {
      nEvents++;
      if (getEVENTS_ECHO() > 0) {
        if (getNEvents() % getEVENTS_ECHO() == 0) {
          LOG_INFO << "(echo) Events processed: " << std::setw(12) << getNEvents();
        }
      }
    }
    void incNEventsBad() { nEventsBad++; }
    void incNEventsGood() { nEventsGood++; }
    void incNEventsCSC() { nEventsCSC++; }
    void incNUnpackedCSC() { nUnpackedCSC++; }

    /**
       * @brief  Increment Chamber counter by 1
       * @param  counter Counter Type
       * @param  crateId CSC Crate ID
       * @param  dmbId CSC DMB ID
       * @return 
       */
    void incChamberCounter(const ChamberCounterType counter, const HwId crateId, const HwId dmbId) {
      setChamberCounterValue(counter, crateId, dmbId, getChamberCounterValue(counter, crateId, dmbId) + 1);
    }

    /**
       * @brief  Set Chamber counter value
       * @param  counter Counter Type
       * @param  crateId CSC Crate ID
       * @param  dmbId CSC DMB ID
       * @param value value to set
       * @return 
       */
    void setChamberCounterValue(const ChamberCounterType counter,
                                const HwId crateId,
                                const HwId dmbId,
                                const uint32_t value) {
      ChamberMapCounterMapType::iterator it = chamberCounters.find(boost::make_tuple(crateId, dmbId));
      if (it == chamberCounters.end()) {
        it = chamberCounters.insert(chamberCounters.end(),
                                    ChamberCounterKeyType(crateId, dmbId, ChamberCounterMapType()));
      }
      ChamberCounterMapType* cs = const_cast<ChamberCounterMapType*>(&it->counters);
      ChamberCounterMapType::iterator itc = cs->find(counter);
      if (itc == cs->end()) {
        cs->insert(std::make_pair(counter, value));
      } else {
        itc->second = value;
      }
    }

    /**
       * @brief  Copy Chamber counter value from one counter to another
       * @param  counter_from Counter Type to copy value from
       * @param  counter_to Counter Type to copy value to
       * @param  crateId CSC Crate ID
       * @param  dmbId CSC DMB ID
       * @return 
       */
    void copyChamberCounterValue(const ChamberCounterType counter_from,
                                 const ChamberCounterType counter_to,
                                 const HwId crateId,
                                 const HwId dmbId) {
      setChamberCounterValue(counter_from, crateId, dmbId, getChamberCounterValue(counter_from, crateId, dmbId));
    }

    /**
       * @brief  Get Chamber counter value
       * @param  counter Counter Type
       * @param  crateId CSC Crate ID
       * @param  dmbId CSC DMB ID
       * @return current counter value
       */
    const uint32_t getChamberCounterValue(const ChamberCounterType counter,
                                          const HwId crateId,
                                          const HwId dmbId) const {
      ChamberMapCounterMapType::iterator it = chamberCounters.find(boost::make_tuple(crateId, dmbId));
      if (it == chamberCounters.end())
        return 0;
      ChamberCounterMapType::const_iterator itc = it->counters.find(counter);
      if (itc == it->counters.end())
        return 0;
      return itc->second;
    }

  private:
    /**
       * Global Counters.
       */

    /** Number of events */
    unsigned long nEvents;

    /** Number of bad events */
    unsigned long nEventsBad;

    /** Number of good events */
    unsigned long nEventsGood;

    /** Number of events that have CSC data (used in Global DQM) */
    unsigned long nEventsCSC;

    /** number of unpacked chambers */
    unsigned long nUnpackedCSC;

    /** Number of Fractional MO updates */
    unsigned long fraCount;

    /** Number of Efficiency MO updates */
    unsigned long effCount;

    /** Map of chamber counters */
    ChamberMapCounterMapType chamberCounters;
  };

}  // namespace cscdqm

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
