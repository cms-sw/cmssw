/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Collection.h
 *
 *    Description:  Histogram Collection management class
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
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <boost/shared_ptr.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_Exception.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_HistoProviderIf.h"

namespace cscdqm {

  using namespace XERCES_CPP_NAMESPACE;

  static const char XML_BOOK_DEFINITION[]     =  "Definition";
  static const char XML_BOOK_DEFINITION_ID[]  =  "id";
  static const char XML_BOOK_HISTOGRAM[]      =  "Histogram";
  static const char XML_BOOK_DEFINITION_REF[] =  "ref";
  static const char XML_BOOK_HISTO_NAME[]     =  "Name";
  static const char XML_BOOK_HISTO_PREFIX[]   =  "Prefix";
  static const char XML_BOOK_HISTO_TYPE[]     =  "Type";
  static const char XML_BOOK_HISTO_TITLE[]    =  "Title";

  static const int  DEF_HISTO_COLOR           =  48;

  /**
  * Type Definition Section
  */
  typedef std::map<std::string, std::string>     CoHistoProps;
  typedef std::map<std::string, CoHistoProps>    CoHisto;
  typedef std::map<std::string, CoHisto>         CoHistoMap;
  
  /**
  * @brief  Converting from string to whatever number (failsafe!) 
  * @param  t result number
  * @param  s source string
  * @param  f base
  * @return true if success, else - false
  */
  template <class T>
  bool stringToNumber(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&)) {
    std::istringstream iss(s);
    return !(iss >> f >> t).fail();
  }

  class Collection {

    public:

      Collection(HistoProvider* p_histoProvider);
      const CoHistoMap& get() const { return collection; };
      void load(const std::string p_bookingFile);
      void book(const CoHisto& hs) const;
      void book(const CoHistoProps& h) const;
      void book(const std::string& prefix) const;

      void printCollection() const;

      static void getCSCTypeToBinMap(std::map<std::string, int>& tmap);
      static const std::string getCSCTypeLabel(int endcap, int station, int ring);
      static const int tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ");
      static void splitString(std::string str, const std::string delim, std::vector<std::string>& results);
      static void trimString(std::string& str);

    private:
      
      static const bool checkHistoValue(const CoHistoProps& h, const std::string& name, std::string& value);
      static const bool checkHistoValue(const CoHistoProps& h, const std::string& name, int& value);
      static const bool checkHistoValue(const CoHistoProps& h, const std::string name, double& value);

      static std::string& getHistoValue(const CoHistoProps& h, const std::string& name, std::string& value, const std::string& def_value = "");
      static int&         getHistoValue(const CoHistoProps& h, const std::string& name, int& value, const int& def_value = 0);
      static double&      getHistoValue(const CoHistoProps& h, const std::string name, double& value, const int def_value = 0.0);
      
      static const int ParseAxisLabels(const std::string& s, std::map<int, std::string>& labels);

      static void getNodeProperties(DOMNode*& node, CoHistoProps& hp);
      
      HistoProvider* histoProvider;
      CoHistoMap     collection;

  };

  class BookingFileErrorHandler : public ErrorHandler {

    public:

      void warning(const SAXParseException& exc) {
        char* message = XMLString::transcode(exc.getMessage());
        LOG_WARN << "Booking File: " << message << ". line: " << exc.getLineNumber() << " col: " << exc.getColumnNumber();
        XMLString::release(&message);
      }

      void error(const SAXParseException& exc) {
        this->fatalError(exc);
      }

      void fatalError(const SAXParseException& exc) {
        char* message = XMLString::transcode(exc.getMessage());
        LOG_ERROR << "Booking File: " << message << ". line: " << exc.getLineNumber() << " col: " << exc.getColumnNumber();
        throw Exception(message);
      }

      void resetErrors () { }

  };

}

#endif
