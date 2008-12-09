/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_Exception.h
 *
 *    Description:  Custom Exception
 *
 *        Version:  1.0
 *        Created:  11/14/2008 11:51:31 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCDQM_Exception_H
#define CSCDQM_Exception_H

#include <string>
#include <exception>

#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>

#include "DQM/CSCMonitorModule/interface/CSCDQM_Logger.h"

namespace cscdqm {

  using namespace XERCES_CPP_NAMESPACE;

  /**
   * @class Exception
   * @brief Application level Exception that is used to cut-off application
   * execution in various cases.
   */
  class Exception: public std::exception {
    private:

      std::string message;

    public:

      Exception(const std::string& message) throw() {
        this->message = message;
      }

      virtual ~Exception() throw() { }

      virtual const char* what() const throw() {
        return message.c_str();
      }

  };

  /**
   * @class XMLFileErrorHandler
   * @brief Takes care of errors and warnings while parsing XML files
   * file in XML format.
   */
  class XMLFileErrorHandler : public ErrorHandler {

    public:

      void warning(const SAXParseException& exc) {
        char* message = XMLString::transcode(exc.getMessage());
        LOG_WARN << "File: " << message << ". line: " << exc.getLineNumber() << " col: " << exc.getColumnNumber();
        XMLString::release(&message);
      }

      void error(const SAXParseException& exc) {
        this->fatalError(exc);
      }

      void fatalError(const SAXParseException& exc) {
        char* message = XMLString::transcode(exc.getMessage());
        LOG_ERROR << "File: " << message << ". line: " << exc.getLineNumber() << " col: " << exc.getColumnNumber();
        throw Exception(message);
      }

      void resetErrors () { }

  };

}

#endif
