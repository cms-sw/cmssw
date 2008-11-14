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

namespace cscdqm {

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

}

#endif
