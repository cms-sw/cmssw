#ifndef CommonTools_Utils_Exception_h
#define CommonTools_Utils_Exception_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     Exception
// 
/**\class Exception Exception.h CommonTools/Utils/interface/Exception.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug 14 17:09:42 EDT 2008
// $Id: Exception.h,v 1.3 2009/03/17 09:38:54 llista Exp $
//

// system include files
#include <sstream>
#include "boost/spirit/include/classic_exceptions.hpp"

// user include files

// forward declarations
namespace reco {
   namespace parser {
      enum SyntaxErrors {
         kSyntaxError,
         kMissingClosingParenthesis,
         kSpecialError
      };
      
      typedef boost::spirit::classic::parser_error<reco::parser::SyntaxErrors> BaseException;
      
      ///returns the appropriate 'what' message for the exception
      inline const char * baseExceptionWhat(const BaseException& e) {
         switch(e.descriptor) {
            case kMissingClosingParenthesis:
            return "Missing close parenthesis.";
            case kSyntaxError:
            return "Syntax error.";
            case kSpecialError:
            default:
               break;
         }
         return e.what();
      }
      class Exception : public BaseException {

      public:
         Exception(const char* iIterator) : BaseException(iIterator,kSpecialError) {}
         Exception(const Exception& iOther) : BaseException(iOther) {
            ost_ << iOther.ost_.str();
         }
         ~Exception() throw() {}

         // ---------- const member functions ---------------------
         const char* what() const throw() { what_ = ost_.str(); return what_.c_str();}

         // ---------- static member functions --------------------

         // ---------- member functions ---------------------------

         template<class T>
         friend Exception& operator<<(Exception&, const T&);
         template<class T>
         friend Exception& operator<<(const Exception&, const T&);
         friend Exception& operator<<(Exception&, std::ostream&(*f)(std::ostream&));
         friend Exception& operator<<(const Exception&, std::ostream&(*f)(std::ostream&));
         friend Exception& operator<<(Exception&, std::ios_base&(*f)(std::ios_base&));
         friend Exception& operator<<(const Exception&, std::ios_base&(*f)(std::ios_base&));
      private:

         //const Exception& operator=(const Exception&); // stop default

         // ---------- member data --------------------------------
         std::ostringstream ost_;
         mutable std::string what_; //needed since ost_.str() returns a temporary string
      };
      
      template<class T>
       inline Exception& operator<<(Exception& e, const T& iT) {
         e.ost_ << iT;
         return e;
      }
      template<class T>
      inline Exception& operator<<(const Exception& e, const T& iT) {
         return operator<<(const_cast<Exception&>(e), iT);
      }
      inline Exception& operator<<(Exception& e, std::ostream&(*f)(std::ostream&))
      {
         f(e.ost_);
         return e;
      }
      inline Exception& operator<<(const Exception& e, std::ostream&(*f)(std::ostream&))
      {
         f(const_cast<Exception&>(e).ost_);
         return const_cast<Exception&>(e);
      }
      inline Exception& operator<<(Exception& e, std::ios_base&(*f)(std::ios_base&)) {
         f(e.ost_);
         return e;
      }      
      inline Exception& operator<<(const Exception& e, std::ios_base&(*f)(std::ios_base&)) {
         f(const_cast<Exception&>(e).ost_);
         return const_cast<Exception&>(e);
      }
   }
}

#endif
