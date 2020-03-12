#ifndef NPSTAT_EXCEPTION_HH_
#define NPSTAT_EXCEPTION_HH_

/*!
// \file NpstatException.h
//
// \brief Exceptions for the npstat namespace
//
// Author: I. Volobouev
//
// November 2012
*/

#include <string>

#include "FWCore/Utilities/interface/Exception.h"

namespace npstat {
  /** Base class for the exceptions specific to the npstat namespace */
  struct NpstatException : public cms::Exception {
    inline NpstatException() : cms::Exception("npstat::NpstatException") {}

    inline explicit NpstatException(const std::string& description) : cms::Exception(description) {}

    inline explicit NpstatException(const char* description) : cms::Exception(description) {}

    ~NpstatException() throw() override {}
  };

  struct NpstatOutOfRange : public NpstatException {
    inline NpstatOutOfRange() : NpstatException("npstat::NpstatOutOfRange") {}

    inline explicit NpstatOutOfRange(const std::string& description) : NpstatException(description) {}

    ~NpstatOutOfRange() throw() override {}
  };

  struct NpstatInvalidArgument : public NpstatException {
    inline NpstatInvalidArgument() : NpstatException("npstat::NpstatInvalidArgument") {}

    inline explicit NpstatInvalidArgument(const std::string& description) : NpstatException(description) {}

    ~NpstatInvalidArgument() throw() override {}
  };

  struct NpstatRuntimeError : public NpstatException {
    inline NpstatRuntimeError() : NpstatException("npstat::NpstatRuntimeError") {}

    inline explicit NpstatRuntimeError(const std::string& description) : NpstatException(description) {}

    ~NpstatRuntimeError() throw() override {}
  };

  struct NpstatDomainError : public NpstatException {
    inline NpstatDomainError() : NpstatException("npstat::NpstatDomainError") {}

    inline explicit NpstatDomainError(const std::string& description) : NpstatException(description) {}

    ~NpstatDomainError() throw() override {}
  };
}  // namespace npstat

#endif  // NPSTAT_EXCEPTION_HH_
