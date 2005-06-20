#ifndef EDM_EDMEXCEPTION_HH
#define EDM_EDMEXCEPTION_HH

/**

 This is the basic exception that is thrown by the framework code.
 It exists primarily to distinguish framework thrown exception types
 from developer thrown exception types. As such there is very little
 interface other than constructors specific to this derived type.

**/

#include "FWCore/FWUtilities/interface/Exception.h"

#include <string>

namespace edm
{
  template <class Code>
  class CodedException : public cms::Exception
  {
  public:
    explicit CodedException(Code category)
    CodedException(Code category,
	      const std::string& message);
    CodedException(Code category,
	      const std::string& message,
	      const Exception& another);
    CodedException(const Exception& other);
    virtual ~CodedException();

    Code categoryCode() const { return category_; }

  private:
    Code category_;
  };


  template <class CodeType>
  class Codes
  {
  public:

    Codes(): code_() { } 
    explicit  Codes(ErrorCodes i):
      code_(i) { /* should we validate the code? */ } 
    int code() const { return code_; }

    static std::string codeToString(int code);
  private:
    int code_;
  };

  template <class Code>
  std::ostream& operator<<(std::ostream& ost, const CodedException<Code>& c)
  {
    ost << CodedException<Code>::codeToString(c.code());
    return ost;
  }

  /// -------------- implementation details ------------------

  template <class Code>
  Exception<Code>::Exception(Code category):
    cms::Exception(Code::codeToString(category))
  {
  }

  template <class Code>
  Exception<Code>::Exception(Code category,
			     const std::string& message):
    cms::Exception(Code::codeToString(category),message)
  {
  }

  template <class Code>
  Exception<Code>::Exception(Code category,
			     const std::string& message,
			     const Exception& another):
    cms::Exception(Code::codeToString(category),message,another)
  {
  }

  template <class Code>
  Exception<Code>::Exception(const Exception& other):
    cms::Exception(other)
  {
  }

  template <class Code>
  Exception<Code>::~Exception()
  {
  }

}

#endif
