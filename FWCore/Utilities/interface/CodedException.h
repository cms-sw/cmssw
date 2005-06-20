#ifndef EDM_CODEDEXCEPTION_HH
#define EDM_CODEEXCEPTION_HH

/**

This is the basic exception that is thrown by the framework code.
It exists primarily to distinguish framework thrown exception types
from developer thrown exception types. As such there is very little
interface other than constructors specific to this derived type.

**/

#include "FWCore/FWUtilities/interface/Exception.h"

#include <string>
#include <map>

namespace edm
{
  template <class Code>
  class CodedException : public cms::Exception
  {
  public:
    explicit CodedException(Code category);

    CodedException(Code category,
		   const std::string& message);

    CodedException(Code category,
		   const std::string& message,
		   const CodedException& another);

    CodedException(const CodedException& other);

    virtual ~CodedException();

    Code categoryCode() const { return category_; }

    static std::string codeToString(Code);

    typedef std::map<Code,std::string> CodeMap; 
  private:

    friend struct TableLoader;
    struct TableLoader
    { TableLoader() { CodedException<Code>::loadtable(); } };

    struct Entry
    {
      int code;
      char* name;
    };

    Code category_;
    static CodeMap trans_;
    static TableLoader loader_;
    static void loadTable();
  };

  template <class Code>
  std::ostream& operator<<(std::ostream& ost, const CodedException<Code>& c)
  {
    ost << CodedException<Code>::codeToString(c.code());
    return ost;
  }

  /// -------------- implementation details ------------------

  template <class Code>
  typename CodedException<Code>::CodeMap CodedException<Code>::trans_;
  template <class Code>
  typename CodedException<Code>::TableLoader CodedException<Code>::loader_;

  template <class Code>
  std::string CodedException<Code>::codeToString(Code c)
  {
    typename CodeMap::const_iterator i(trans_.find(c));
    return i!=trans_.end() ? i->second : std::string("UnknownCode");
  }


  template <class Code>
  CodedException<Code>::CodedException(Code category):
    cms::Exception(Code::codeToString(category))
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(Code category,
				       const std::string& message):
    cms::Exception(Code::codeToString(category),message)
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(Code category,
				       const std::string& message,
				       const CodedException& another):
    cms::Exception(Code::codeToString(category),message,another)
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(const CodedException& other):
    cms::Exception(other)
  {
  }

  template <class Code>
  CodedException<Code>::~CodedException()
  {
  }

}

#endif
