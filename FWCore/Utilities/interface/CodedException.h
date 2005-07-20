#ifndef EDM_CODEDEXCEPTION_HH
#define EDM_CODEEXCEPTION_HH

/**

This is the basic exception that is thrown by the framework code.
It exists primarily to distinguish framework thrown exception types
from developer thrown exception types. As such there is very little
interface other than constructors specific to this derived type.

**/

#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <map>

#define EDM_MAP_ENTRY(ns, name) trans_[ns::name]=#name

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
		   const cms::Exception& another);

    CodedException(const CodedException& other);

    virtual ~CodedException();

    Code categoryCode() const { return category_; }

    static std::string codeToString(Code);

    typedef std::map<Code,std::string> CodeMap; 
  private:

    friend struct TableLoader;
    struct TableLoader
    {
      TableLoader()
      { CodedException<Code>::loadTable(); c_=&trans_; }
      CodeMap* c_;
    };

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

#if 0
  template <class Code>
  void CodedException<Code>::loadTable() { }
#endif

  template <class Code>
  std::string CodedException<Code>::codeToString(Code c)
  {
    CodeMap* trans = loader_.c_;
    typename CodeMap::const_iterator i(trans->find(c));
    return i!=trans->end() ? i->second : std::string("UnknownCode");
  }


  template <class Code>
  CodedException<Code>::CodedException(Code category):
    cms::Exception(codeToString(category)),
    category_(category)
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(Code category,
				       const std::string& message):
    cms::Exception(codeToString(category),message),
    category_(category)
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(Code category,
				       const std::string& message,
				       const cms::Exception& another):
    cms::Exception(codeToString(category),message,another),
    category_(category)
  {
  }

  template <class Code>
  CodedException<Code>::CodedException(const CodedException& other):
    cms::Exception(other),
    category_(other.category_)
  {
  }

  template <class Code>
  CodedException<Code>::~CodedException()
  {
  }

}

#endif
