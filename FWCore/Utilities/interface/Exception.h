#ifndef FWK_CMS_EXCEPTION_HH
#define FWK_CMS_EXCEPTION_HH

/**
  This a basic exception type expected to be thown by
  developer-written code.  We recommend that you use it directly.
  It can also be used as a base class if needed

  Each Exception is identified by a category string.  This is category
  is a short word (no spaces) that described the problem that was
  encountered.  These category identifiers can be concatenated if
  exceptions are and caught and rethrown and the entire list can be
  accessed using the history() call.

  Example:
    try { func(); }
    catch ( cms::Exception& e )
    { throw cms::Exception("DataCorrupt","encountered while unpacking",e); }

  Information can be added to the Exception using the stream insertion
  operator (as one uses cout).  We recommend using it in the following
  manner:

  Example:
    if ( (rc=func()) < 0 )
    {
      throw cms::Exception("DataCorrupt") << "I died with rc = " << rc << endl;
    }

  Derived types are expected to fix the category by passing a string
  literal to the base constructor or force the developer to give a
  category name.

  Example:
    class InfiniteLoop : public Exception
    {
       InfiniteLoop(const std::string& msg) : Exception("InfiniteLoop",msg) { }
    };

  The information from what() has a simple format that makes it easy
  to locate the reason for and context of the error and separate it
  from the user-supplied free-format text.  The what() contains all
  the category and context information in a nested hierarchical
  format.

  Example:
    ---- InfiniteLoop BEGIN
    -- who: moduletype=PixelUnpacker modulelabel=unpackthing
    -- where: event=122234553.1233123456 runsegment=3 store=446
    I am really sad about this
    ---- DataCorrupt BEGIN
    Unpacking of  pixel detector region 14 failed to get valid cell ID
    ---- DataCorrupt END
    -- action: skip event
    ---- InfiniteLoop BEGIN

  Fixed format Framework supplied context information will be
  specially tagged.  See the framework error section of the roadmap
  for details on the format and tags of framework suppied information.
 **/

#include "SealBase/Error.h"

#include <string>
#include <sstream>
#include <list>

namespace cms
{
  class Exception : public seal::Error
  {
  public:
    typedef std::string Category;
    typedef std::list<Category> CategoryList;

    explicit Exception(const Category& category);
    Exception(const Category& category,
	      const std::string& message);
    Exception(const Category& category,
	      const std::string& message,
	      const Exception& another);
    Exception(const Exception& other);
    virtual ~Exception();

    std::string what() const;
    std::string category() const;

    const CategoryList& history() const;

    void append(const Exception& another);
    void append(const std::string& more_information);
    void append(const char* more_information);

    template <class T> Exception& operator<<(const T& stuff);
    template <class T> Exception& operator<<(T& stuff);
    Exception& operator<<(std::ostream& (*f)(std::ostream&));
    Exception& operator<<(std::ios_base& (*f)(std::ios_base&));
    Exception& operator<<(const char*);
    Exception& operator<<(char*);

  private:

    // the following are required by seal::Error
    virtual std::string explainSelf() const;
    virtual seal::Error* clone() const;
    virtual void rethrow();

    // data members
    std::ostringstream ost_;
    CategoryList category_;
  };

  inline std::ostream& operator<<(std::ostream& ost, const Exception& e)
  {
    ost << e.what();
    return ost;
  }


  // -------- implementation ---------

  template <class T>
    inline Exception& Exception::operator<<(const T& stuff)
  {
    ost_ << stuff;
    return *this;
  }

  template <class T>
    inline Exception& Exception::operator<<(T& stuff)
  {
    ost_ << stuff;
    return *this;
  }

  inline Exception& Exception::operator<<(std::ostream& (*f)(std::ostream&))
  {
    f(ost_);
    return *this;
  }

  inline Exception& Exception::operator<<(std::ios_base& (*f)(std::ios_base&))
  {
    f(ost_);
    return *this;
  }

  inline Exception& Exception::operator<<(const char* c)
  {
    ost_ << c;
    return *this;
  }

  inline Exception& Exception::operator<<(char* c)
  {
    ost_ << c;
    return *this;
  }

}

#endif
