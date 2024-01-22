#ifndef FWCore_Utilities_Exception_h
#define FWCore_Utilities_Exception_h

/**
   This a basic exception type expected to be thrown by
   developer-written code.  We recommend that you use it directly.
   It can also be used as a base class if needed.

   Each Exception is identified by a category string.  This category
   is a short word or phrase (no spaces) that described the problem
   that was encountered.

   Information can be added to the Exception using the stream insertion
   operator (as one uses cout).  We recommend using it in the following
   manner:

   Example:
   if ((rc=func()) < 0)
   {
   throw cms::Exception("DataCorrupt") << "I died with rc = " 
     << rc << std::endl;
   }

   Derived types are expected to fix the category, either by
   1) passing a string literal to the base class constructor, or
   2) ensuring the developer gives a category name.

   Example:
   class InfiniteLoop : public Exception
   {
   InfiniteLoop(const std::string& msg) : Exception("InfiniteLoop",msg) { }
   };
**/

#include <atomic>
#include <list>
#include <sstream>
#include <string>
#include <exception>
#include <type_traits>

#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/Visibility.h"

namespace cms {

  class Exception;
  namespace detail {
    // Used for SFINAE to control the instantiation of the stream insertion
    // member template needed to support streaming output to an object
    // of type cms::Exception, or a subclass of cms::Exception.
    template <typename E>
    using exception_type =
        std::enable_if_t<std::is_base_of_v<Exception, std::remove_reference_t<E>>, std::remove_reference_t<E>>;

  }  // namespace detail

  class dso_export Exception : public std::exception {
  public:
    explicit Exception(std::string const& aCategory);
    explicit Exception(char const* aCategory);

    Exception(std::string const& aCategory, std::string const& message);
    Exception(char const* aCategory, std::string const& message);
    Exception(std::string const& aCategory, char const* message);
    Exception(char const* aCategory, char const* message);

    Exception(std::string const& aCategory, std::string const& message, Exception const& another);

    Exception(Exception const& other);
    ~Exception() noexcept override;

    void swap(Exception& other);

    Exception& operator=(Exception const& other);

    // The signature for what() must be identical to that of std::exception::what().
    // This function is NOT const thread safe
    char const* what() const noexcept override;

    virtual std::string explainSelf() const;

    std::string const& category() const;
    std::string message() const;
    std::list<std::string> const& context() const;
    std::list<std::string> const& additionalInfo() const;
    int returnCode() const;

    void raise() { rethrow(); }

    void append(Exception const& another);
    void append(std::string const& more_information);
    void append(char const* more_information);

    void clearMessage();
    void clearContext();
    void clearAdditionalInfo();

    void addContext(std::string const& context);
    void addContext(char const* context);

    void addAdditionalInfo(std::string const& info);
    void addAdditionalInfo(char const* info);

    void setContext(std::list<std::string> const& context);
    void setAdditionalInfo(std::list<std::string> const& info);

    bool alreadyPrinted() const;
    void setAlreadyPrinted();

    virtual Exception* clone() const;

    // In the following templates, class E is our Exception class or
    // any subclass thereof. The complicated return type exists to
    // make sure the template can only be instantiated for such
    // classes.
    //

    template <typename E, typename T>
    friend typename detail::exception_type<E>& operator<<(E&& e, T const& stuff);

    template <typename E>
    friend typename detail::exception_type<E>& operator<<(E&& e, std::ostream& (*f)(std::ostream&));

    template <typename E>
    friend typename detail::exception_type<E>& operator<<(E&& e, std::ios_base& (*f)(std::ios_base&));

    // This function is deprecated and we are in the process of removing
    // all code that uses it from CMSSW.  It will then be deleted.
    std::list<std::string> history() const;

  private:
    void init(std::string const& message);
    virtual void rethrow();
    virtual int returnCode_() const;

    // data members
    std::ostringstream ost_;
    std::string category_;
    //The exception class should not be accessed across threads
    CMS_SA_ALLOW mutable std::string what_;
    std::list<std::string> context_;
    std::list<std::string> additionalInfo_;
    std::atomic<bool> alreadyPrinted_;
  };

  inline std::ostream& operator<<(std::ostream& ost, Exception const& e) {
    ost << e.explainSelf();
    return ost;
  }

  // -------- implementation ---------

  template <typename E, typename T>
  inline typename detail::exception_type<E>& operator<<(E&& e, T const& stuff) {
    e.ost_ << stuff;
    return e;
  }

  template <typename E>
  inline typename detail::exception_type<E>& operator<<(E&& e, std::ostream& (*f)(std::ostream&)) {
    f(e.ost_);
    return e;
  }

  template <typename E>
  inline typename detail::exception_type<E>& operator<<(E&& e, std::ios_base& (*f)(std::ios_base&)) {
    f(e.ost_);
    return e;
  }

}  // namespace cms

#endif
