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

#include <list>
#include <sstream>
#include <string>
#include <exception>
#include <type_traits>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace cms {

  namespace detail {
    // The struct template Desired exists in order to allow us to use
    // SFINAE to control the instantiation of the stream insertion
    // member template needed to support streaming output to an object
    // of type cms::Exception, or a subclass of cms::Exception.

    template <typename T, bool b>
    struct Desired;
    template <typename T>
    struct Desired<T, true> {
      typedef T type;
    };

    // The following struct template is a metafunction which combines
    // two of the boost type_traits metafunctions.

    template <typename BASE, typename DERIVED>
    struct is_derived_or_same {
      static bool const value = std::is_base_of<BASE, DERIVED>::value || std::is_same<BASE, DERIVED>::value;
    };

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
    void setAlreadyPrinted(bool value);

    virtual Exception* clone() const;

    // In the following templates, class E is our Exception class or
    // any subclass thereof. The complicated return type exists to
    // make sure the template can only be instantiated for such
    // classes.
    //

    // We have provided versions of these templates which accept, and
    // return, reference-to-const objects of type E. These are needed
    // so that the expression:
    //
    //     throw Exception("category") << "some message";
    //
    // shall compile. The technical reason is that the un-named
    // temporary created by the explicit call to the constructor
    // allows only const access, except by member functions. It seems
    // extremely unlikely that any Exception object will be created in
    // read-only memory; thus it seems unlikely that allowing the
    // stream operators to write into a nominally 'const' Exception
    // object is a real danger.

    template <typename E, typename T>
    friend typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
    operator<<(E&& e, T const& stuff);

    template <typename E>
    friend typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
    operator<<(E&& e, std::ostream& (*f)(std::ostream&));

    template <typename E>
    friend typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
    operator<<(E&& e, std::ios_base& (*f)(std::ios_base&));

    // The following two function templates should be included, to help
    // reduce the number of function templates instantiated. However,
    // GCC 3.2.3 crashes with an internal compiler error when
    // instantiating these functions.

    //     template <typename E>
    //     friend
    //     typename detail::Desired<E, (std::is_base_of<Exception,E>::value ||
    // 				 std::is_same<Exception,E>::value)>::type &
    //     operator<<(E& e, const char*);

    //    template <typename E>
    //    friend
    //    typename detail::Desired<E, (std::is_base_of<Exception,E>::value ||
    //  			       std::is_same<Exception,E>::value)>::type &
    //  			       operator<<(E& e, char*);

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
    bool alreadyPrinted_;
  };

  inline std::ostream& operator<<(std::ostream& ost, Exception const& e) {
    ost << e.explainSelf();
    return ost;
  }

  // -------- implementation ---------

  template <typename E, typename T>
  inline typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
  operator<<(E&& e, T const& stuff) {
    e.ost_ << stuff;
    return e;
  }

  template <typename E>
  inline typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
  operator<<(E&& e, std::ostream& (*f)(std::ostream&)) {
    f(e.ost_);
    return e;
  }

  template <typename E>
  inline typename detail::Desired<E, detail::is_derived_or_same<Exception, std::remove_reference_t<E>>::value>::type&
  operator<<(E&& e, std::ios_base& (*f)(std::ios_base&)) {
    f(e.ost_);
    return e;
  }

  // The following four function templates should be included, to help
  // reduce the number of function templates instantiated. However,
  // GCC 3.2.3 crashes with an internal compiler error when
  // instantiating these functions.

  // template <typename E>
  // inline
  // typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
  // operator<<(E& e, char const* c)
  // {
  //   e.ost_ << c;
  //   return e;
  // }

  // template <typename E>
  // inline
  // typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
  // operator<<(E const& e, char const* c)
  // {
  //   E& ref = const_cast<E&>(e);
  //   ref.ost_ << c;
  //   return e;
  // }

  //  template <typename E>
  //  inline
  //  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
  //  operator<<(E& e, char* c)
  //  {
  //    e.ost_ << c;
  //    return e;
  //  }

  //  template <typename E>
  //  inline
  //  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
  //  operator<<(E const& e, char* c)
  //  {
  //    E& ref = const_cast<E&>(e);
  //    ref.ost_ << c;
  //    return e;
  //  }
}  // namespace cms

#endif
