#ifndef FWCore_Utilities_Exception_h
#define FWCore_Utilities_Exception_h

/**
   This a basic exception type expected to be thrown by
   developer-written code.  We recommend that you use it directly.
   It can also be used as a base class if needed.

   Each Exception is identified by a category string.  This category
   is a short word or phrase (no spaces) that described the problem
   that was encountered.  These category identifiers can be
   concatenated if exceptions are caught and rethrown and the
   entire list can be accessed using the history() call.

   Example:
   try { func(); }
   catch (cms::Exception& e)
   { throw cms::Exception("DataCorrupt","encountered while unpacking",e); }

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

   The output from what() has a simple format that makes it easy to
   locate the reason for and context of the error and separate it from
   the user-supplied free-format text.  The output from what() contains
   all the category and context information in a nested hierarchical
   format.

   The following 'example' shows one possible output. Please note that,
   in this early version, the format is still changing; what you
   actually obtain should be similar to, but not necessarily identical
   with, the following:

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
   for details on the format and tags of framework supplied information.


   TODO: Update the example to match the final formatting, when that
   formatting has settled down.

**/

#include <list>
#include <sstream>
#include <string>
#include <exception>

#include "boost/type_traits/is_base_and_derived.hpp"

namespace cms {

  namespace detail {
    // The struct template Desired exists in order to allow us to use
    // SFINAE to control the instantiation of the stream insertion
    // member template needed to support streaming output to an object
    // of type cms::Exception, or a subclass of cms::Exception.

    template <typename T, bool b> struct Desired;
    template <typename T> struct Desired<T, true> { typedef T type; };


    // The following struct template is a metafunction which combines
    // two of the boost type_traits metafunctions.

    template <typename BASE, typename DERIVED>
    struct is_derived_or_same {
      static bool const value = 
	boost::is_base_and_derived<BASE,DERIVED>::value || boost::is_same<BASE,DERIVED>::value;
    };

  }


  class Exception : public std::exception {
  public:
    typedef std::string Category;
    typedef std::list<Category> CategoryList;

    explicit Exception(Category const& category);
    Exception(Category const& category,
	      std::string const& message);
    Exception(Category const& category,
	      std::string const& message,
	      Exception const& another);
    Exception(Exception const& other); 
    virtual ~Exception() throw();

    // The signature for what() must be identical to that of std::exception::what().
    virtual char const* what() const throw();

    virtual std::string explainSelf() const;

    std::string category() const;

    CategoryList const& history() const;
    std::string rootCause() const;

    void raise() {rethrow();}

    void append(Exception const& another);
    void append(std::string const& more_information);
    void append(char const* more_information);


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
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
    operator<<(E& e, T const& stuff);

    template <typename E, typename T>
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
    operator<<(E const& e, T const& stuff);

    template <typename E>
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
    operator<<(E& e, std::ostream&(*f)(std::ostream&));

    template <typename E>
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
    operator<<(E const& e, std::ostream&(*f)(std::ostream&));

  
    template <typename E>
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
    operator<<(E& e, std::ios_base&(*f)(std::ios_base&));

    template <typename E>
    friend
    typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
    operator<<(E const& e, std::ios_base&(*f)(std::ios_base&));

    // The following two function templates should be included, to help
    // reduce the number of function templates instantiated. However,
    // GCC 3.2.3 crashes with an internal compiler error when
    // instantiating these functions.

    //     template <typename E>
    //     friend
    //     typename detail::Desired<E, (boost::is_base_and_derived<Exception,E>::value ||
    // 				 boost::is_same<Exception,E>::value)>::type &
    //     operator<<(E& e, const char*);
  
    //    template <typename E>
    //    friend
    //    typename detail::Desired<E, (boost::is_base_and_derived<Exception,E>::value ||
    //  			       boost::is_same<Exception,E>::value)>::type &
    //  			       operator<<(E& e, char*);

  private:

    virtual std::exception* clone() const;
    virtual void rethrow();

    // data members
    std::ostringstream ost_;
    CategoryList category_;
    mutable std::string what_;
  };


  inline 
  std::ostream& 
  operator<<(std::ostream& ost, Exception const& e)
  {
    ost << e.explainSelf();
    return ost;
  }

  // -------- implementation ---------

  template <typename E, typename T>
  inline
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
  operator<<(E& e, T const& stuff)
  {
    e.ost_ << stuff;
    return e;
  }

  template <typename E, typename T>
  inline
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
  operator<<(E const& e, T const& stuff)
  {
    E& ref = const_cast<E&>(e);
    ref.ost_ << stuff;
    return e;
  }

  template <typename E>
  inline 
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type &
  operator<<(E& e, std::ostream&(*f)(std::ostream&))
  {
    f(e.ost_);
    return e;
  }

  template <typename E>
  inline 
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const&
  operator<<(E const& e, std::ostream&(*f)(std::ostream&))
  {
    E& ref = const_cast<E&>(e);
    f(ref.ost_);
    return e;
  }
  
  template <typename E>
  inline
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type & 
  operator<<(E& e, std::ios_base&(*f)(std::ios_base&))
  {
    f(e.ost_);
    return e;
  }


  template <typename E>
  inline
  typename detail::Desired<E, detail::is_derived_or_same<Exception,E>::value>::type const& 
  operator<<(E const& e, std::ios_base&(*f)(std::ios_base&))
  {
    E& ref = const_cast<E&>(e);
    f(ref.ost_);
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


}

#endif
