#ifndef DataFormats_PatCandidates_UserData_h
#define DataFormats_PatCandidates_UserData_h

/** \class    pat::UserData UserData.h "DataFormats/PatCandidates/interface/UserData.h"
 *
 *  \brief    Base class for data that users can add to pat objects
 *
 *  
 *
 *  \author   Sal Rappoccio
 *
 *  \version  $Id: UserData.h,v 1.01 
 *
 */

#include <string>
#include <vector>
#include <typeinfo>
#include <cxxabi.h>
#include <DataFormats/Common/interface/OwnVector.h>


namespace pat {

  class UserData {
  public:
    UserData() {}
    virtual ~UserData() {}

    /// Necessary for deep copy in OwnVector
    virtual UserData * clone() const = 0;

    /// Concrete type of stored data
    virtual const std::type_info & typeId() const = 0;
    /// Human readable name of the concrete type of stored data
    virtual const std::string    & typeName() const = 0;

    /// Extract data in a typesafe way. <T> must be the *concrete* type of the data.
    /*  I don't think there is an easy way to get it working with generic type T,
        barrying the use of ROOT::Reflex and all the edm::Ptr technicalities.
        I'd say we can live without polymorphic storage of polymorphic data */
    template<typename T> const T * get() const {
        if (typeid(T) != typeId()) return 0;
        return static_cast<const T *>(data_());
    }

    /// Get the data as a void *, for CINT usage.
    /// COMPLETELY UNSUPPORTED, USE ONLY FOR DEBUGGING
    //  Really needed for CINT? I would really like to avoid this
    const void * bareData() const { return data_(); }

    /// Make a UserData pointer from some value, wrapping it appropriately.
    /// It will check for dictionaries, unless 'transientOnly' is true
    template<typename T>
    static std::auto_ptr<UserData> make(const T &value, bool transientOnly=false) ;

  protected:
    /// Get out the data (can't template non virtual functions)
    virtual const void * data_  () const = 0;

  private:
    static void checkDictionaries(const std::type_info &type) ;
    
  };

  template<typename T>
  class UserHolder : public UserData {
    public:
        UserHolder() : obj_() {}
        UserHolder(const T &data) : obj_(data) {}
        /// Clone
        virtual UserHolder<T> * clone() const { return new UserHolder<T>(*this); }
        /// Concrete type of stored data
        virtual const std::type_info & typeId()   const { return typeid(T); }
        /// Human readable name of the concrete type of stored data
        virtual const std::string    & typeName() const { return typeName_(); }
    protected:
        virtual const void *           data_()  const { return &obj_; }
    private: 
        T obj_;
        static const std::string & typeName_() ;
  };

  typedef edm::OwnVector<pat::UserData>   UserDataCollection;
}


template<typename T>
std::auto_ptr<pat::UserData> pat::UserData::make(const T &value, bool transientOnly) {
    if (!transientOnly) {
        checkDictionaries(typeid(T));
        checkDictionaries(typeid(pat::UserHolder<T>));
    }
    return std::auto_ptr<UserData>(new pat::UserHolder<T>(value));  
}

template<typename T> 
const std::string & pat::UserHolder<T>::typeName_() {
    static int status = 0;
    static const char * demangled = abi::__cxa_demangle(typeid(T).name(),  0, 0, &status);
    static const std::string name(status == 0 ? demangled : "[UNKNOWN]");
    return name;
}

#endif
