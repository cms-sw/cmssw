#ifndef Fireworks_Core_FWItemRandomAccessor_h
#define Fireworks_Core_FWItemRandomAccessor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemRandomAccessor
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id$
//

// system include files
#include "Reflex/Object.h"

// user include files
#include "Fireworks/Core/interface/FWItemAccessorBase.h"
#include "Fireworks/Core/interface/FWItemAccessorRegistry.h"

// forward declarations

/** Non templated part of the generic FWItemRandomAccessor<T>
    class. 
   
    Notice that the constructor is protected, so that it is
    impossible to instanciate this baseclass directly.
  */

class FWItemRandomAccessorBase : public FWItemAccessorBase
{
public:
   virtual ~FWItemRandomAccessorBase();

   const void*    data() const;
   const TClass*  type() const;
   const TClass*  modelType() const;

   bool           isCollection() const;

   void           setWrapper(const ROOT::Reflex::Object&);
   virtual void   reset();

protected:
   void *getDataPtr() const;   
   FWItemRandomAccessorBase(const TClass *type, const type_info &modelTypeName);
   const TClass* m_type;
   const TClass* m_modelType;
   mutable void* m_data;

private:
   FWItemRandomAccessorBase(const FWItemRandomAccessorBase&); // stop default

   const FWItemRandomAccessorBase& operator=(const FWItemRandomAccessorBase&); // stop default
};

/** A generic helper class which can be used to create
    a specialized FWItemAccessorBase plugin for
    all the classes that expose a std::vector like interface.

    The template argument T is the actual type of the
    container you want to have an accessor for and it must 
    satisty the following:

    - It must have a random access operator (i.e. operator[]()). 
    - It must have a size() method which returns the amount of
      objects contained in the collection.
    - It must contain a value_type typedef which indicates
      the type of the contained objects. If this is not the
      case, you must specify the extra template argument V.

    Notice that most of the work is actually done by the baseclass.
  */
template <class C, class V = typename C::value_type>
class FWItemRandomAccessor : public FWItemRandomAccessorBase 
{
   typedef C container_type;
   typedef V container_value_type;
public:
   FWItemRandomAccessor(const TClass *iClass)
      : FWItemRandomAccessorBase(iClass, typeid(container_value_type))
      {}

   REGISTER_FWITEMACCESSOR_METHODS();

   // ---------- const member functions ---------------------
   const void*    modelData(int iIndex) const
      {
         if (!getDataPtr())
            return 0;
         return &(reinterpret_cast<container_type *>(getDataPtr())->operator[](iIndex));
      }

   unsigned int   size() const
      {
         if (!getDataPtr())
            return 0;
         return reinterpret_cast<const container_type*>(getDataPtr())->size();
      }
};


#endif
