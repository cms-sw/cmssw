#ifndef Fireworks_Core_FWItemRandomAccessor_h
#define Fireworks_Core_FWItemRandomAccessor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemRandomAccessor
//
// Original Author:  Giulio Eulisse
//         Created:  Thu Feb 18 15:19:44 EDT 2008
// $Id: FWItemRandomAccessor.h,v 1.10 2012/08/03 18:20:27 wmtan Exp $
//

// system include files
#include "FWCore/Utilities/interface/ObjectWithDict.h"

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

   void           setData(const edm::ObjectWithDict&);
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

/** Generic class for creating accessors for containers which 
    are implemented as a container of containers. This for example includes
    `DetSetVector` hence the name. Both the outer and the inner containers
    must follow the Random Access Container model and in particular
    must have a size() method. The outer collection must be iterable, while
    the inner collections must have an array subscript operator. 
  */
template <class C, class COLL = typename C::value_type, class V = typename COLL::value_type >
class FWItemDetSetAccessor : public FWItemRandomAccessorBase
{
public:
   typedef C      container_type;
   typedef COLL   collection_type;
   typedef V      collection_value_type;

   FWItemDetSetAccessor(const TClass *iClass)
      : FWItemRandomAccessorBase(iClass, typeid(collection_value_type))
      {}

   REGISTER_FWITEMACCESSOR_METHODS();

   const void*    modelData(int iIndex) const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t collectionOffset = 0;
         for (typename container_type::const_iterator ci = c->begin(), ce = c->end(); ci != ce; ++ci)
         {
            size_t i = iIndex - collectionOffset;
            if (i < ci->size())
               return &(ci->operator[](i));
            collectionOffset += ci->size();
         }

         return 0;
      }

   unsigned int   size() const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t finalSize = 0;

         for (typename container_type::const_iterator i = c->begin(), e = c->end(); i != e; ++i)
            finalSize += i->size();
         
         return finalSize;
      }
};

/** Specialized accessor for the new edmNew::DetSetVector classes.
  */
template <class C, class COLL = typename C::value_type, class V = typename COLL::value_type >
class FWItemNewDetSetAccessor : public FWItemRandomAccessorBase
{
public:
   typedef C      container_type;
   typedef COLL   collection_type;
   typedef V      collection_value_type;

   FWItemNewDetSetAccessor(const TClass *iClass)
      : FWItemRandomAccessorBase(iClass, typeid(collection_value_type))
      {}

   REGISTER_FWITEMACCESSOR_METHODS();

   const void*    modelData(int iIndex) const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         if (iIndex < 0)
            return 0;

         return &(c->data().operator[](iIndex));
      }

   unsigned int   size() const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         return c->dataSize();
      }
};

template <class C, class R = typename C::Range, class V = typename R::value_type>
class FWItemRangeAccessor : public FWItemRandomAccessorBase
{
public:
   typedef C            container_type;
   typedef R            range_type;
   typedef V            value_type;

   FWItemRangeAccessor(const TClass *iClass)
      : FWItemRandomAccessorBase(iClass, typeid(value_type))
      {}

  REGISTER_FWITEMACCESSOR_METHODS();

   const void*    modelData(int iIndex) const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t collectionOffset = 0;
         for (typename container_type::const_iterator ci = c->begin(), ce = c->end(); ci != ce; ++ci)
         {
            size_t i = iIndex - collectionOffset;
            if (i < std::distance(ci->first, ci->second))
               return &(*(ci + i));
            collectionOffset += ci->size();
         }

         return 0;
      }

   unsigned int   size() const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t finalSize = 0;

         for (typename range_type::const_iterator ci = c->begin(), ce = c->end(); ci != ce; ++ci)
            finalSize += std::distance(ci->first, ci->second);

         return finalSize;
      }
};

template <class C, class V>
class FWItemMuonDigiAccessor : public FWItemRandomAccessorBase
{
public:
   typedef C            container_type;
   typedef V            value_type;

   FWItemMuonDigiAccessor(const TClass *iClass)
      : FWItemRandomAccessorBase(iClass, typeid(value_type))
      {}

  REGISTER_FWITEMACCESSOR_METHODS();

   const void*    modelData(int iIndex) const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t collectionOffset = 0;

         for (typename container_type::DigiRangeIterator ci = c->begin(), ce = c->end(); ci != ce; ++ci)
         {
            int i = iIndex - collectionOffset;

            typename container_type::DigiRangeIterator::value_type vt = *ci;


            if (i < std::distance(vt.second.first, vt.second.second))
               return &(*(vt.second.first + i));
            collectionOffset += std::distance(vt.second.first, vt.second.second);
         }

         return 0;
      }

   unsigned int   size() const
      {
         if (!getDataPtr())
            return 0;
         const container_type *c = reinterpret_cast<const container_type*>(getDataPtr());
         size_t finalSize = 0;

         for (typename container_type::DigiRangeIterator ci = c->begin(), ce = c->end(); ci != ce; ++ci)
         {
           typename container_type::DigiRangeIterator::value_type vt = *ci;
           finalSize += std::distance(vt.second.first, vt.second.second);
         }
         
         return finalSize;
      }
};

#endif
