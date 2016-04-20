/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/


#ifndef EventFilter_TotemRawToDigi_VFATFrameCollection
#define EventFilter_TotemRawToDigi_VFATFrameCollection

#include "CondFormats/TotemReadoutObjects/interface/TotemFramePosition.h"

#include "EventFilter/TotemRawToDigi/interface/VFATFrame.h"

#include <string>

/**
 * Interface class for VFAT-frame collections.
**/
class VFATFrameCollection 
{
  public:
    VFATFrameCollection() {}
    virtual ~VFATFrameCollection() {}

    /// returns pointer to frame with ID, performs NO duplicity check (if there is precisely one frame with this 12bit ID)
    virtual const VFATFrame* GetFrameByID(unsigned int ID) const = 0;

    /// returns frame at given position in Slink frame
    virtual const VFATFrame* GetFrameByIndex(TotemFramePosition index) const = 0;

    /// returns frame at given position in Slink frame and checks 12bit ID
    virtual const VFATFrame* GetFrameByIndexID(TotemFramePosition index, unsigned int ID);

    /// return the number of VFAT frames in the collection
    virtual unsigned int Size() const = 0;

    /// returns whether the collection is empty
    virtual bool Empty() const = 0;

    /// pair: frame DAQ position, frame data
    typedef std::pair<TotemFramePosition, const VFATFrame*> value_type;

    /// the VFATFrameCollection interator
    class Iterator {
      protected:
        /// interator value
        value_type value;

        /// the pointer to the collection
        const VFATFrameCollection* collection;

      public:
        /// constructor, automatically sets the iterator to the beginning
        Iterator(const VFATFrameCollection* c = NULL) : collection(c)
          { if (collection) value = collection->BeginIterator(); }

        /// returns the DAQ position of the current element
        TotemFramePosition Position()
          { return value.first; }

        /// returns the frame data of the current element
        const VFATFrame* Data()
          { return value.second; }

        /// shifts the iterator
        void Next()
          { value = collection->NextIterator(value); }

        /// returns whether the iterator points over the end of the collection
        bool IsEnd()
          { return collection->IsEndIterator(value); }
    };
    
  protected:
    /// returns the beginning of the collection
    virtual value_type BeginIterator() const = 0;

    /// shifts the iterator
    virtual value_type NextIterator(const value_type&) const = 0;

    /// checks whether the iterator points over the end of the collection
    virtual bool IsEndIterator(const value_type&) const = 0;
};

#endif
