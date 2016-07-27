/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Mate Csanad (mate.csanad@cern.ch)
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/


#ifndef EventFilter_CTPPSRawToDigi_SimpleVFATFrameCollection
#define EventFilter_CTPPSRawToDigi_SimpleVFATFrameCollection

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrameCollection.h"

#include <map>

/**
 * A basic implementation of VFAT frame collection, as map: TotemFramePosition --> VFATFrame.
**/
class SimpleVFATFrameCollection : public VFATFrameCollection
{
  protected:
    typedef std::map<TotemFramePosition, VFATFrame> MapType;

    MapType data;

    virtual value_type BeginIterator() const;
    virtual value_type NextIterator(const value_type&) const;
    virtual bool IsEndIterator(const value_type&) const;

  public:
    SimpleVFATFrameCollection();
    ~SimpleVFATFrameCollection();

    const VFATFrame* GetFrameByID(unsigned int ID) const;
    const VFATFrame* GetFrameByIndex(TotemFramePosition index) const;

    virtual unsigned int Size() const
    {
      return data.size();
    }

    virtual bool Empty() const
    {
      return (data.size() == 0);
    }

    void Insert(const TotemFramePosition &index, const VFATFrame &frame)
    {
      data.insert({index, frame});
    }

    /// inserts an empty (default) frame to the given position and returns pointer to the frame
    VFATFrame* InsertEmptyFrame(TotemFramePosition index)
    {
      return &data.insert({index, VFATFrame()}).first->second;
    }

    /// cleans completely the collection
    void Clear()
    {
      data.clear();
    }
};

#endif
