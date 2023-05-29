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
#include "CondFormats/PPSObjects/interface/TotemT2FramePosition.h"

#include <map>

/**
 * A basic implementation of VFAT frame collection, as map: TotemFramePosition --> VFATFrame.
**/
class SimpleVFATFrameCollection : public VFATFrameCollection {
protected:
  typedef std::map<TotemFramePosition, VFATFrame> MapType;

  MapType data;

  value_type BeginIterator() const override;
  value_type NextIterator(const value_type&) const override;
  bool IsEndIterator(const value_type&) const override;

public:
  SimpleVFATFrameCollection();
  ~SimpleVFATFrameCollection() override;

  const VFATFrame* GetFrameByID(unsigned int ID) const override;
  const VFATFrame* GetFrameByIndex(TotemFramePosition index) const override;

  unsigned int Size() const override { return data.size(); }

  bool Empty() const override { return (data.empty()); }

  void Insert(const TotemFramePosition& index, const VFATFrame& frame) { data.insert({index, frame}); }
  void Insert(const TotemT2FramePosition& index, const VFATFrame& frame) {
    data.insert({TotemFramePosition(index.getRawPosition()), frame});
  }
  /// inserts an empty (default) frame to the given position and returns pointer to the frame
  VFATFrame* InsertEmptyFrame(TotemFramePosition index) { return &data.insert({index, VFATFrame()}).first->second; }

  /// cleans completely the collection
  void Clear() { data.clear(); }
};

#endif
