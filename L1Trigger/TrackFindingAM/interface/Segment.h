#ifndef _SEGMENT_H_
#define _SEGMENT_H_

#include "SuperStrip.h"
#include <vector>

/**
   \brief Representation of a segment (vector of superstrips)
**/
class Segment{

 private:
  vector<SuperStrip*> strips;
  short sStripSize;

 public:
  /**
     \brief Constructor
     \param stripNumber Total number of strips in a segment (should be 1024 for CMS external layers)
     \param sstripSize Size of a super strip (16,32,64,128,256,1024)
  **/
  Segment(int stripNumber, int sstripSize);
  /**
     \brief Destructor
  **/
  ~Segment();
  /**
     \brief Get a super strip pointer from a strip number
     \param stripNumber The number of a strip
     \return A pointer (not a copy) of the superStrip object containing the given strip
  **/
  SuperStrip* getSuperStrip(int stripNumber);
  /**
     \brief Get a super strip pointer from a super strip position
     \param sstripNumber The position of the superstrip
     \return A pointer (not a copy) of the superStrip object containing the given strip
  **/
  SuperStrip* getSuperStripFromIndex(int sstripNumber);
  /**
     \brief Clear all the superstrips
  **/
  void clear();
};
#endif
