#ifndef _MODULE_H_
#define _MODULE_H_

#include "Segment.h"
#include <vector>

/**
\brief Representation of a Module (2 segments)
**/
class Module{

 private:
  Segment* segments[2];
  
 public:
  /**
     \brief Constructor
     \param segmentSize Number of strips in a segment
     \param sstripSize Number of strips in a super strip
  **/
  Module(int segmentSize, int sstripSize);
  /**
     \brief Destructor
  **/
  ~Module();
  /**
     \brief Retrieves on of the segment in the module
     \param n The position of the segment (0 or 1).
     \return A pointer on the Segment (not a copy), NULL if not found.
  **/
  Segment* getSegment(int n);
  /**
     \brief Desactivates all the super strips in the module
  **/
  void clear();

};
#endif
