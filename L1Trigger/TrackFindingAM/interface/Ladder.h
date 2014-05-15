#ifndef _LADDER_H_
#define _LADDER_H_

#include "Module.h"
#include <vector>

/**
   \brief Representation of a Ladder (vector of modules)
**/
class Ladder{

 private:
  vector<Module*> modules;
  
 public:
  /**
     \brief Constructor
     \param nbMod Number of modules in the ladder
     \param segmentSize Number of strips in a segment
     \param sstripSize Number of strips in a super strip
  **/
  Ladder(int nbMod, int segmentSize, int sstripSize);
  /**
     \brief Destructor
  **/
  ~Ladder();
  /**
     \brief Retrieves a module from the ladder
     \param zPos Position of the module in the ladder (0 to N)
     \return Returns a pointer on the module (not a copy), NULL if not found.
  **/
  Module* getModule(int zPos);
  /**
     Desactivates all the superstrips of the module
  **/
  void clear();

};
#endif
