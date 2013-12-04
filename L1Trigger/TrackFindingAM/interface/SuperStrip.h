#ifndef _SUPERSTRIP_H_
#define _SUPERSTRIP_H_

#include <iostream>
#include <vector>
#include "Hit.h"

using namespace std;

/**
\brief Representation of a detector super strip (group of strips).
It contains the list of Hits that have touched the super strip.
**/

class SuperStrip{

 private :
  bool hit;//touched or not
  short size;//number of strips in the SuperStrip
  vector<Hit*> hits;//list of full resolution hits

 public:
  /**
     \brief Constructor
     \param s The number of strips in the super strip
  **/
  SuperStrip(int s);
  /**
     \brief Destructor
  **/
  ~SuperStrip();
  /**
     \brief Get the number of strips in the super strip
     \return The size of the super strip
  **/
  short getSize();
  /**
     \brief Check if the super strip has been hit
     \return True if a Hit occured
  **/
  bool isHit();
  /**
     \brief getHits
     \return A vector containing pointers on the Hit objects (the Hits are not copied)
  **/
  vector<Hit*>& getHits();
  /**
     \brief Reset the super strip
  **/
  void clear();
  /**
     \brief Add a new hit to the super strip
     \param h A pointer on the Hit object : this object will be copied. 
  **/
  void touch(const Hit* h);
};
#endif
