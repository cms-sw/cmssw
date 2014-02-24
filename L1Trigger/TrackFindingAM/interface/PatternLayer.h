#ifndef _PATTERNLAYER_H_
#define _PATTERNLAYER_H_

#include <iostream>
#include <bitset>
#include <map>
#include <cmath>
#include <cstring>

#include "Detector.h"

#include <boost/serialization/split_member.hpp>

using namespace std;

/**
   \brief Layer part of a pattern
   This class contains the bits for one layer
**/

class PatternLayer{

 public:
  /**
     \brief Number of bits per patternLayer
  **/
  static const int LAYER_BITS=16;
  /**
     \brief Number of maximum DC bits
  **/
  static const int DC_BITS=3;
   /**
     \brief Map between GRAY code and decimal values
   **/
  static map<string, int> GRAY_POSITIONS;
  /**
     \brief Cache between DC bits values and positions
  **/
  static map<string, vector<string> > positions_cache;
  /**
     \brief Constructor
  **/
  PatternLayer();
  /**
     \brief Destructor
  **/
  virtual ~PatternLayer(){};
  /**
     \brief Returns a copy of the PatternLayer
     \return A pointer on the copy
  **/
  virtual PatternLayer* clone()=0;

  /**
     \brief Returns the ladder's position in phi
     \return The ladder's phi position
  **/
  virtual short getPhi()=0;

  /**
     \brief Set the values of the Don't Care bits
     \param index The index of the bit
     \param val The value of the bit (between 0 and 3)
  **/
  void setDC(int index, char val);
  /**
     \brief Get a DC bit
     \param index The index of the bit
     \return The value of the DC bit in position index
  **/
  char getDC(int index);
  /**
     \brief Returns the int value of the bitset as a string of 5 characters
     \return A string (ie : "28653" or "00142")
  **/
  string getCode();
  /**
     \brief Get the bitset value as an integer
     \return The value of the bitset (ie 28653 or 142)
  **/
  int getIntValue() const;
  /**
     \brief Change the value contained in the bitset
     \param v The new value as an integer
  **/
  void setIntValue(int v);
  /**
     \brief Retrieve the SuperStrip objects corresponding to the PatternLayer from the Detector structure
     \param l The layer of the PatternLayer (starting from 0)
     \param ladd The ladders of this layer for the current sector
     \param modules The modules in the current sector
     \param d The detector structure
     \return A list of SuperStrip*. If no DC bits are used we have only one value.
  **/
  virtual vector<SuperStrip*> getSuperStrip(int l, const vector<int>& ladd, const map<int, vector<int> >& modules, Detector& d)=0;

 /**
     \brief Allows to display a PatternLayer as a string
  **/
  virtual string toString()=0;

 /**
     \brief Returns the Super strip position
     \return The position of the super strip
  **/
  virtual short getStrip()=0;
  /**
     \brief Get the list of positions from the DC bits
  **/
  vector<string> getPositionsFromDC();

  /**
     \brief Get the number of DC bits used
     \return The number of active DC bits for this PatternLayer
  **/
  int getDCBitsNumber();

 private:
  /**
     Get the list of positions from the DC bits
  **/
  void getPositionsFromDC(vector<char> dc, vector<string>& positions);
  static map<string, int> CreateMap();


 protected:
  bitset<LAYER_BITS> bits;
  /*  
      4 possible values for a DC bit:
      0 : 0
      1 : 1
      2 : X (Don't Care)
      3 : UNUSED
  */
  char dc_bits[DC_BITS];

  friend class boost::serialization::access;
  
  template<class Archive> void save(Archive & ar, const unsigned int version) const//const boost::serialization::version_type& version) const 
    {
      unsigned long i(this->getIntValue());
      ar << i; 
      for(int j=0;j<DC_BITS;j++){
	ar << dc_bits[j];
      }
    }
  
  template<class Archive> void load(Archive & ar, const unsigned int version)
    {
      int i;
      ar >> i;
      setIntValue(i); 
      for(int j=0;j<DC_BITS;j++){
	ar >> dc_bits[j];
      }
    }
  
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  
};
#endif
