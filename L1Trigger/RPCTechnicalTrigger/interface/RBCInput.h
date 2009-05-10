// $Id: RBCInput.h,v 1.2 2009/05/08 10:24:04 aosorio Exp $
#ifndef INTERFACE_RBCINPUT_H 
#define INTERFACE_RBCINPUT_H 1

// Include files
#include <stdlib.h>
#include <istream>
#include <ostream>
#include <iostream>
#include <bitset>
#include <vector>

/** @class RBCInput RBCInput.h interface/RBCInput.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-16
 */
class RBCInput {
public: 
  /// Standard constructor
  RBCInput( ) { 
    input_sec = new std::bitset<15>[2];
    needmapping = false; 
    m_debug = false; 
  }; 
  
  virtual ~RBCInput( ) {
    if ( input_sec ) delete[] input_sec;
  }; ///< Destructor
  
  RBCInput( const RBCInput & _in )
  {
    for(int i=0; i < 30; ++i) input[i] = _in.input[i];
    for(int i=0; i <  2; ++i) input_sec[i] = _in.input_sec[i];
    needmapping = _in.needmapping;
    m_debug = _in.m_debug;
  };
  
  RBCInput & operator=(const RBCInput & rhs) 
  {
    if (this == &rhs) return (*this);
    for(int i=0; i < 30; ++i) (*this).input[i]     = rhs.input[i];
    for(int i=0; i <  2; ++i) (*this).input_sec[i] = rhs.input_sec[i];
    (*this).needmapping = rhs.needmapping;
    (*this).m_debug = rhs.m_debug;
    return (*this);
  };
  
  // io functions
  friend std::istream& operator>>(std::istream &istr, RBCInput &);
  friend std::ostream& operator<<(std::ostream &ostr, RBCInput &);
  
  bool input[30];
  std::bitset<15>  * input_sec;
  
  void printinfo() {
    std::cout << "RBCInput: " << (*this);
  };
  
  void mask ( const std::vector<int> & );
  
  void force( const std::vector<int> & );
  
  bool needmapping;
  
protected:
  
private:
  
  bool m_debug;
  
};
#endif // INTERFACE_RBCINPUT_H
