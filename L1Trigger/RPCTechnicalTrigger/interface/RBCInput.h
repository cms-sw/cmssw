#ifndef INTERFACE_RBCINPUT_H 
#define INTERFACE_RBCINPUT_H 1

// Include files
#include <cstdlib>
#include <istream>
#include <ostream>
#include <iostream>
#include <bitset>
#include <vector>
#include <array>

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
    needmapping = false; 
    m_debug = false; 
    hasData = false;
  }; 
  
  RBCInput( const RBCInput &) = default;  
  RBCInput( RBCInput &&) = default;  
  RBCInput & operator=(const RBCInput & ) = default;
  RBCInput & operator=(RBCInput && ) = default;
  
  // io functions
  friend std::istream& operator>>(std::istream &istr, RBCInput&);
  friend std::ostream& operator<<(std::ostream &ostr, RBCInput const &);
  
  bool input[30];
  std::array<std::bitset<15>,2> input_sec;
  
  void printinfo() const {
    std::cout << "RBCInput: " << (*this);
  };
  
  void mask ( const std::vector<int> & );
  
  void force( const std::vector<int> & );

  bool hasData;
  bool needmapping;
  
  void setWheelId( int wid ) { 
    m_wheelId = wid;
  };
  
  int wheelId() const {
    return m_wheelId;
  };
  
  
private:
  
  bool m_debug;

  int m_wheelId;
    
};
#endif // INTERFACE_RBCINPUT_H
