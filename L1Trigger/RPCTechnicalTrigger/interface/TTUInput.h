// $Id: TTUInput.h,v 1.4 2009/05/24 21:45:39 aosorio Exp $
#ifndef INTERFACE_TTUINPUT_H 
#define INTERFACE_TTUINPUT_H 1

// Include files
#include <bitset>
#include <vector>

/** @class TTUInput TTUInput.h interface/TTUInput.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-16
 */
class TTUInput {
public: 
  
  /// Standard constructor
  TTUInput( );
  
  ///< Destructor
  virtual ~TTUInput( );
  
  TTUInput( const TTUInput & in )
  {
    m_bx = in.m_bx;
    m_hasHits = in.m_hasHits;
    input_sec = new std::bitset<6>[12];
    for(int i=0; i < 12; ++i) 
      input_sec[i] = in.input_sec[i];
  };
  
  TTUInput & operator=( const TTUInput & rhs )
  {
    if (this == &rhs) return (*this);
    (*this).m_bx = rhs.m_bx;
    (*this).m_hasHits = rhs.m_hasHits;
    (*this).input_sec = new std::bitset<6>[12];
    for(int i=0; i < 12; ++i)
      (*this).input_sec[i] = rhs.input_sec[i];
    return (*this);
  };
  
  void reset();
  
  int m_bx;
  bool m_hasHits;
  std::bitset<6> * input_sec;
  
  void mask ( const std::vector<int> & );
  
  void force( const std::vector<int> & );
  
protected:
  
private:
  
  bool m_debug;
    
};
#endif // INTERFACE_TTUINPUT_H
