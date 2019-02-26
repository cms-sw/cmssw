#ifndef INTERFACE_TTUINPUT_H 
#define INTERFACE_TTUINPUT_H 1

// Include files
#include <bitset>
#include <vector>
#include <array>

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
  
  TTUInput( const TTUInput & in ) = default;
  TTUInput( TTUInput && in ) = default;
  TTUInput & operator=( const TTUInput & rhs ) = default;
  TTUInput & operator=( TTUInput && rhs ) = default;
  
  void reset();
  
  int m_bx;
  
  int m_wheelId;
  
  bool m_hasHits;
  
  std::array<std::bitset<6>, 12> input_sec;
  std::bitset<12> m_rbcDecision;
  
  void mask ( const std::vector<int> & );
  void force( const std::vector<int> & );
  
protected:
  
private:
  
  bool m_debug;
    
};
#endif // INTERFACE_TTUINPUT_H
