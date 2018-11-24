#ifndef RBCID_H 
#define RBCID_H 1

// Include files
#include <iostream>

/** @class RBCId RBCId.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-12
 */
class RBCId {
public: 
  /// Standard constructor
  RBCId( ); 

  RBCId(int , int * );
  
  RBCId(const RBCId &) = default;
  RBCId(RBCId &&) = default;
  RBCId& operator=(RBCId const&) = default;
  RBCId& operator=(RBCId&&) = default;
  
  int wheel() const { return m_wheel;};
  
  int wheelIdx() const { return (m_wheel+2);}; // wheel index starts from 0
  
  int sector( int _sec ) const { return m_sector[_sec]; };
  
  void setid ( int _wh, int *_sec) { 
    m_wheel = _wh;
    m_sector[0] = _sec[0];
    m_sector[1] = _sec[1];
  };
  
  void printinfo() const;
    
protected:
  
private:
  
  int m_wheel;
  int m_sector[2];
  
};
#endif // RBCID_H
