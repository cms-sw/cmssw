// $Id: RBCId.h,v 1.4 2009/05/24 21:45:39 aosorio Exp $
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
  
  RBCId(const RBCId &);
  
  virtual ~RBCId( ); ///< Destructor
  
  int wheel() const { return m_wheel;};
  
  int wheelIdx() const { return (m_wheel+2);}; // wheel index starts from 0
  
  int sector( int _sec ) const { return m_sector[_sec]; };
  
  void setid ( int _wh, int *_sec) { 
    m_wheel = _wh;
    m_sector[0] = _sec[0];
    m_sector[1] = _sec[1];
  };
  
  void printinfo();
    
protected:
  
private:
  
  int m_wheel;
  int m_sector[2];
  
};
#endif // RBCID_H
