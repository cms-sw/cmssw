#ifndef RPCDigi_RPCDigi_h
#define RPCDigi_RPCDigi_h

/** \class RPCDigi
 *
 * Digi for Resistive Plate Chambers.
 *  
 *  $Date: 2006/04/06 07:54:01 $
 *  $Revision: 1.5 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


#include <boost/cstdint.hpp>

class RPCDigi{

public:
 
  ///Default constructor
  RPCDigi(){}
 
  ///Constructor from Strip and BX
  explicit RPCDigi(int strip,int bx);
  
  /// Copy constructor
  RPCDigi (const RPCDigi& digi);

  ///Destructor
  ~RPCDigi(){}
  

  /// Assignment operator
  RPCDigi& operator=(const RPCDigi& digi);

  /// Precedence operator
   bool operator<(const  RPCDigi& d)const;
   
  ///Comparison operator
  bool operator==(const RPCDigi& digi) const;

  ///Print content of Digi
  void print() const;

  /// Get Strip
  int strip() const;
  
  /// Get BX
  int bx() const;

  /// Set strip and bx
  void setStripBx(int strip, int bx);  

  ///Set strip
  void setStrip(int strip);  

  ///Set bx
  void setBx(int bx);


private:
  
  uint16_t strip_;
  int16_t bx_;
   

   /// The packed digi content  
  struct PackedDigiType {
    uint16_t theStrip; 
    int16_t theBx;    
  };


};

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const RPCDigi& digi) {
  return o << digi.strip()
         << " " << digi.bx();
}

#endif

