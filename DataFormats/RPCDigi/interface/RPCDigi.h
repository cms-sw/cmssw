#ifndef RPCDigi_RPCDigi_h
#define RPCDigi_RPCDigi_h

/** \class RPCDigi
 *
 * Digi for Resistive Plate Chambers.
 *  
 *  $Date: 2005/11/04 17:28:13 $
 *  $Revision: 1.3 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */



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
  

  /// Lenght of packed fields
  enum packing{strip_s = 10,
	       bx_s   = 10,
	       trailer_s  = 2  // padding (unused)
  };
   /// The packed digi content  
  struct PackedDigiType {
    unsigned int strip  : strip_s;
    unsigned int bx     : bx_s;
  };

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

  /// Access to the packed data
  PackedDigiType* data();

  /// Const access to the packed data
  const PackedDigiType* data() const;

 public:
  /// the packed data as seen by the persistency - should never be used 
  /// directly, only by calling data()
  /// made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };


 private:

  PersistentPacking persistentData;

};

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const RPCDigi& digi) {
  return o << digi.strip()
         << " " << digi.bx();
}

#endif

