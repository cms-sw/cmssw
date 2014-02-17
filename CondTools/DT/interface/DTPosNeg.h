#ifndef DTPosNeg_H
#define DTPosNeg_H
/** \class DTPosNeg
 *
 *  Description: 
 *
 *
 *  $Date: 2008/08/15 13:46:46 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTChamberId;

//---------------
// C++ Headers --
//---------------
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

/*
class DTPosNegCompare {
 public:
  bool operator()( const DTCCBId& idl,
                   const DTCCBId& idr ) const;
};
*/

class DTPosNeg {

 public:

  /** Constructor
   */
  DTPosNeg();

  /** Destructor
   */
  virtual ~DTPosNeg();

  /** Operations
   */
  /// dump map
  static void dump();
  static int getPN( int whe, int sec, int sta );
  static int getPN( const DTChamberId& cha );
  static int getCT( int whe, int sec, int sta );
  static int getCT( const DTChamberId& cha );

 private:

  static bool initRequest;
  static std::map<int,int> geomMap;
//  static std::map<DTCCBId,int,DTPosNegCompare> geomMap;

  static void fillMap();
  static int idCode( int whe, int sec, int sta );
  static int pnCode( int p, int t );
  static void decode( int code, int& whe, int& sec, int& sta );
  static void decode( int code, int& p, int& t );
  static int getData( int whe, int sec, int sta );

};


#endif // DTPosNeg_H






