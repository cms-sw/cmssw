#ifndef DTPosNegType_H
#define DTPosNegType_H
/** \class DTPosNegType
 *
 *  Description: 
 *
 *
 *  $Date: 2009/06/12 10:58:46 $
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

class DTPosNegType {

 public:

  /** Constructor
   */
  DTPosNegType();

  /** Destructor
   */
  virtual ~DTPosNegType();

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


#endif // DTPosNegType_H






