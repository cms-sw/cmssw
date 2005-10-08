#ifndef EBMonitorUtils_h
#define EBMonitorUtils_h

/*
 *
 * \file EBMonitorUtils.h
 * $Date: 2005/10/07 08:33:52 $
 * \$Revision: 1.1 $
 * \author B. Gobbo
 *
*/

class EBMonitorUtils {

 public:

  //! Returns the SM number, given phi [1:360], and zed [-1 or 1]
  static int getSuperModuleID( const int phi, const int zed ); 

  //! Returns a crystal number given eta [1:85] and phi [1:360] 
  static int getCrystalID( const int eta, const int phi );

  //! Returns eta and phi given a crystal [1:1700] in a SM [1:26] 
  static void getEtaPhi( const int crystal, const int sm, int &eta, int &phi );

};

#endif //EBMonitorUtils_h
