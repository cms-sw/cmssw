#ifndef CSCRPCDigi_CSCRPCDigi_h
#define CSCRPCDigi_CSCRPCDigi_h

/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU. 
 *
 * $Date: 2006/03/01 09:40:30 $
 * $Revision: 1.2 $
 *
 * \author N. Terentiev, CMU
 */

class CSCRPCDigi{

public:

  /// Constructors
  explicit CSCRPCDigi (int rpc, int pad, int bxn , int tbin);  /// from the rpc#, pad#, bxn#, tbin#
  CSCRPCDigi (const CSCRPCDigi& digi);       /// copy
  CSCRPCDigi ();                             /// default

  /// Assignment operator
  CSCRPCDigi& operator=(const CSCRPCDigi& digi);

  
  
  /// get RPC
  int getRpc() const {return rpc_;}
  /// return pad number
  int getPad() const {return pad_;}
  /// return tbin number
  int getTbin() const {return tbin_;}
  /// return BXN
  int getBXN() const {return bxn_;}
  
  /// Print content of digi
  void print() const;


private:
  friend class testCSCDigis;
  unsigned int rpc_;
  unsigned int pad_;
  unsigned int bxn_;
  unsigned int tbin_;


};




#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCRPCDigi& digi) {
  return o << " RPC = " << digi.getRpc() << "  Pad = "<< digi.getPad()
	   << "  Tbin = " << digi.getTbin() << "  Bxn = " << digi.getBXN();
}



#endif
