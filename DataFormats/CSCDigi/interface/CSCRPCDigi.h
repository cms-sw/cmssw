#ifndef CSCRPCDigi_CSCRPCDigi_h
#define CSCRPCDigi_CSCRPCDigi_h

/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU. 
 *
 * $Date: 2008/10/29 18:34:40 $
 * $Revision: 1.7 $
 *
 * \author N. Terentiev, CMU
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class CSCRPCDigi{

public:

  /// Constructors
  CSCRPCDigi (int rpc, int pad, int bxn , int tbin);  /// from the rpc#, pad#, bxn#, tbin#
  CSCRPCDigi ();                             /// default

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

  uint16_t rpc_;
  uint16_t pad_;
  uint16_t bxn_;
  uint16_t tbin_;

};

std::ostream & operator<<(std::ostream & o, const CSCRPCDigi& digi);

#endif
