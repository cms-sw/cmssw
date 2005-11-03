#ifndef RPCDigi_RPCDigi_h
#define RPCDigi_RPCDigi_h

/** \class RPCDigi
 *
 * Digi for RPC
 *  
 *  $Date: 2005/10/25 13:48:54 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni (CERN)
 *
 */


class RPCDigi{

public:
 RPCDigi(int channelID, int bxID);
 ~RPCDigi();


private:

  int chID_;
  int bxID_;

};

#endif

