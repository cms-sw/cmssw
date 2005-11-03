/** \file
 * 
 *  $Date: 2005/11/03 13:48:54 $
 *  $Revision: 1.2 $
 *
 * \author Ilaria Segoni
 */


#include <DataFormats/RPCDigi/interface/RPCDigi.h>

#include <iostream>
#include <bitset>

using namespace std;


RPCDigi::RPCDigi (int channelID, int bxID){
  chID_=channelID;
  bxID_=bxID;
}


