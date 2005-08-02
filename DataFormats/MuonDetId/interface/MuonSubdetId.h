/**
   \file
   declaration of enum MuonSubDetId

   \author Stefano ARGIRO
   \version $Id$
   \date 27 Jul 2005
*/

#ifndef __MuonSubdetId_h_
#define __MuonSubdetId_h_

static const char CVSId__MuonSubdetId[] = 
"$Id$";


class MuonSubdetId {
public:

  static const unsigned int DT= 1;  
  static const unsigned int CSC=2;
  static const unsigned int RPC=3; 
};

#endif // __MuonSubdetId_h_

