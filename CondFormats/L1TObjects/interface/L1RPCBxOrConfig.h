#ifndef CondFormats_RPCObjects_L1RPCBxOrConfig_h
#define CondFormats_RPCObjects_L1RPCBxOrConfig_h
// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCBxOrConfig
// 
/**\class L1RPCBxOrConfig L1RPCBxOrConfig.h CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h

 Description: Contains configuration of multiple BX triggering for L1RPC emulator

 Usage:
    <usage>

*/

// forward declarations
#include "CondFormats/Serialization/interface/Serializable.h"

#include <set>
#include <vector>
#include <sstream>

#include <iostream>



class L1RPCBxOrConfig
{

   public:
      L1RPCBxOrConfig();
      virtual ~L1RPCBxOrConfig();


      int getFirstBX() const {return m_firstBX;};
      int getLastBX() const {return m_lastBX;};

      void setFirstBX(int bx) { m_firstBX = bx;};
      void setLastBX(int bx) {  m_lastBX = bx;};


   private:

      int m_firstBX;
      int m_lastBX;



   COND_SERIALIZABLE;
};


#endif
