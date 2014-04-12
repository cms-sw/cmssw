#ifndef CondFormats_L1TObjects_L1RPCHsbConfig_h
#define CondFormats_L1TObjects_L1RPCHsbConfig_h
// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCHsbConfig
// 
/**\class L1RPCHsbConfig L1RPCHsbConfig.h CondFormats/L1TObjects/interface/L1RPCHsbConfig.h

 Description: Contains configuration of HSB inputs

 Usage:
    <usage>

*/

// forward declarations
#include <set>
#include <vector>
#include <sstream>

#include <iostream>



class L1RPCHsbConfig
{

   public:
      L1RPCHsbConfig();
      virtual ~L1RPCHsbConfig();

      void setHsbMask(int hsb, const std::vector<int>& mask );
      int getHsbMask(int hsb, int input) const;
      int getMaskSize() const {return sizeof(m_hsb0)/sizeof(m_hsb0[0]);};

   private:

      int m_hsb0[8];
      int m_hsb1[8];


};


#endif
