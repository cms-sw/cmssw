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

      void setHsb0Mask(std::vector<int > hsb0) { m_hsb0 = hsb0;};
      void setHsb1Mask(std::vector<int > hsb1) { m_hsb1 = hsb1;};

      const std::vector<int > & getHsb0Mask() const { return m_hsb0;};
      const std::vector<int > & getHsb1Mask() const { return m_hsb1;};


   private:

      std::vector<int > m_hsb0;
      std::vector<int > m_hsb1;


};


#endif
