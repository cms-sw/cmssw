#ifndef RPCObjects_L1RPCConfig_h
#define RPCObjects_L1RPCConfig_h
// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCConfig
// 
/**\class L1RPCConfig L1RPCConfig.h CondFormats/RPCObjects/interface/L1RPCConfig.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Tue Mar 20 12:03:50 CET 2007
// $Id: L1RPCConfig.h,v 1.1 2007/03/26 09:03:50 fruboes Exp $
//

#include <string>
#include <vector>

#include "CondFormats/RPCObjects/interface/RPCPattern.h"

class L1RPCConfig
{

   public:
      L1RPCConfig();
      ~L1RPCConfig();

      void setPPT(int data) {m_ppt = data;};
      int  getPPT() const {return m_ppt;};

     // void setDataDir(const std::string &dir); // Temporary
     // std::string getDataDir()const {return m_dataDir;};



      std::vector< std::vector< std::vector< RPCPattern::RPCPatVec > > >  m_pats;
      std::vector< std::vector< std::vector< RPCPattern::TQualityVec > > >  m_quals;

   private:
      int m_ppt;
     // std::string m_dataDir;

      
      // m_pats[tower][sector][segment][patternNo]
      
};


#endif
