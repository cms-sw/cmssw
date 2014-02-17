#ifndef L1Trigger_RPCTrigger_MuonsGrabber_h
#define L1Trigger_RPCTrigger_MuonsGrabber_h
// -*- C++ -*-
//
// Package:     RPCTrigger
// Class  :     MuonsGrabber
// 
/**\class MuonsGrabber MuonsGrabber.h L1Trigger/RPCTrigger/interface/MuonsGrabber.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Sep 17 14:20:56 CEST 2009
// $Id: MuonsGrabber.h,v 1.1 2009/09/23 11:01:55 fruboes Exp $
//

// system include files

// user include files

// forward declarations

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/interface/RPCBasicTrigConfig.h"

#include <map>
#include <vector>

//#include <xercesc/util/PlatformUtils.hpp>
//#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
//#include <xercesc/framework/LocalFileFormatTarget.hpp>


struct RPCMuonExtraStruct {
   RPCMuonExtraStruct(signed char l, signed char r, signed char h, signed char i, RPCTBMuon & mu): 
                                _level(l)  , _region(r), _hsHalf(h), _index(i), _mu(mu){};	
   signed char _level;
   signed char _region; // brl/endcap
   signed char _hsHalf; // Determines which halfsorter 	
   signed char _index; 
   RPCTBMuon _mu;
   static bool lvlCompare(const RPCMuonExtraStruct &a, const RPCMuonExtraStruct &b) {
     return a._level > b._level;
   };
};

class MuonsGrabber
{
  
      MuonsGrabber();
      virtual ~MuonsGrabber();
   public:
      static MuonsGrabber & Instance();

      void setRPCBasicTrigConfig(RPCBasicTrigConfig * c) {m_trigConfig = c;};
   
      void startNewEvent(int event, int bx);
      void writeDataForRelativeBX(int bx);
      void addMuon(RPCTBMuon & mu, int lvl, int region, int hs, int index);

   private:
      MuonsGrabber(const MuonsGrabber&); // stop default

      const MuonsGrabber& operator=(const MuonsGrabber&); // stop default
      std::string IntToString(int i);

      // ---------- member data --------------------------------
      //std::map<int, std::vector< RPCTBMuon  > > m_muons;
      std::vector< RPCMuonExtraStruct > m_muons;
      RPCBasicTrigConfig* m_trigConfig;
           
      int m_currentEvent;
      int m_currentBX;
      XERCES_CPP_NAMESPACE::DOMImplementation* m_dom;
      XERCES_CPP_NAMESPACE::DOMDocument* m_doc;
      XERCES_CPP_NAMESPACE::DOMElement* m_rootElem;
      XERCES_CPP_NAMESPACE::DOMElement* m_currEvent;
};


#endif
