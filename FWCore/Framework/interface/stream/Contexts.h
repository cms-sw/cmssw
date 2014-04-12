#ifndef FWCore_Framework_stream_Contexts_h
#define FWCore_Framework_stream_Contexts_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     Contexts
// 
/**\class Contexts Contexts.h "Contexts.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:19:39 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
  namespace stream {
    
    template<typename R, typename G>
    class RunContextT {
    public:
      RunContextT(R const* iRun, G const* iGlobal): m_run(iRun), m_global(iGlobal) {}
      R const* run() const { return m_run;}
      G const* global() const { return m_global;}
  
    private:
      R const* m_run;
      G const* m_global;
    };
    
    template<typename L, typename R, typename G>
    class LuminosityBlockContextT {
    public:
      LuminosityBlockContextT(L const* iLumi, R const* iRun, G const* iGlobal):
      m_lumi(iLumi),m_run(iRun),m_global(iGlobal) {}
      
      L const* luminosityBlock() const { return m_lumi;}
      R const* run() const {return m_run;}
      G const* global() const { return m_global;}
    private:
      L const* m_lumi;
      R const* m_run;
      G const* m_global;
    };
  }
}

#endif
