#ifndef FWCore_ServiceRegistry_SystemBounds_h
#define FWCore_ServiceRegistry_SystemBounds_h
// -*- C++ -*-
//
// Package:     FWCore/ServiceRegistry
// Class  :     SystemBounds
// 
/**\class SystemBounds SystemBounds.h "SystemBounds.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 08 Sep 2013 16:16:25 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
  namespace service {
    class SystemBounds
    {
      
    public:
      SystemBounds(unsigned int iNStreams,
                   unsigned int iNLumis,
                   unsigned int iNRuns,
                   unsigned int iNThreads) :
      m_nStreams(iNStreams),
      m_nLumis(iNLumis),
      m_nRuns(iNRuns),
      m_nThreads(iNThreads){}
      
      // ---------- const member functions ---------------------
      unsigned int maxNumberOfStreams() const {return m_nStreams; }
      unsigned int maxNumberOfConcurrentRuns() const {return m_nRuns;}
      unsigned int maxNumberOfConcurrentLuminosityBlocks() const {return m_nLumis;}
      unsigned int maxNumberOfThreads() const { return m_nThreads; }
      
    private:
      
      // ---------- member data --------------------------------
      unsigned int m_nStreams;
      unsigned int m_nLumis;
      unsigned int m_nRuns;
      unsigned int m_nThreads;
    };

  }
}


#endif
