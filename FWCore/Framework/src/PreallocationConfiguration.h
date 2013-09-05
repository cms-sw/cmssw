#ifndef FWCore_Framework_PreallocationConfiguration_h
#define FWCore_Framework_PreallocationConfiguration_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     PreallocationConfiguration
// 
/**\class edm::PreallocationConfiguration PreallocationConfiguration.h "PreallocationConfiguration.h"

 Description: Holds number of simultaneous Streams, LuminosityBlocks and Runs the job will allow.

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 11 Aug 2013 19:27:57 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
  class PreallocationConfiguration
  {
    
  public:
    PreallocationConfiguration():
    PreallocationConfiguration(1,1,1,1) {}
    PreallocationConfiguration(unsigned int iNThreads,
                               unsigned int iNStreams,
                               unsigned int iNLumis,
                               unsigned int iNRuns ):
    m_nthreads(iNThreads),
    m_nStreams(iNStreams),
    m_nLumis(iNLumis),
    m_nRuns(iNRuns) {}
    
    // ---------- const member functions ---------------------
    unsigned int numberOfThreads() const {return m_nthreads;}
    unsigned int numberOfStreams() const {return m_nStreams;}
    unsigned int numberOfLuminosityBlocks() const {return m_nLumis;}
    unsigned int numberOfRuns() const {return m_nRuns;}
    
  private:
    //PreallocationConfiguration(const PreallocationConfiguration&) = delete; // stop default
    
    //const PreallocationConfiguration& operator=(const PreallocationConfiguration&) = delete; // stop default
    
    // ---------- member data --------------------------------
    unsigned int m_nthreads;
    unsigned int m_nStreams;
    unsigned int m_nLumis;
    unsigned int m_nRuns;
  };
}


#endif
