//-------------------------------------------------
//
/**  \class L1MuDTTFSetup
 *
 *   Setup the L1 barrel Muon Trigger Track Finder
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_TF_SETUP_H 
#define L1MUDT_TF_SETUP_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/ConsumesCollector.h>
class L1MuDTTrackFinder;

//              ---------------------
//              -- Class Interface --
//              ---------------------
 
class L1MuDTTFSetup {

  public:

    /// constructor
    L1MuDTTFSetup(const edm::ParameterSet & ps,edm::ConsumesCollector && ix);

    /// destructor
    virtual ~L1MuDTTFSetup();

    /// perform action per run

    /// return the main trigger object
    L1MuDTTrackFinder* TrackFinder() { return m_tf; }

  private:

    L1MuDTTrackFinder* m_tf;

};

#endif
