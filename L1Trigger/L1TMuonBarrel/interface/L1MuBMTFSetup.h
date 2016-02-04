//-------------------------------------------------
//
/**  \class L1MuBMTFSetup
 *
 *   Setup the L1 barrel Muon Trigger Track Finder
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TF_SETUP_H
#define L1MUBM_TF_SETUP_H

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
#include "FWCore/Framework/interface/Event.h"

class L1MuBMTrackFinder;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTFSetup {

  public:

    /// constructor
    L1MuBMTFSetup(const edm::ParameterSet & ps,edm::ConsumesCollector && ix);

    /// destructor
    virtual ~L1MuBMTFSetup();

    /// perform action per run

    /// return the main trigger object
    L1MuBMTrackFinder* TrackFinder() { return m_tf; }

  private:

    L1MuBMTrackFinder* m_tf;
    const edm::EventSetup* m_es;
};

#endif
