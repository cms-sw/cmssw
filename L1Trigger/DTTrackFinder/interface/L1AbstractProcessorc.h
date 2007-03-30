//-------------------------------------------------
//
/**  \class L1AbstractProcessorc
 *
 *   Abstract Base Class for L1 Trigger Devices with EventSetup
*/
//
//   $Date: 2006/06/26 15:52:12 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
#ifndef L1_ABSTRACT_PROCESSORC_H
#define L1_ABSTRACT_PROCESSORC_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <FWCore/Framework/interface/EventSetup.h>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1AbstractProcessorc {

  public:

    /// destructor 
    virtual ~L1AbstractProcessorc() {}

    /// run processor logic
    virtual void run(const edm::EventSetup& c) = 0;

    /// clear event memory of processor
    virtual void reset() = 0;

};

#endif
