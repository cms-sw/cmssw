//-------------------------------------------------
//
/**  \class L1AbstractProcessor
 *
 *   Abstract Base Class for L1 Trigger Devices with EventSetup
*/
//
//   $Date: 2008/02/18 17:38:01 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//--------------------------------------------------
#ifndef L1_ABSTRACT_PROCESSOR_H
#define L1_ABSTRACT_PROCESSOR_H

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

class L1AbstractProcessor {

  public:

    /// destructor 
    virtual ~L1AbstractProcessor() {}

    /// run processor logic
    virtual void run() {};

    virtual void run(const edm::EventSetup& c) {};

    /// clear event memory of processor
    virtual void reset() = 0;

};

#endif
