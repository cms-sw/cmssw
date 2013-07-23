//-------------------------------------------------
//
/**  \class L1AbstractProcessor
 *
 *   Abstract Base Class for L1 Trigger Devices with EventSetup
*/
//
//   $Date: 2007/03/30 08:51:21 $
//   $Revision: 1.1 $
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
