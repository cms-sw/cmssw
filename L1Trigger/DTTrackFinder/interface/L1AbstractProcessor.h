//-------------------------------------------------
//
/**  \class L1AbstractProcessor
 *
 *   Abstract Base Class for L1 Trigger Devices
*/
//
//   $Date: 2006/06/26 15:52:12 $
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


//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1AbstractProcessor {

  public:

    /// destructor 
    virtual ~L1AbstractProcessor() {}

    /// run processor logic
    virtual void run() = 0;

    /// clear event memory of processor
    virtual void reset() = 0;

};

#endif
