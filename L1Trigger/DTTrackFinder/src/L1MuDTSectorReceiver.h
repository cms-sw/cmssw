//-------------------------------------------------
//
/**  \class L1MuDTSectorReceiver
 *
 *   Sector Receiver:
 *
 *   The Sector Receiver receives track segment
 *   data from the DTBX and CSC chamber triggers
 *   and stores it into the Data Buffer
 *
 *
 *   $Date: 2007/04/01 10:05:05 $
 *   $Revision: 1.5 $
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUDT_SECTOR_RECEIVER_H
#define L1MUDT_SECTOR_RECEIVER_H

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
class L1MuDTSectorProcessor;
class L1MuDTTFParameters;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSectorReceiver {

  public:

    /// constructor
    L1MuDTSectorReceiver(L1MuDTSectorProcessor& );

    /// destructor
    virtual ~L1MuDTSectorReceiver();

    /// receive track segment data from the DTBX and CSC chamber triggers
    void run(int bx, const edm::Event& e, const edm::EventSetup& c);
    
    /// clear Sector Receiver
    void reset();

  private:

    /// receive track segment data from DTBX chamber trigger
    void receiveDTBXData(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// receive track segment data from CSC chamber trigger
    void receiveCSCData(int bx, const edm::Event& e, const edm::EventSetup& c);
    
    /// find the right sector for a given address
    int address2sector(int adr) const;
    
    /// find the right wheel for a given address
    int address2wheel(int adr) const;

  private:

    L1MuDTSectorProcessor& m_sp;

    edm::ESHandle< L1MuDTTFParameters > pars;

};
  
#endif
