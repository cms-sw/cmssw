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
 *   $Date: 2006/06/26 16:11:13 $
 *   $Revision: 1.1 $
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

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
class L1MuDTSectorProcessor;

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
    void run(int bx, const edm::Event& e);
    
    /// clear Sector Receiver
    void reset();

  private:

    /// receive track segment data from DTBX chamber trigger
    void receiveDTBXData(int bx, const edm::Event& e);

    /// receive track segment data from CSC chamber trigger
    void receiveCSCData(int bx, const edm::Event& e);
    
    /// find the right sector for a given address
    int address2sector(int adr) const;
    
    /// find the right wheel for a given address
    int address2wheel(int adr) const;

  private:

    L1MuDTSectorProcessor& m_sp;

};
  
#endif
