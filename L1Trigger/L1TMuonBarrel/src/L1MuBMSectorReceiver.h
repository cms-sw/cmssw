//-------------------------------------------------
//
/**  \class L1MuBMSectorReceiver
 *
 *   Sector Receiver:
 *
 *   The Sector Receiver receives track segment
 *   data from the BBMX and CSC chamber triggers
 *   and stores it into the Data Buffer
 *
 *
 *
 *   N. Neumeister            CERN EP
 *   J. Troconiz              UAM Madrid
 */
//
//--------------------------------------------------
#ifndef L1MUBM_SECTOR_RECEIVER_H
#define L1MUBM_SECTOR_RECEIVER_H

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
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"

class L1MuBMSectorProcessor;
class L1MuDTTFParameters;
class L1MuDTTFMasks;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMSectorReceiver {

  public:

    /// constructor
    L1MuBMSectorReceiver(L1MuBMSectorProcessor& , edm::ConsumesCollector&& iC);

    /// destructor
    virtual ~L1MuBMSectorReceiver();

    /// receive track segment data from the BBMX and CSC chamber triggers
    void run(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// clear Sector Receiver
    void reset();

  private:

    /// receive track segment data from BBMX chamber trigger
    void receiveBBMXData(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// receive track segment data from CSC chamber trigger
    void receiveCSCData(int bx, const edm::Event& e, const edm::EventSetup& c);

    /// find the right sector for a given address
    int address2sector(int adr) const;

    /// find the right wheel for a given address
    int address2wheel(int adr) const;

  private:

    L1MuBMSectorProcessor& m_sp;

    edm::ESHandle< L1TMuonBarrelParams > bmtfParamsHandle;
    L1MuDTTFMasks       msks;

    edm::ESHandle< L1MuDTTFParameters > pars;
    //edm::ESHandle< L1MuDTTFMasks >      msks;
    edm::EDGetTokenT<L1MuDTChambPhContainer> m_DTDigiToken;

};

#endif
