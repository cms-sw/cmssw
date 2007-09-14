#ifndef L1GlobalTrigger_L1GlobalTriggerReadoutSetupFwd_h
#define L1GlobalTrigger_L1GlobalTriggerReadoutSetupFwd_h

/**
 * \class L1GlobalTriggerReadoutSetup
 * 
 * 
 * Description: group typedefs for GT readout record.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

// user include files
//   base class
// forward declarations

/// typedefs

/// algorithm bits: 128 bits
typedef std::vector<bool> DecisionWord;

/// extend DecisionWord with 64 bits
/// need a new FDL chip :-)
typedef std::vector<bool> DecisionWordExtended;

/// technical trigger bits (64 bits)
typedef std::vector<bool> TechnicalTriggerWord;

// muons
typedef unsigned MuonDataWord;

// e-gamma, jet objects
typedef boost::uint16_t CaloDataWord;

// missing Et
typedef boost::uint32_t CaloMissingEtWord;

// twelve jet counts, encoded in five bits per count; six jets per 32-bit word
// code jet count = 31 indicate overflow condition
typedef std::vector<unsigned> CaloJetCountsWord;

/// GT objects
enum L1GtObject { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts };

#endif /*L1GlobalTrigger_L1GlobalTriggerReadoutSetupFwd_h*/
