#ifndef HLTfilters_HLTBeamModeFilter_h
#define HLTfilters_HLTBeamModeFilter_h

/**
 * \class HLTBeamModeFilter
 *
 *
 * Description: filter according to the beam mode using the BST information received by L1 GT.
 *
 * Implementation:
 *    This class is an HLTFilter (-> EDFilter). It implements:
 *      - filter according to the beam mode using the BST information received by L1 GT
 *      - the code for a given mode is the code given in the BST message document
 *        (LHC-BOB-ES-0001-20-00.pdf or newer version)
 *      - it requires as input the unpacked L1GlobalTriggerEvmReadoutRecord
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <vector>

// user include files

//   base class
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

// class declaration
class HLTBeamModeFilter: public HLTFilter {

public:

    /// constructor
    explicit HLTBeamModeFilter(const edm::ParameterSet&);

    /// destructor
    virtual ~HLTBeamModeFilter();

    /// filter the event
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);

private:

    /// InputTag for the L1 Global Trigger EVM readout record
    edm::InputTag m_l1GtEvmReadoutRecordTag;

    /// vector of beam modes (coded as integer numbers)
    std::vector<unsigned int> m_allowedBeamMode;

    /// cache edm::isDebugEnabled()
    bool m_isDebugEnabled;

};

#endif // HLTfilters_HLTBeamModeFilter_h
