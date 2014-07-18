/**
 * \class GtProducer
 *
 *
 * Description: see header file.
 *
 *   Based off legacy code written by Vasile Mihai Ghete - HEPHY Vienna
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author:   Brian Winer - Ohio State
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1Trigger/L1TGlobal/plugins/GtProducer.h"

// system include files
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <boost/cstdint.hpp>

// user include files
#include "FWCore/Utilities/interface/typedefs.h"

// Objects to produce for the output record.
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/GlobalStableParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"


#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"



// constructors

l1t::GtProducer::GtProducer(const edm::ParameterSet& parSet) :
            m_muInputTag(parSet.getParameter<edm::InputTag> ("GmtInputTag")),
            m_caloInputTag(parSet.getParameter<edm::InputTag> ("caloInputTag")),

            m_produceL1GtDaqRecord(parSet.getParameter<bool> ("ProduceL1GtDaqRecord")),
            m_produceL1GtObjectMapRecord(parSet.getParameter<bool> ("ProduceL1GtObjectMapRecord")),           
	    
            m_emulateBxInEvent(parSet.getParameter<int> ("EmulateBxInEvent")),
	    m_L1DataBxInEvent(parSet.getParameter<int> ("L1DataBxInEvent")),

            m_alternativeNrBxBoardDaq(parSet.getParameter<unsigned int> ("AlternativeNrBxBoardDaq")),
            m_psBstLengthBytes(parSet.getParameter<int> ("BstLengthBytes")),
            m_algorithmTriggersUnprescaled(parSet.getParameter<bool> ("AlgorithmTriggersUnprescaled")),
            m_algorithmTriggersUnmasked(parSet.getParameter<bool> ("AlgorithmTriggersUnmasked")),

            m_verbosity(parSet.getUntrackedParameter<int>("Verbosity", 0)),
            m_isDebugEnabled(edm::isDebugEnabled())
{

  
    if (m_verbosity) {

        LogDebug("l1t|Global") << std::endl;

        LogTrace("l1t|Global")
                << "\nInput tag for muon collection from GMT:         " << m_muInputTag
                << "\nInput tag for calorimeter collections from GCT: " << m_caloInputTag
                << std::endl;


        LogTrace("l1t|Global")
                << "\nProduce the L1 GT DAQ readout record:           " << m_produceL1GtDaqRecord
                << "\nProduce the L1 GT Object Map record:            " << m_produceL1GtObjectMapRecord
                << " \n"
                << "\nWrite Psb content to L1 GT DAQ Record:          " << m_writePsbL1GtDaqRecord
                << " \n"
                << "\nNumber of BxInEvent to be emulated:             " << m_emulateBxInEvent
                << " \n"
                << "\nAlternative for number of BX in GT DAQ record:   0x" << std::hex
                << m_alternativeNrBxBoardDaq
                << " \n"
                << "\nLength of BST message [bytes]:                  " << m_psBstLengthBytes
                << "\n"
                << "\nRun algorithm triggers unprescaled:             " << m_algorithmTriggersUnprescaled
                << "\nRun algorithm triggers unmasked (all enabled):  " << m_algorithmTriggersUnmasked
                << "\n"
                << std::endl;
    }


    if ( ( m_emulateBxInEvent > 0 ) && ( ( m_emulateBxInEvent % 2 ) == 0 )) {
        m_emulateBxInEvent = m_emulateBxInEvent - 1;

        if (m_verbosity) {
            edm::LogWarning("GtProducer")
                    << "\nWARNING: Number of bunch crossing to be emulated rounded to: "
                    << m_emulateBxInEvent << "\n         The number must be an odd number!\n"
                    << std::endl;
        }
    }


    if ( ( m_L1DataBxInEvent > 0 ) && ( ( m_L1DataBxInEvent % 2 ) == 0 )) {
        m_L1DataBxInEvent = m_L1DataBxInEvent - 1;

        if (m_verbosity) {
            edm::LogWarning("GtProducer")
                    << "\nWARNING: Number of bunch crossing for incoming L1 Data rounded to: "
                    << m_L1DataBxInEvent << "\n         The number must be an odd number!\n"
                    << std::endl;
        }
    } else if( m_L1DataBxInEvent<0) {
        m_L1DataBxInEvent = 1;

        if (m_verbosity) {
            edm::LogWarning("GtProducer")
                    << "\nWARNING: Number of bunch crossing for incoming L1 Data was changed to: "
                    << m_L1DataBxInEvent << "\n         The number must be an odd positive number!\n"
                    << std::endl;
        }        
    }

    

  
    // register products
    if (m_produceL1GtDaqRecord) {
	produces<GlobalAlgBlkBxCollection>();
	produces<GlobalExtBlkBxCollection>();
    }

/*  **** Needs Modifying ***
    if (m_produceL1GtObjectMapRecord) {
        produces<GtProducerObjectMapRecord>();
    }
*/


    // create new uGt Board
    m_uGtBrd = new GtBoard();
    m_uGtBrd->setVerbosity(m_verbosity);

    // initialize cached IDs

    //
    m_l1GtStableParCacheID = 0ULL;

    m_numberPhysTriggers = 0;
    m_numberDaqPartitions = 0;

    m_nrL1Mu = 0;
    m_nrL1EG = 0;
    m_nrL1Tau = 0;

    m_nrL1Jet = 0;


    m_nrL1JetCounts = 0;

    m_ifMuEtaNumberBits = 0;
    m_ifCaloEtaNumberBits = 0;

    //
    m_l1GtParCacheID = 0ULL;

    m_totalBxInEvent = 0;

    m_activeBoardsGtDaq = 0;
    m_bstLengthBytes = 0;

    //
    m_l1GtBMCacheID = 0ULL;

    //
    m_l1GtPfAlgoCacheID = 0ULL;

    m_l1GtTmAlgoCacheID = 0ULL;

    m_l1GtTmVetoAlgoCacheID = 0ULL;

}

// destructor
l1t::GtProducer::~GtProducer()
{

    delete m_uGtBrd;

}

// member functions

// method called to produce the data
void l1t::GtProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
{

    // get / update the parameters from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtParCacheID = evSetup.get<L1GtParametersRcd>().cacheIdentifier();

    if (m_l1GtParCacheID != l1GtParCacheID) {

        edm::ESHandle< L1GtParameters > l1GtPar;
        evSetup.get< L1GtParametersRcd >().get( l1GtPar );
        m_l1GtPar = l1GtPar.product();

        //    total number of Bx's in the event coming from EventSetup
        m_totalBxInEvent = m_l1GtPar->gtTotalBxInEvent();

        //    active boards in L1 GT DAQ record and in L1 GT EVM record
        m_activeBoardsGtDaq = m_l1GtPar->gtDaqActiveBoards();

        ///   length of BST message (in bytes) for L1 GT EVM record
        m_bstLengthBytes = m_l1GtPar->gtBstLengthBytes();


        m_l1GtParCacheID = l1GtParCacheID;

    }

    // negative value: emulate TotalBxInEvent as given in EventSetup
    if (m_emulateBxInEvent < 0) {
        m_emulateBxInEvent = m_totalBxInEvent;
    }

    int minEmulBxInEvent = (m_emulateBxInEvent + 1)/2 - m_emulateBxInEvent;
    int maxEmulBxInEvent = (m_emulateBxInEvent + 1)/2 - 1;

    int minL1DataBxInEvent = (m_L1DataBxInEvent + 1)/2 - m_L1DataBxInEvent;
    int maxL1DataBxInEvent = (m_L1DataBxInEvent + 1)/2 - 1;

    // process event iEvent
    // get / update the stable parameters from the EventSetup
    // local cache & check on cacheIdentifier

    unsigned long long l1GtStableParCacheID =
            evSetup.get<L1TGlobalStableParametersRcd>().cacheIdentifier();

    if (m_l1GtStableParCacheID != l1GtStableParCacheID) {

        edm::ESHandle< GlobalStableParameters > l1GtStablePar;
        evSetup.get< L1TGlobalStableParametersRcd >().get( l1GtStablePar );
        m_l1GtStablePar = l1GtStablePar.product();

        // number of physics triggers
        m_numberPhysTriggers = m_l1GtStablePar->gtNumberPhysTriggers();

        // number of DAQ partitions
        m_numberDaqPartitions = 8; // FIXME add it to stable parameters

        // number of objects of each type
        m_nrL1Mu = static_cast<int> (m_l1GtStablePar->gtNumberL1Mu());
        //m_nrL1Mu = static_cast<int> (8);
	
// ***** Doe we need to change the StablePar class for generic. EG	
        m_nrL1EG = static_cast<int> (m_l1GtStablePar->gtNumberL1NoIsoEG());
        m_nrL1Tau= static_cast<int> (m_l1GtStablePar->gtNumberL1TauJet());


// ********* Do we need to change the StablePar class for generic jet?
        m_nrL1Jet = static_cast<int> (m_l1GtStablePar->gtNumberL1CenJet());

        m_nrL1JetCounts = static_cast<int> (m_l1GtStablePar->gtNumberL1JetCounts());

        // ... the rest of the objects are global

        m_ifMuEtaNumberBits = static_cast<int> (m_l1GtStablePar->gtIfMuEtaNumberBits());
        m_ifCaloEtaNumberBits = static_cast<int> (m_l1GtStablePar->gtIfCaloEtaNumberBits());


        // Initialize Board
        m_uGtBrd->init(m_numberPhysTriggers, m_nrL1Mu, m_nrL1EG, m_nrL1Tau, m_nrL1Jet, minL1DataBxInEvent, maxL1DataBxInEvent );

        //
        m_l1GtStableParCacheID = l1GtStableParCacheID;

    }


    // get / update the board maps from the EventSetup
    // local cache & check on cacheIdentifier

/*   *** Drop L1GtBoard Maps for now
    typedef std::vector<L1GtBoard>::const_iterator CItBoardMaps;

    unsigned long long l1GtBMCacheID = evSetup.get<L1GtBoardMapsRcd>().cacheIdentifier();
*/

/*  ** Drop board mapping for now
    if (m_l1GtBMCacheID != l1GtBMCacheID) {

        edm::ESHandle< L1GtBoardMaps > l1GtBM;
        evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM );
        m_l1GtBM = l1GtBM.product();

        m_l1GtBMCacheID = l1GtBMCacheID;

    }
   

    // TODO need changes in CondFormats to cache the maps
    const std::vector<L1GtBoard>& boardMaps = m_l1GtBM->gtBoardMaps();
*/
    // get / update the prescale factors from the EventSetup
    // local cache & check on cacheIdentifier


/*  **** For Now Leave out Prescale Factors ****
    unsigned long long l1GtPfAlgoCacheID =
        evSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {

        edm::ESHandle< L1GtPrescaleFactors > l1GtPfAlgo;
        evSetup.get< L1GtPrescaleFactorsAlgoTrigRcd >().get( l1GtPfAlgo );
        m_l1GtPfAlgo = l1GtPfAlgo.product();

        m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());

        m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;

    }
*/
    

    // get / update the trigger mask from the EventSetup
    // local cache & check on cacheIdentifier


/*  **** For now Leave out Masks  *****
    unsigned long long l1GtTmAlgoCacheID =
        evSetup.get<L1GtTriggerMaskAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmAlgoCacheID != l1GtTmAlgoCacheID) {

        edm::ESHandle< L1GtTriggerMask > l1GtTmAlgo;
        evSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( l1GtTmAlgo );
        m_l1GtTmAlgo = l1GtTmAlgo.product();

        m_triggerMaskAlgoTrig = m_l1GtTmAlgo->gtTriggerMask();

        m_l1GtTmAlgoCacheID = l1GtTmAlgoCacheID;

    }
*/



/*  **** For now Leave out Veto Masks  *****
    unsigned long long l1GtTmVetoAlgoCacheID =
        evSetup.get<L1GtTriggerMaskVetoAlgoTrigRcd>().cacheIdentifier();

    if (m_l1GtTmVetoAlgoCacheID != l1GtTmVetoAlgoCacheID) {

        edm::ESHandle< L1GtTriggerMask > l1GtTmVetoAlgo;
        evSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd >().get( l1GtTmVetoAlgo );
        m_l1GtTmVetoAlgo = l1GtTmVetoAlgo.product();

        m_triggerMaskVetoAlgoTrig = m_l1GtTmVetoAlgo->gtTriggerMask();

        m_l1GtTmVetoAlgoCacheID = l1GtTmVetoAlgoCacheID;

    }
*/



// ******  Board Maps Need to be redone....hard code for now ******
    // loop over blocks in the GT DAQ record receiving data, count them if they are active
    // all board type are defined in CondFormats/L1TObjects/L1GtFwd
    // enum L1GtBoardType { GTFE, FDL, PSB, GMT, TCS, TIM };
    // &
    // set the active flag for each object type received from GMT and GCT
    // all objects in the GT system are defined in enum L1GtObject from
    // DataFormats/L1Trigger/GtProducerReadoutSetupFwd

    //
    bool receiveMu = true;
    bool receiveEG = true;
    bool receiveTau = true;    
    bool receiveJet = true;
    bool receiveEtSums = true;

/*  *** Boards need redefining *****
    for (CItBoardMaps
            itBoard = boardMaps.begin();
            itBoard != boardMaps.end(); ++itBoard) {

        int iPosition = itBoard->gtPositionDaqRecord();
        if (iPosition > 0) {

            int iActiveBit = itBoard->gtBitDaqActiveBoards();
            bool activeBoard = false;

            if (iActiveBit >= 0) {
                activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
            }

            // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
            //               in the record and ActiveBoardsMap, and active
            if ((iActiveBit < 0) || activeBoard) {

// ******  Decide what board manipulation (if any we want here)

            }
        }

    }
*/



/* *** No Output Record for Now
    // produce the GtProducerReadoutRecord now, after we found how many
    // BxInEvent the record has and how many boards are active
    std::auto_ptr<GtProducerReadoutRecord> gtDaqReadoutRecord(
        new GtProducerReadoutRecord(
            m_emulateBxInEvent, daqNrFdlBoards, daqNrPsbBoards) );

*/

    // Produce the Output Records for the GT
    std::auto_ptr<GlobalAlgBlkBxCollection> uGtAlgRecord( new GlobalAlgBlkBxCollection(0,minEmulBxInEvent,maxEmulBxInEvent));
    std::auto_ptr<GlobalExtBlkBxCollection> uGtExtRecord( new GlobalExtBlkBxCollection(0,minEmulBxInEvent,maxEmulBxInEvent));

    // * produce the L1GlobalTriggerObjectMapRecord
    std::auto_ptr<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord(
        new L1GlobalTriggerObjectMapRecord() );


    // fill the boards not depending on the BxInEvent in the L1 GT DAQ record
    // GMT, PSB and FDL depend on BxInEvent

    // fill in emulator the same bunch crossing (12 bits - hardwired number of bits...)
    // and the same local bunch crossing for all boards
    int bxCross = iEvent.bunchCrossing();
    boost::uint16_t bxCrossHw = 0;
    if ((bxCross & 0xFFF) == bxCross) {
        bxCrossHw = static_cast<boost::uint16_t> (bxCross);
    }
    else {
        bxCrossHw = 0; // Bx number too large, set to 0!
        if (m_verbosity) {

            LogDebug("l1t|Global")
                << "\nBunch cross number [hex] = " << std::hex << bxCross
                << "\n  larger than 12 bits. Set to 0! \n" << std::dec
                << std::endl;
        }
    }
    LogDebug("l1t|Global") << "HW BxCross " << bxCrossHw << std::endl;  

/*  ** No Record for Now 
    if (m_produceL1GtDaqRecord) {

        for (CItBoardMaps
                itBoard = boardMaps.begin();
                itBoard != boardMaps.end(); ++itBoard) {

            int iPosition = itBoard->gtPositionDaqRecord();
            if (iPosition > 0) {

                int iActiveBit = itBoard->gtBitDaqActiveBoards();
                bool activeBoard = false;

                if (iActiveBit >= 0) {
                    activeBoard = m_activeBoardsGtDaq & (1 << iActiveBit);
                }

                // use board if: in the record, but not in ActiveBoardsMap (iActiveBit < 0)
                //               in the record and ActiveBoardsMap, and active
                if ((iActiveBit < 0) || activeBoard) {

                    switch (itBoard->gtBoardType()) {

                        case GTFE: {
                                L1GtfeWord gtfeWordValue;

                                gtfeWordValue.setBoardId( itBoard->gtBoardId() );

                                // cast int to boost::uint16_t
                                // there are normally 3 or 5 BxInEvent
                                gtfeWordValue.setRecordLength(
                                    static_cast<boost::uint16_t>(recordLength0));

                                gtfeWordValue.setRecordLength1(
                                    static_cast<boost::uint16_t>(recordLength1));

                                // bunch crossing
                                gtfeWordValue.setBxNr(bxCrossHw);

                                // set the list of active boards
                                gtfeWordValue.setActiveBoards(m_activeBoardsGtDaq);

                                // set alternative for number of BX per board
                                gtfeWordValue.setAltNrBxBoard(
                                    static_cast<boost::uint16_t> (m_alternativeNrBxBoardDaq));

                                // set the TOTAL_TRIGNR as read from iEvent
                                // TODO check again - PTC stuff

                                gtfeWordValue.setTotalTriggerNr(
                                    static_cast<boost::uint32_t>(iEvent.id().event()));

                                // ** fill L1GtfeWord in GT DAQ record

                                gtDaqReadoutRecord->setGtfeWord(gtfeWordValue);
                            }

                            break;
                        case TCS: {
                                // nothing
                            }

                            break;
                        case TIM: {
                                // nothing
                            }

                            break;
                        default: {
                                // do nothing, all blocks are given in GtBoardType enum
                            }

                            break;
                    }
                }
            }

        }

    }
*/


    // get the prescale factor set used in the actual luminosity segment
//     int pfAlgoSetIndex = 0; // FIXME
  // comment out for now DMP
//     const std::vector<int>& prescaleFactorsAlgoTrig =
//         (*m_prescaleFactorsAlgoTrig).at(pfAlgoSetIndex);
    
//     LogDebug("l1t|Global") << "Size of prescale vector" << prescaleFactorsAlgoTrig.size() << std::endl;
    //


// Load the calorimeter input onto the uGt Board
     m_uGtBrd->receiveCaloObjectData(iEvent, m_caloInputTag,
        			     receiveEG, m_nrL1EG,
        			     receiveTau, m_nrL1Tau,				     
        			     receiveJet, m_nrL1Jet,
        			     receiveEtSums     );

     m_uGtBrd->receiveMuonObjectData(iEvent, m_muInputTag,
                                     receiveMu, m_nrL1Mu  );


    // loop over BxInEvent
    for (int iBxInEvent = minEmulBxInEvent; iBxInEvent <= maxEmulBxInEvent;
            ++iBxInEvent) {

        //  run GTL
        LogDebug("l1t|Global")
         << "\nGtProducer : running GTL  for bx = " << iBxInEvent << "\n"
         << std::endl;


//  Run the GTL for this BX
        m_uGtBrd->runGTL(iEvent, evSetup, 
            m_produceL1GtObjectMapRecord, iBxInEvent, gtObjectMapRecord,
            m_numberPhysTriggers,
            m_nrL1Mu,
            m_nrL1EG,
	    m_nrL1Tau,
            m_nrL1Jet,
	    m_nrL1JetCounts  );


        //  run FDL
        LogDebug("l1t|Global")
          << "\nGtProducer : running FDL for bx = " << iBxInEvent << "\n"
          << std::endl;

 
//  Run the Final Decision Logic for this BX
	 m_uGtBrd->runFDL(iEvent,
		          iBxInEvent,
		          m_algorithmTriggersUnprescaled,
		          m_algorithmTriggersUnmasked
		          );



// Fill in the DAQ Records
        if (m_produceL1GtDaqRecord) {

            // These need to be defined elsewhere
	    cms_uint64_t orbNr = iEvent.orbitNumber();
	    int abBx = iEvent.bunchCrossing();
            m_uGtBrd->fillAlgRecord(iBxInEvent, uGtAlgRecord, orbNr, abBx);
	    m_uGtBrd->fillExtRecord(iBxInEvent, uGtExtRecord, orbNr, abBx);
        }



    } //End Loop over Bx


    // Add explicit reset of Board
    m_uGtBrd->reset();



    if ( m_verbosity && m_isDebugEnabled ) {

    	std::ostringstream myCoutStream;

       for(int bx=minEmulBxInEvent; bx<maxEmulBxInEvent; bx++) {
        
	   /// Needs error checking that something exists at this bx.
	   (uGtAlgRecord->at(bx,0)).print(myCoutStream); 
	   (uGtExtRecord->at(bx,0)).print(myCoutStream);   
                
       }

        LogTrace("l1t|Global")
        << "\n The following L1 GT DAQ readout record was produced:\n"
        << myCoutStream.str() << "\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();
/*
        const std::vector<L1GlobalTriggerObjectMap> objMapVec =
            gtObjectMapRecord->gtObjectMap();

        for (std::vector<L1GlobalTriggerObjectMap>::const_iterator
                it = objMapVec.begin(); it != objMapVec.end(); ++it) {

            (*it).print(myCoutStream);

        }


        LogDebug("l1t|Global")
        << "Test gtObjectMapRecord in GtProducer \n\n" << myCoutStream.str() << "\n\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();
*/
    }



    
    // register products
    if (m_produceL1GtDaqRecord) {
	iEvent.put( uGtAlgRecord );
	iEvent.put( uGtExtRecord );
    }

/**  OUTPUT RECORD
    if (m_produceL1GtObjectMapRecord) {
        iEvent.put( gtObjectMapRecord );
    }
*/


}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(l1t::GtProducer);
