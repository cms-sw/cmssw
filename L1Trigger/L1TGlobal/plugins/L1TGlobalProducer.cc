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
#include "L1Trigger/L1TGlobal/interface/TriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "TriggerMenuParser.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/GlobalStableParameters.h"
#include "CondFormats/DataRecord/interface/L1TGlobalStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "L1Trigger/L1TGlobal/plugins/L1TGlobalProducer.h"

using namespace l1t;

// constructors

void L1TGlobalProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // These parameters are part of the L1T/HLT interface, avoid changing if possible::
  desc.add<edm::InputTag> ("MuonInputTag", edm::InputTag(""))->setComment("InputTag for Global Muon Trigger (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("EGammaInputTag", edm::InputTag(""))->setComment("InputTag for Calo Trigger EGamma (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("TauInputTag", edm::InputTag(""))->setComment("InputTag for Calo Trigger Tau (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("JetInputTag", edm::InputTag(""))->setComment("InputTag for Calo Trigger Jet (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("EtSumInputTag", edm::InputTag(""))->setComment("InputTag for Calo Trigger EtSum (required parameter:  default value is invalid)");
  desc.add<edm::InputTag> ("ExtInputTag", edm::InputTag(""))->setComment("InputTag for external conditions (not required, but recommend to specify explicitly in config)");
  desc.add<bool>("AlgorithmTriggersUnprescaled", false)->setComment("not required, but recommend to specify explicitly in config");
  desc.add<bool>("AlgorithmTriggersUnmasked", false)->setComment("not required, but recommend to specify explicitly in config");
  // These parameters have well defined  default values and are not currently 
  // part of the L1T/HLT interface.  They can be cleaned up or updated at will:
  desc.add<bool> ("ProduceL1GtDaqRecord",true);
  desc.add<bool> ("ProduceL1GtObjectMapRecord",true);           
  desc.add<int> ("EmulateBxInEvent",1);
  desc.add<int> ("L1DataBxInEvent",5);
  desc.add<unsigned int> ("AlternativeNrBxBoardDaq",0);
  desc.add<int> ("BstLengthBytes",-1);
  desc.add<unsigned int> ("PrescaleSet",1);    
  desc.addUntracked<int>("Verbosity",0);            
  desc.add<std::string>("TriggerMenuLuminosity","startup");
  desc.add<std::string>("PrescaleCSVFile","prescale_L1TGlobal.csv");
  descriptions.add("L1TGlobalProducer", desc);
}

L1TGlobalProducer::L1TGlobalProducer(const edm::ParameterSet& parSet) :
            m_muInputTag(parSet.getParameter<edm::InputTag> ("MuonInputTag")),
            m_egInputTag(parSet.getParameter<edm::InputTag> ("EGammaInputTag")),
            m_tauInputTag(parSet.getParameter<edm::InputTag> ("TauInputTag")),
            m_jetInputTag(parSet.getParameter<edm::InputTag> ("JetInputTag")),
            m_sumInputTag(parSet.getParameter<edm::InputTag> ("EtSumInputTag")),
	    m_extInputTag(parSet.getParameter<edm::InputTag> ("ExtInputTag")),
            m_produceL1GtDaqRecord(parSet.getParameter<bool> ("ProduceL1GtDaqRecord")),
            m_produceL1GtObjectMapRecord(parSet.getParameter<bool> ("ProduceL1GtObjectMapRecord")),           	    
            m_emulateBxInEvent(parSet.getParameter<int> ("EmulateBxInEvent")),
	    m_L1DataBxInEvent(parSet.getParameter<int> ("L1DataBxInEvent")),
            m_alternativeNrBxBoardDaq(parSet.getParameter<unsigned int> ("AlternativeNrBxBoardDaq")),
            m_psBstLengthBytes(parSet.getParameter<int> ("BstLengthBytes")),
            m_prescaleSet(parSet.getParameter<unsigned int> ("PrescaleSet")),
            m_algorithmTriggersUnprescaled(parSet.getParameter<bool> ("AlgorithmTriggersUnprescaled")),
            m_algorithmTriggersUnmasked(parSet.getParameter<bool> ("AlgorithmTriggersUnmasked")),
            m_verbosity(parSet.getUntrackedParameter<int>("Verbosity")),
            m_isDebugEnabled(edm::isDebugEnabled())
{


  m_egInputToken = consumes <BXVector<EGamma> > (m_egInputTag);
  m_tauInputToken = consumes <BXVector<Tau> > (m_tauInputTag);
  m_jetInputToken = consumes <BXVector<Jet> > (m_jetInputTag);
  m_sumInputToken = consumes <BXVector<EtSum> > (m_sumInputTag);
  m_muInputToken = consumes <BXVector<Muon> > (m_muInputTag);

  m_extInputToken = consumes <BXVector<GlobalExtBlk> > (m_extInputTag);

    if (m_verbosity) {

        LogDebug("l1t|Global") << std::endl;

        LogTrace("l1t|Global")
                << "\nInput tag for muon collection from GMT:         " << m_muInputTag
                << "\nInput tag for eg collection           :         " << m_egInputTag
                << "\nInput tag for tau collection          :         " << m_tauInputTag
                << "\nInput tag for jet collection          :         " << m_jetInputTag
                << "\nInput tag for sum collection          :         " << m_sumInputTag
		<< "\nInput tag for external conditions     :         " << m_extInputTag
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
            edm::LogWarning("L1TGlobalProducer")
                    << "\nWARNING: Number of bunch crossing to be emulated rounded to: "
                    << m_emulateBxInEvent << "\n         The number must be an odd number!\n"
                    << std::endl;
        }
    }


    if ( ( m_L1DataBxInEvent > 0 ) && ( ( m_L1DataBxInEvent % 2 ) == 0 )) {
        m_L1DataBxInEvent = m_L1DataBxInEvent - 1;

        if (m_verbosity) {
            edm::LogWarning("L1TGlobalProducer")
                    << "\nWARNING: Number of bunch crossing for incoming L1 Data rounded to: "
                    << m_L1DataBxInEvent << "\n         The number must be an odd number!\n"
                    << std::endl;
        }
    } else if( m_L1DataBxInEvent<0) {
        m_L1DataBxInEvent = 1;

        if (m_verbosity) {
            edm::LogWarning("L1TGlobalProducer")
                    << "\nWARNING: Number of bunch crossing for incoming L1 Data was changed to: "
                    << m_L1DataBxInEvent << "\n         The number must be an odd positive number!\n"
                    << std::endl;
        }        
    }

    

  
    // register products
    if( m_produceL1GtDaqRecord ){
      produces<GlobalAlgBlkBxCollection>();
      produces<GlobalExtBlkBxCollection>();
    }


    if( m_produceL1GtObjectMapRecord ) {
      produces<L1GlobalTriggerObjectMapRecord>();
    }


    // create new uGt Board
    m_uGtBrd = new GtBoard();
    m_uGtBrd->setVerbosity(m_verbosity);

    // initialize cached IDs

    //
    m_l1GtStableParCacheID = 0ULL;
    m_l1GtMenuCacheID = 0ULL;

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


    // directory in /data/Luminosity for the trigger menu
    std::string menuDir = parSet.getParameter<std::string>("TriggerMenuLuminosity");
    //std::string menuDir = "startup";

    // prescale CSV file file
    std::string prescaleFileName = parSet.getParameter<std::string>("PrescaleCSVFile");

    // def.xml file
    //std::string prescaleFileName = "prescale_L1TGlobal.csv";

    edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/" +
                       menuDir + "/" + prescaleFileName);

    m_prescalesFile = f1.fullPath();

    unsigned int temp_numberPhysTriggers = 512;
 
    // Get prescale factors from CSV file for now
    std::ifstream inputPrescaleFile;
    inputPrescaleFile.open(m_prescalesFile);

    std::vector<std::vector<int> > vec;
    std::vector<std::vector<int> > prescale_vec;

    std::vector<unsigned int> temp_triggerMask;
    std::vector<unsigned int> temp_triggerVetoMask;

    if( inputPrescaleFile ){
      std::string prefix1("#");
      std::string prefix2("-1");

      std::string line; 

      bool first = true;

      while( getline(inputPrescaleFile,line) ){

	if( !line.compare(0, prefix1.size(), prefix1) ) continue;
	//if( !line.compare(0, prefix2.size(), prefix2) ) continue;

	istringstream split(line);
	int value;
	int col = 0;
	char sep;

	while( split >> value ){
	  if( first ){
	    // Each new value read on line 1 should create a new inner vector
	    vec.push_back(std::vector<int>());
	  }

	  vec[col].push_back(value);
	  ++col;

	  // read past the separator
	  split>>sep;
	}

	// Finished reading line 1 and creating as many inner
	// vectors as required
	first = false;
      }


      int NumPrescaleSets = 0;

      int maskColumn = -1;
      int maskVetoColumn = -1;
      for( int iCol=0; iCol<int(vec.size()); iCol++ ){
	if( vec[iCol].size() > 0 ){
	  int firstRow = vec[iCol][0];

	  if( firstRow > 0 ) NumPrescaleSets++;
	  else if( firstRow==-2 ) maskColumn = iCol;
	  else if( firstRow==-3 ) maskVetoColumn = iCol;
	}
      }

      // Fill default values for mask and veto mask
      for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
	unsigned int inputDefaultMask = 1;
	unsigned int inputDefaultVetoMask = 0;
	temp_triggerMask.push_back(inputDefaultMask);
	temp_triggerVetoMask.push_back(inputDefaultVetoMask);
      }

      // Fill non-trivial mask and veto mask
      if( maskColumn>=0 || maskVetoColumn>=0 ){
	for( int iBit=1; iBit<int(vec[0].size()); iBit++ ){
	  unsigned int algoBit = vec[0][iBit];
	  // algoBit must be less than the number of triggers
	  if( algoBit < temp_numberPhysTriggers ){
	    if( maskColumn>=0 ){
	      unsigned int triggerMask = vec[maskColumn][iBit];
	      temp_triggerMask[algoBit] = triggerMask;
	    }
	    if( maskVetoColumn>=0 ){
	      unsigned int triggerVetoMask = vec[maskVetoColumn][iBit];
	      temp_triggerVetoMask[algoBit] = triggerVetoMask;
	    }
	  }
	}
      }


      if( NumPrescaleSets > 0 ){
	// Fill default prescale set
	for( int iSet=0; iSet<NumPrescaleSets; iSet++ ){
	  prescale_vec.push_back(std::vector<int>());
	  for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
	    int inputDefaultPrescale = 1;
	    prescale_vec[iSet].push_back(inputDefaultPrescale);
	  }
	}

	// Fill non-trivial prescale set
	for( int iBit=1; iBit<int(vec[0].size()); iBit++ ){
	  unsigned int algoBit = vec[0][iBit];
	  // algoBit must be less than the number of triggers
	  if( algoBit < temp_numberPhysTriggers ){
	    for( int iSet=0; iSet<int(vec.size()); iSet++ ){
	      int useSet = -1;
	      if( vec[iSet].size() > 0 ){
		useSet = vec[iSet][0];
	      }
	      useSet -= 1;
	      
	      if( useSet<0 ) continue;

	      int prescale = vec[iSet][iBit];
	      prescale_vec[useSet][algoBit] = prescale;
	    }
	  }
	  else{
	    LogTrace("l1t|Global")
	      << "\nPrescale file has algo bit: " << algoBit
	      << "\nThis is larger than the number of triggers: " << m_numberPhysTriggers
	      << "\nSomething is wrong. Ignoring."
	      << std::endl;
	  }
	}
      }

    }
    else {
      LogTrace("l1t|Global")
	<< "\nCould not find file: " << m_prescalesFile
	<< "\nFilling the prescale vectors with prescale 1"
	<< "\nSetting prescale set to 1"
	<< std::endl;

      m_prescaleSet = 1;

      for( int col=0; col < 1; col++ ){
	prescale_vec.push_back(std::vector<int>());
	for( unsigned int iBit = 0; iBit < temp_numberPhysTriggers; ++iBit ){
	  int inputDefaultPrescale = 1;
	  prescale_vec[col].push_back(inputDefaultPrescale);
	}
      }
    }

    inputPrescaleFile.close();

    m_initialPrescaleFactorsAlgoTrig = prescale_vec;

    m_initialTriggerMaskAlgoTrig = temp_triggerMask;
    m_initialTriggerMaskVetoAlgoTrig = temp_triggerVetoMask;

}

// destructor
L1TGlobalProducer::~L1TGlobalProducer()
{

    delete m_uGtBrd;

}

// member functions

// method called to produce the data
void L1TGlobalProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup)
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

    if (m_emulateBxInEvent < 1) m_emulateBxInEvent=1;
    if (m_L1DataBxInEvent < 1) m_L1DataBxInEvent=1;

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


    // get / update the trigger menu from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1GtMenuCacheID = evSetup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();

    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {

        edm::ESHandle<L1TUtmTriggerMenu> l1GtMenu;
        evSetup.get< L1TUtmTriggerMenuRcd>().get(l1GtMenu) ;
        const L1TUtmTriggerMenu* utml1GtMenu =  l1GtMenu.product();
        
	// Instantiate Parser
        TriggerMenuParser gtParser = TriggerMenuParser();   


	gtParser.setGtNumberConditionChips(m_l1GtStablePar->gtNumberConditionChips());
	gtParser.setGtPinsOnConditionChip(m_l1GtStablePar->gtPinsOnConditionChip());
	gtParser.setGtOrderConditionChip(m_l1GtStablePar->gtOrderConditionChip());
	gtParser.setGtNumberPhysTriggers(m_l1GtStablePar->gtNumberPhysTriggers());
        
	//Parse menu into emulator classes
	gtParser.parseCondFormats(utml1GtMenu); 
        
    // transfer the condition map and algorithm map from parser to L1uGtTriggerMenu
        m_l1GtMenu  =  new TriggerMenu(gtParser.gtTriggerMenuName(), m_l1GtStablePar->gtNumberConditionChips(),
                        gtParser.vecMuonTemplate(),
                        gtParser.vecCaloTemplate(),
                        gtParser.vecEnergySumTemplate(),
                        gtParser.vecExternalTemplate(),
                        gtParser.vecCorrelationTemplate(),
                        gtParser.corMuonTemplate(),
                        gtParser.corCaloTemplate(),
                        gtParser.corEnergySumTemplate()) ;

 
	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtTriggerMenuInterface(gtParser.gtTriggerMenuInterface());
	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtTriggerMenuImplementation(gtParser.gtTriggerMenuImplementation());
	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtScaleDbKey(gtParser.gtScaleDbKey());
	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtScales(gtParser.gtScales());

	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtAlgorithmMap(gtParser.gtAlgorithmMap());
	(const_cast<TriggerMenu*>(m_l1GtMenu))->setGtAlgorithmAliasMap(gtParser.gtAlgorithmAliasMap());	        

        (const_cast<TriggerMenu*>(m_l1GtMenu))->buildGtConditionMap();

        m_l1GtMenuCacheID = l1GtMenuCacheID;
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
    

    // Set Prescale factors to initial 
    m_prescaleFactorsAlgoTrig = &m_initialPrescaleFactorsAlgoTrig;
    m_triggerMaskAlgoTrig = &m_initialTriggerMaskAlgoTrig;
    m_triggerMaskVetoAlgoTrig = &m_initialTriggerMaskVetoAlgoTrig;


    // // Used for testing
    // for( int iL1Set=0; iL1Set<int(m_prescaleFactorsAlgoTrig->size()); iL1Set++ ){
    //   if( m_prescaleFactorsAlgoTrig->size()>0 ){
    // 	const std::vector<int>& testPrescaleSet = (*m_prescaleFactorsAlgoTrig).at(iL1Set);

    // 	printf(" iL1Set=%4d", iL1Set);
    // 	for( int iPrescale=0; iPrescale<int(testPrescaleSet.size()); iPrescale++ ){
    // 	  printf(", %2d", testPrescaleSet[iPrescale]);
    // 	}
    // 	printf("\n");
    //   }
    // }


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
    bool receiveExt = true;

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


    // get the prescale factor from the configuration for now
    // Prescale set indexed by zero internally, but externally indexed by 1
    unsigned int pfAlgoSetIndex = m_prescaleSet-1;

    // Require that prescale set be positive
    if( m_prescaleSet<=0 ) pfAlgoSetIndex = 0;

    if( pfAlgoSetIndex > (*m_prescaleFactorsAlgoTrig).size()-1 ){
      LogTrace("l1t|Global")
	<< "\nAttempting to access prescale algo set: " << m_prescaleSet
	<< "\nNumber of prescale algo sets available: " << (*m_prescaleFactorsAlgoTrig).size()
	<< "Setting former to latter."
	<< std::endl;

      pfAlgoSetIndex = (*m_prescaleFactorsAlgoTrig).size()-1;
    }

    const std::vector<int>& prescaleFactorsAlgoTrig = (*m_prescaleFactorsAlgoTrig).at(pfAlgoSetIndex);

    const std::vector<unsigned int>& triggerMaskAlgoTrig = *m_triggerMaskAlgoTrig;
    const std::vector<unsigned int>& triggerMaskVetoAlgoTrig = *m_triggerMaskVetoAlgoTrig;

    LogDebug("l1t|Global") << "Size of prescale vector" << prescaleFactorsAlgoTrig.size() << std::endl;


// Load the calorimeter input onto the uGt Board
     m_uGtBrd->receiveCaloObjectData(iEvent, 
				     m_egInputToken,
				     m_tauInputToken,
				     m_jetInputToken,
				     m_sumInputToken,
        			     receiveEG, m_nrL1EG,
        			     receiveTau, m_nrL1Tau,				     
        			     receiveJet, m_nrL1Jet,
        			     receiveEtSums     );

     m_uGtBrd->receiveMuonObjectData(iEvent, m_muInputToken,
                                     receiveMu, m_nrL1Mu  );

     m_uGtBrd->receiveExternalData(iEvent, m_extInputToken,
                                     receiveExt  );


    // loop over BxInEvent
    for (int iBxInEvent = minEmulBxInEvent; iBxInEvent <= maxEmulBxInEvent;
            ++iBxInEvent) {

        //  run GTL
        LogDebug("l1t|Global")
         << "\nGtProducer : running GTL  for bx = " << iBxInEvent << "\n"
         << std::endl;


//  Run the GTL for this BX
        m_uGtBrd->runGTL(iEvent, evSetup, m_l1GtMenu,
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
			  m_totalBxInEvent,
			  m_numberPhysTriggers,
			  prescaleFactorsAlgoTrig,
			  triggerMaskAlgoTrig,
			  triggerMaskVetoAlgoTrig,
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
        << "Test gtObjectMapRecord in L1TGlobalProducer \n\n" << myCoutStream.str() << "\n\n"
        << std::endl;

        myCoutStream.str("");
        myCoutStream.clear();
*/
    }



    
    // register products
    if( m_produceL1GtDaqRecord ){
      iEvent.put( uGtAlgRecord );
      iEvent.put( uGtExtRecord );
    }


    if( m_produceL1GtObjectMapRecord ){
      iEvent.put( gtObjectMapRecord );
    }


}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TGlobalProducer);
