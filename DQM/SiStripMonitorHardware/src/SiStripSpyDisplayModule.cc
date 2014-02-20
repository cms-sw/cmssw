/* \file SiStripSpyDisplayModule.cc
 * \brief File containing code for the SiStripMonitorFEDProcessing plugin module.
 */
// -*- C++ -*-
//
// Package:    SiStripMonitorHardware
// Class:      SiStripSpyDisplayModule
// 
// Standard
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

// Framework include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Needed for the SST cabling
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

// Needed for the pedestal values
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

// Needed for the noise values
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

// For translating between FED key and det ID
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

// Needed for the FED raw data processing
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferGenerator.h"

// #include "EventFilter/SiStripRawToDigi/interface/SiStripDigiToRaw.h"

//for cabling
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"

// For plotting
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1S.h"
#include "TH1D.h"

//
// constants, enums and typedefs
//
enum FEDSpyHistogramType {SCOPE_MODE,
                          PAYLOAD_RAW,
                          REORDERED_PAYLOAD_RAW,
                          REORDERED_MODULE_RAW,
                          PEDESTAL_VALUES,
                          NOISE_VALUES,
                          POST_PEDESTAL,
                          POST_COMMON_MODE,
                          ZERO_SUPPRESSED_PADDED,
                          ZERO_SUPPRESSED,
                          VR_COMP,
                          ZERO_SUPPRESSED_COMP};

//
// class declaration
//


/*! \brief EDAnalyzer for the online monitoring of the FED using STT spy channel data.
 *
 * \author Tom Whyntie
 * \date Autumn 2009
 * 
 * See https://twiki.cern.ch/twiki/bin/view/CMS/SiStripSpyDisplayModule for
 * further code documentation, and
 * https://twiki.cern.ch/twiki/bin/view/CMS/FEDSpyChannelMonitoring
 * for more information about the spy channel monitoring project.
 *
 */
class SiStripSpyDisplayModule : public edm::EDAnalyzer {
  public:
    explicit SiStripSpyDisplayModule(const edm::ParameterSet&);
    ~SiStripSpyDisplayModule();

  private:
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void beginJob() override ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override ;

  Bool_t MakeRawDigiHist_(const edm::Handle< edm::DetSetVector<SiStripRawDigi> > & digi_handle,
                            uint32_t specifier,
                            const TFileDirectory & dir,
                            FEDSpyHistogramType type);  
                          
  Bool_t MakeProcessedRawDigiHist_(const edm::Handle< edm::DetSetVector<SiStripProcessedRawDigi> > & digi_handle,
				   uint32_t specifier,
				   const TFileDirectory & dir,
				   FEDSpyHistogramType type);                            

    Bool_t MakeDigiHist_(   const edm::Handle< edm::DetSetVector<SiStripDigi> > & digi_handle,
                            uint32_t detID,
                            const TFileDirectory & dir,
                            FEDSpyHistogramType type);                            

    // ----------member data ---------------------------
    std::vector<uint32_t>            detIDs_;         //!< Vector of detIDs that are of interest (config-specified).
  //now from utility class
  //    edm::ESHandle<SiStripDetCabling> cabling_;        //!< The Strip Tracker cabling object.
  sistrip::SpyUtilities utility_;
  
    // Data input labels
    //===================
    edm::InputTag inputScopeModeRawDigiLabel_; //!< Label for the scope-mode RawDigi collection input tag
    edm::InputTag inputPayloadRawDigiLabel_; //!< Label for the virgin raw RawDigi collection input tag.
    edm::InputTag inputReorderedPayloadRawDigiLabel_; //!< Label for the re-ordered RawDigi module input tag.
    edm::InputTag inputReorderedModuleRawDigiLabel_; //!< Label for the re-ordered RawDigi module input tag.
    edm::InputTag inputPedestalsLabel_;               //!< Label for the pedestals.
    edm::InputTag inputNoisesLabel_;               //!< Label for the noises.
    edm::InputTag inputPostPedestalRawDigiLabel_; //!< Label for the post-pedestal subtraction RawDigi module input tag.
    edm::InputTag inputPostCMRawDigiLabel_; //!< Label for the post-common mode subtraction RawDigi module input tag.
    edm::InputTag inputZeroSuppressedRawDigiLabel_; //!< Label for the zero-suppressed, zero-padded RawDigi module input tag.
    edm::InputTag inputZeroSuppressedDigiLabel_;    //!< Guess what? It's the input label for the zero-suppressed digi
    edm::InputTag inputCompVirginRawDigiLabel_;    //!< VR RawDigis to compare (from mainline)
    edm::InputTag inputCompZeroSuppressedDigiLabel_;    //!< Zero-suppressed digis to compare (from mainline)

    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputScopeModeRawDigiToken_; //!< Token for the scope-mode RawDigi collection input tag
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputPayloadRawDigiToken_; //!< Token for the virgin raw RawDigi collection input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputReorderedPayloadRawDigiToken_; //!< Token for the re-ordered RawDigi module input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputReorderedModuleRawDigiToken_; //!< Token for the re-ordered RawDigi module input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputPedestalsToken_;               //!< Token for the pedestals.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputNoisesToken_;               //!< Token for the noises.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputPostPedestalRawDigiToken_; //!< Token for the post-pedestal subtraction RawDigi module input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputPostCMRawDigiToken_; //!< Token for the post-common mode subtraction RawDigi module input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputZeroSuppressedRawDigiToken_; //!< Token for the zero-suppressed, zero-padded RawDigi module input tag.
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputZeroSuppressedDigiToken_;    //!< Guess what? It's the input label for the zero-suppressed digi
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputCompVirginRawDigiToken_;    //!< VR RawDigis to compare (from mainline)
    edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > inputCompZeroSuppressedDigiToken_;    //!< Zero-suppressed digis to compare (from mainline)
    //
    // Output information
    //====================
    std::string outputFolderName_; //!< Name for the folder in the TFileService file output.


}; // end of SiStripSpyDisplayModule class

                          
//
// static data member definitions
//

using namespace sistrip;
using namespace std;

//
// constructors and destructor
//
SiStripSpyDisplayModule::SiStripSpyDisplayModule(const edm::ParameterSet& iConfig) :
    detIDs_(                             iConfig.getParameter< std::vector<uint32_t> >( "detIDs")),
    inputScopeModeRawDigiLabel_(         iConfig.getParameter<edm::InputTag>( "InputScopeModeRawDigiLabel" ) ),
    inputPayloadRawDigiLabel_(           iConfig.getParameter<edm::InputTag>( "InputPayloadRawDigiLabel" ) ),
    inputReorderedPayloadRawDigiLabel_(  iConfig.getParameter<edm::InputTag>( "InputReorderedPayloadRawDigiLabel" ) ),
    inputReorderedModuleRawDigiLabel_(   iConfig.getParameter<edm::InputTag>( "InputReorderedModuleRawDigiLabel" ) ),
    inputPedestalsLabel_(                iConfig.getParameter<edm::InputTag>( "InputPedestalsLabel" ) ),
    inputNoisesLabel_(                   iConfig.getParameter<edm::InputTag>( "InputNoisesLabel" ) ),
    inputPostPedestalRawDigiLabel_(      iConfig.getParameter<edm::InputTag>( "InputPostPedestalRawDigiLabel" ) ),
    inputPostCMRawDigiLabel_(            iConfig.getParameter<edm::InputTag>( "InputPostCMRawDigiLabel" ) ),
    inputZeroSuppressedRawDigiLabel_(    iConfig.getParameter<edm::InputTag>( "InputZeroSuppressedRawDigiLabel" ) ),
    inputZeroSuppressedDigiLabel_(       iConfig.getParameter<edm::InputTag>( "InputZeroSuppressedDigiLabel" ) ),
    inputCompVirginRawDigiLabel_(        iConfig.getParameter<edm::InputTag>( "InputCompVirginRawDigiLabel" ) ),
    inputCompZeroSuppressedDigiLabel_(   iConfig.getParameter<edm::InputTag>( "InputCompZeroSuppressedDigiLabel" ) ),
    outputFolderName_(                   iConfig.getParameter<std::string>(   "OutputFolderName"    ) )
{
   //now do what ever initialization is needed
  inputScopeModeRawDigiToken_        = consumes<edm::DetSetVector<SiStripDigi> >(inputScopeModeRawDigiLabel_        );  
  inputPayloadRawDigiToken_	     = consumes<edm::DetSetVector<SiStripDigi> >(inputPayloadRawDigiLabel_	   );
  inputReorderedPayloadRawDigiToken_ = consumes<edm::DetSetVector<SiStripDigi> >(inputReorderedPayloadRawDigiLabel_ );
  inputReorderedModuleRawDigiToken_  = consumes<edm::DetSetVector<SiStripDigi> >(inputReorderedModuleRawDigiLabel_  );
  inputPedestalsToken_	             = consumes<edm::DetSetVector<SiStripDigi> >(inputPedestalsLabel_	           );
  inputNoisesToken_		     = consumes<edm::DetSetVector<SiStripDigi> >(inputNoisesLabel_		   );
  inputPostPedestalRawDigiToken_     = consumes<edm::DetSetVector<SiStripDigi> >(inputPostPedestalRawDigiLabel_     );
  inputPostCMRawDigiToken_	     = consumes<edm::DetSetVector<SiStripDigi> >(inputPostCMRawDigiLabel_	   );
  inputZeroSuppressedRawDigiToken_   = consumes<edm::DetSetVector<SiStripDigi> >(inputZeroSuppressedRawDigiLabel_   );
  inputZeroSuppressedDigiToken_      = consumes<edm::DetSetVector<SiStripDigi> >(inputZeroSuppressedDigiLabel_      );
  inputCompVirginRawDigiToken_       = consumes<edm::DetSetVector<SiStripDigi> >(inputCompVirginRawDigiLabel_       );
  inputCompZeroSuppressedDigiToken_  = consumes<edm::DetSetVector<SiStripDigi> >(inputCompZeroSuppressedDigiLabel_  );

}


SiStripSpyDisplayModule::~SiStripSpyDisplayModule()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

void
SiStripSpyDisplayModule::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup)
{
    // Retrieve FED cabling object
    //iSetup.get<SiStripDetCablingRcd>().get( cabling_ );
    //std::stringstream ss;
    //cabling_->print(ss);
    //std::cout << ss.str() << std::endl;

} // end of beginRun method.

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripSpyDisplayModule::beginJob()
{
    // register to the TFileService
    edm::Service<TFileService> fs;
    // Check that the TFileService has been configured
    if ( !fs.isAvailable() ) {
        throw cms::Exception("Configuration") << "TFileService not available: did you configure it ?";
    }
} // end of beginJob method.

// ------------ method called once each job just after ending the event loop  ------------
void SiStripSpyDisplayModule::endJob() {}

// ------------ method called to for each event  ------------
void
SiStripSpyDisplayModule::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    using namespace std;

    //retrieve cabling
    const SiStripDetCabling* lCabling = utility_.getDetCabling( iSetup );

    // Set up the event-level histogram folder
    //-----------------------------------------
    // register to the TFileService
    edm::Service<TFileService> fs;

    // Make the EDAnalyzer instance name directory
    TFileDirectory an_dir = fs->mkdir( outputFolderName_ );
    
    // Make the event directory filename
    stringstream ev_dir_name; 
    ev_dir_name << "run" << iEvent.id().run() << "_event" << iEvent.id().event();
    TFileDirectory evdir = an_dir.mkdir( ev_dir_name.str() );

    //if there are no detIds, get them from the comparison digis...
    if (detIDs_.size()==0) {
        //get the detIds of the modules in the zero-suppressed comparison
        if (!((inputCompZeroSuppressedDigiLabel_.label()=="") && (inputCompZeroSuppressedDigiLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripDigi > > czs_digis;
	    //            iEvent.getByLabel( inputCompZeroSuppressedDigiLabel_, czs_digis );
            iEvent.getByToken( inputCompZeroSuppressedDigiToken_, czs_digis );
            std::vector< edm::DetSet<SiStripDigi> >::const_iterator digis_it = czs_digis->begin();
            for (; digis_it!=czs_digis->end(); ++digis_it) {
                detIDs_.push_back(digis_it->detId());
            }
        }
        else if (!((inputCompVirginRawDigiLabel_.label()=="") && (inputCompVirginRawDigiLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > cvr_digis;
	    //            iEvent.getByLabel( inputCompVirginRawDigiLabel_, cvr_digis );
            iEvent.getByToken( inputCompVirginRawDigiToken_, cvr_digis );
            std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis_it = cvr_digis->begin();
            for (; digis_it!=cvr_digis->end(); ++digis_it) {
                detIDs_.push_back(digis_it->detId());
            }
        }

    }

    // Loop over detIDs as obtained from the SpyChannelMonitor config file.
    for (std::vector<uint32_t>::iterator d = detIDs_.begin(); d!=detIDs_.end(); ++d) {
        // TODO: Need some error checking here, probably...
        const std::vector<const FedChannelConnection *> & conns = lCabling->getConnections( *d );
        //cout << "________________________________________________" << endl;
        //cout << "FED channels found in detId " << *d << " is " << conns.size() << endl;
        if (!(conns.size())) {
            // TODO: Properly DEBUG/warning this...
            //cout << "Skipping detID " << uint32_t(*d) << endl;
            continue;
        }

        // Create a histogram directory for each specified and available detID
        stringstream sss;  //!< detID folder filename
        sss << "detID_" << *d;
        TFileDirectory detID_dir = evdir.mkdir( sss.str() );

        // Loop over the channels found with the detID and add directories.
        for (uint32_t ch = 0; ch<conns.size(); ch++) {
            
            // Name of channel histogram directory
            stringstream ssss; ssss << sss.str() << "_APVpair_" << ch;
            TFileDirectory chan_dir = detID_dir.mkdir(ssss.str());
            
            // Get the fed key from the detID and the channel
            uint32_t fedkey = SiStripFedKey::fedIndex(conns[ch]->fedId(), conns[ch]->fedCh());
            
            // (Spy) Scope Mode (SM)
            //=======================
            // Get the fed key from the FED ID and the FED channel (from conns)
            // This is because scope mode always stores in the collection by FED ID
            if (!((inputScopeModeRawDigiLabel_.label()=="") && (inputScopeModeRawDigiLabel_.instance()==""))) {
                // Use the SiStripFedKey object to return the FED key
                //cout << "detID=" << *d << ", FED key looking for is " << fedkey << endl;
                //cout << "Attempting to find scope mode raw digis" << endl;
                //
                edm::Handle< edm::DetSetVector< SiStripRawDigi > > sm_rawdigis;
		//                iEvent.getByLabel( inputScopeModeRawDigiLabel_, sm_rawdigis );
                iEvent.getByToken( inputScopeModeRawDigiToken_, sm_rawdigis );
                //
                // Note that the fed key (also a uint32_t) is passed in this case.
                // The method itself doesn't actually care, but it assumes whatever collection
                // is stored in sm_rawdigis is indexed by FED key ;-)
                // TODO: Make this, um, better.
                if (!(MakeRawDigiHist_(sm_rawdigis, fedkey, chan_dir, SCOPE_MODE))) { ; }
            }

            // Payload Unordered Raw (UR)
            //============================
            if (!((inputPayloadRawDigiLabel_.label()=="") && (inputPayloadRawDigiLabel_.instance()==""))) {
                uint32_t fedindex = SiStripFedKey::fedIndex(conns[ch]->fedId(), conns[ch]->fedCh());
                //cout << "Attempting to find payload mode raw digis" << endl;
                edm::Handle< edm::DetSetVector< SiStripRawDigi > > ur_rawdigis;
		//                iEvent.getByLabel( inputPayloadRawDigiLabel_, ur_rawdigis );
                iEvent.getByToken( inputPayloadRawDigiToken_, ur_rawdigis );
                if (!(MakeRawDigiHist_(ur_rawdigis, fedindex, chan_dir, PAYLOAD_RAW))) { ; }
            }
            // Payload Reordered Raw
            //=======================
            if (!((inputReorderedPayloadRawDigiLabel_.label()=="") && (inputReorderedPayloadRawDigiLabel_.instance()==""))) {
                uint32_t fedkey = SiStripFedKey::fedIndex(conns[ch]->fedId(), conns[ch]->fedCh());
                edm::Handle< edm::DetSetVector< SiStripRawDigi > > rrp_rawdigis;
		//                iEvent.getByLabel( inputReorderedPayloadRawDigiLabel_, rrp_rawdigis );
                iEvent.getByToken( inputReorderedPayloadRawDigiToken_, rrp_rawdigis );
                if (!(MakeRawDigiHist_(rrp_rawdigis, fedkey, chan_dir, REORDERED_PAYLOAD_RAW))) { ; }
            }
        } // end of loop over channels
        //
        // Module Reordered Raw (RR)
        //====================
        if (!((inputReorderedModuleRawDigiLabel_.label()=="") && (inputReorderedModuleRawDigiLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > rr_rawdigis;
	    //            iEvent.getByLabel( inputReorderedModuleRawDigiLabel_, rr_rawdigis );
            iEvent.getByToken( inputReorderedModuleRawDigiToken_, rr_rawdigis );
            //cout << "Making Reordered module histogram for detID " << *d << endl;
            if (!(MakeRawDigiHist_(rr_rawdigis, *d, detID_dir, REORDERED_MODULE_RAW))) { ; }
        } // end of ReorderedModuleRaw check
        
        //
        // Pedestal values 
        //========================
        if (!((inputPedestalsLabel_.label()=="") && (inputPedestalsLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > pd_rawdigis;
	    //            iEvent.getByLabel( inputPedestalsLabel_, pd_rawdigis );
            iEvent.getByToken( inputPedestalsToken_, pd_rawdigis );
            //cout << "Making pedestal values module histogram for detID " << *d << endl;
            if (!(MakeRawDigiHist_(pd_rawdigis, *d, detID_dir, PEDESTAL_VALUES))) { ; }
        }
         //
        // Noise values 
        //========================
        if (!((inputNoisesLabel_.label()=="") && (inputNoisesLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripProcessedRawDigi > > pd_rawdigis;
	    //            iEvent.getByLabel( inputNoisesLabel_, pd_rawdigis );
            iEvent.getByToken( inputNoisesToken_, pd_rawdigis );
            //cout << "Making noise values module histogram for detID " << *d << endl;
            if (!(MakeProcessedRawDigiHist_(pd_rawdigis, *d, detID_dir, NOISE_VALUES))) { ; }
        }
        //
        // Post-Pedestal Raw (PP)
        //========================
        if (!((inputPostPedestalRawDigiLabel_.label()=="") && (inputPostPedestalRawDigiLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > pp_rawdigis;
	    //            iEvent.getByLabel( inputPostPedestalRawDigiLabel_, pp_rawdigis );
            iEvent.getByToken( inputPostPedestalRawDigiToken_, pp_rawdigis );
            //cout << "Making post-pedestal module histogram for detID " << *d << endl;
            if (!(MakeRawDigiHist_(pp_rawdigis, *d, detID_dir, POST_PEDESTAL))) { ; }
        }
        //
        // Post-Common Mode Subtraction Raw (PC)
        //=======================================
        if (!((inputPostCMRawDigiLabel_.label()=="") && (inputPostCMRawDigiLabel_.instance()==""))) {
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > pc_rawdigis;
	    //            iEvent.getByLabel( inputPostCMRawDigiLabel_, pc_rawdigis );
            iEvent.getByToken( inputPostCMRawDigiToken_, pc_rawdigis );
            //cout << "Making post-CM module histogram for detID " << *d << endl;
            if (!(MakeRawDigiHist_(pc_rawdigis, *d, detID_dir, POST_COMMON_MODE))) { ; }
        }
        
        //
        // Zero-Suppressed Digis
        //=======================
        //bool founddigispy = false, founddigimain = false;
        if (!((inputZeroSuppressedDigiLabel_.label()=="") && (inputZeroSuppressedDigiLabel_.instance()==""))) {
            //cout << "Making ZeroSuppressed histogram!" << endl;
            edm::Handle< edm::DetSetVector< SiStripDigi > > zs_digis;
	    //            iEvent.getByLabel( inputZeroSuppressedDigiLabel_, zs_digis );
            iEvent.getByToken( inputZeroSuppressedDigiToken_, zs_digis );
            //founddigispy = 
	    MakeDigiHist_(zs_digis, *d, detID_dir, ZERO_SUPPRESSED);
        }
        //comparison to mainline data
        if (!((inputCompVirginRawDigiLabel_.label()=="") && (inputCompVirginRawDigiLabel_.instance()==""))) {
            //cout << "Making Mainline VirginRaw histogram!" << endl;
            edm::Handle< edm::DetSetVector< SiStripRawDigi > > cvr_digis;
	    //            iEvent.getByLabel( inputCompVirginRawDigiLabel_, cvr_digis );
            iEvent.getByToken( inputCompVirginRawDigiToken_, cvr_digis );
            //founddigimain = 
	    MakeRawDigiHist_(cvr_digis, *d, detID_dir, VR_COMP);
        }
        if (!((inputCompZeroSuppressedDigiLabel_.label()=="") && (inputCompZeroSuppressedDigiLabel_.instance()==""))) {
            //cout << "Making ZeroSuppressed histogram!" << endl;
            edm::Handle< edm::DetSetVector< SiStripDigi > > czs_digis;
	    //            iEvent.getByLabel( inputCompZeroSuppressedDigiLabel_, czs_digis );
            iEvent.getByToken( inputCompZeroSuppressedDigiToken_, czs_digis );
            //founddigimain = 
	    MakeDigiHist_(czs_digis, *d, detID_dir, ZERO_SUPPRESSED_COMP);
        }
        //if (founddigimain && founddigispy) cout << "Found digis for both in detid=" << *d << endl;
        
    } // end of loop over detIDs specified in the config.

} // end of Analyze method.

Bool_t SiStripSpyDisplayModule::MakeRawDigiHist_(
    const edm::Handle< edm::DetSetVector< SiStripRawDigi > > & digi_handle,
    uint32_t specifier,
    const TFileDirectory & dir,
    FEDSpyHistogramType type)
    //const std::string & name)
{
    // TODO: Remove the hard-coded numbers(!).
    TH1S * hist;
    if      (type==SCOPE_MODE)             hist = dir.make<TH1S>("ScopeMode",           ";Sample number;ADC counts / strip", 298, 0, 298);
    else if (type==PAYLOAD_RAW)            hist = dir.make<TH1S>("PayloadRaw",          ";Sample number;ADC counts / strip", 256, 0, 256);
    else if (type==REORDERED_PAYLOAD_RAW)  hist = dir.make<TH1S>("ReorderedPayloadRaw", ";Sample number;ADC counts / strip", 256, 0, 256);
    else if (type==REORDERED_MODULE_RAW)   hist = dir.make<TH1S>("ReorderedModuleRaw",  ";Sample number;ADC counts / strip", 768, 0, 768);
    else if (type==PEDESTAL_VALUES)        hist = dir.make<TH1S>("PedestalValues",      ";Strip number;Pedestal / strip",    768, 0, 768);
    else if (type==POST_PEDESTAL)          hist = dir.make<TH1S>("PostPedestal",        ";Strip number;ADC counts / strip",  768, 0, 768);
    else if (type==POST_COMMON_MODE)       hist = dir.make<TH1S>("PostCommonMode",      ";Strip number;ADC counts / strip",  768, 0, 768);
    else if (type==ZERO_SUPPRESSED_PADDED) hist = dir.make<TH1S>("ZeroSuppressedRaw" ,  ";Strip number;ADC counts / strip",  768, 0, 768);
    else if (type==VR_COMP)                hist = dir.make<TH1S>("VirginRawCom" ,       ";Strip number;ADC counts / strip",  768, 0, 768);
    else                                  {hist = 0; return false;}

    // TODO: May need to make this error checking independent when refactoring...
    //std::cout << "| * digis for " << type << " and detID " << specifier;
    std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis_it = digi_handle->find( specifier );
    if (digis_it == digi_handle->end()) { 
        //std::cout << " not found :( ";
        return false;
    }
    //std::cout << std::endl;

    // Loop over the digis for the detID and APV pair.
    edm::DetSet<SiStripRawDigi>::const_iterator idigi = digis_it->data.begin();
    uint32_t count = 0;
    for ( ; idigi != digis_it->data.end(); ++idigi ) {
        count++;
        hist->SetBinContent(count,static_cast<int>((*idigi).adc()));
    } // end of loop over the digis
    return true; // Success! (Probably.)
}

Bool_t SiStripSpyDisplayModule::MakeProcessedRawDigiHist_(
    const edm::Handle< edm::DetSetVector< SiStripProcessedRawDigi > > & digi_handle,
    uint32_t specifier,
    const TFileDirectory & dir,
    FEDSpyHistogramType type)
    //const std::string & name)
{
    // TODO: Remove the hard-coded numbers(!).
    TH1F * hist;
    if (type==NOISE_VALUES) hist = dir.make<TH1F>("NoiseValues",";Strip number;Noise / strip",768, 0, 768);
    else {
      hist = 0; 
      return false;
    }

    // TODO: May need to make this error checking independent when refactoring...
    //std::cout << "| * digis for " << type << " and detID " << specifier;
    std::vector< edm::DetSet<SiStripProcessedRawDigi> >::const_iterator digis_it = digi_handle->find( specifier );
    if (digis_it == digi_handle->end()) { 
        //std::cout << " not found :( ";
        return false;
    }
    //std::cout << std::endl;

    // Loop over the digis for the detID and APV pair.
    edm::DetSet<SiStripProcessedRawDigi>::const_iterator idigi = digis_it->data.begin();
    uint32_t count = 0;
    for ( ; idigi != digis_it->data.end(); ++idigi ) {
        count++;
        hist->SetBinContent(count,static_cast<float>((*idigi).adc()));
    } // end of loop over the digis
    return true; // Success! (Probably.)
}

Bool_t SiStripSpyDisplayModule::MakeDigiHist_(
    const edm::Handle< edm::DetSetVector< SiStripDigi > > & digi_handle,
    uint32_t detID,
    //uint32_t channel,
    const TFileDirectory & dir,
    FEDSpyHistogramType type)
    //const std::string & name)
{
    // TODO: Remove the hard-coded numbers.
    TH1S * hist;
    if      (type==ZERO_SUPPRESSED)        hist = dir.make<TH1S>("ZeroSuppressedDigi",      ";Strip number;ADC counts / strip",  768, 0, 768);
    else if (type==ZERO_SUPPRESSED_COMP)   hist = dir.make<TH1S>("ZeroSuppressedDigiComp",  ";Strip number;ADC counts / strip",  768, 0, 768);
    else                                  {hist = 0; return false;}
    
    // TODO: May need to make this error checking independent when refactoring...
    std::vector< edm::DetSet<SiStripDigi> >::const_iterator digis_it = digi_handle->find( detID );
    if (digis_it == digi_handle->end()) {
        return false;
    }
    else {
        //cout << "--* ZS digis found for detID " << detID << endl;
    }

    // Loop over the digis for the detID and APV pair.
    edm::DetSet<SiStripDigi>::const_iterator idigi = digis_it->data.begin();
    bool founddigi = false;
    for ( ; idigi != digis_it->data.end(); ++idigi ) {
        // Check strip number is within the channel limits
        //if ( static_cast<uint16_t>( (*idigi).strip()/256. ) == channel ) {
        //    hist->SetBinContent( ((*idigi).strip())%256 + 1,(*idigi).adc());
        //}
        hist->SetBinContent( static_cast<int>(((*idigi).strip())) + 1, static_cast<int>((*idigi).adc()) );
        if ( (*idigi).adc() > 0 ) founddigi = true;
        //cout << "----* ZS digi found at " << static_cast<int>(((*idigi).strip()))
        //     << ", " << static_cast<int>((*idigi).adc()) << endl;
    } // end of loop over the digis

    return founddigi;
}

// Define this as a plug-in
DEFINE_FWK_MODULE(SiStripSpyDisplayModule);
