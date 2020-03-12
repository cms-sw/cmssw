//-------------------------------------------------
//
/**  \class DTTrigProd
 *     Main EDProducer for the DTTPG
 *
 *
 *
 *   \author C. Battilana
 *
 */
//
//--------------------------------------------------

// Framework related classes
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1Trigger/DTTrigger/interface/DTTrig.h"

// Data Formats classes
#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

// Collaborating classes
#include <iostream>

using namespace edm;
using namespace std;

// DataFormats interface
typedef vector<DTSectCollPhSegm> SectCollPhiColl;
typedef SectCollPhiColl::const_iterator SectCollPhiColl_iterator;
typedef vector<DTSectCollThSegm> SectCollThetaColl;
typedef SectCollThetaColl::const_iterator SectCollThetaColl_iterator;

class DTTrigProd : public edm::stream::EDProducer<> {
public:
  //! Constructor
  DTTrigProd(const edm::ParameterSet& pset);

  //! Create Trigger Units before starting event processing
  void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  //! Producer: process every event and generates trigger data
  void produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) override;

private:
  // Trigger istance
  DTTrig my_trig;

  edm::EDPutTokenT<L1MuDTChambPhContainer> phToken_;
  edm::EDPutTokenT<L1MuDTChambThContainer> thToken_;

  // Trigger Configuration Manager CCB validity flag
  bool my_CCBValid = false;

  // Sector Format Flag true=[0-11] false=[1-12]
  const bool my_DTTFnum;

  // Debug Flag
  const bool my_debug;

  // Lut dump file parameters
  const bool my_lut_dump_flag;
  const short int my_lut_btic;
};

DTTrigProd::DTTrigProd(const ParameterSet& pset)
    : my_trig(pset, consumesCollector()),
      phToken_{produces<L1MuDTChambPhContainer>()},
      thToken_{produces<L1MuDTChambThContainer>()},
      my_DTTFnum{pset.getParameter<bool>("DTTFSectorNumbering")},
      my_debug{pset.getUntrackedParameter<bool>("debug")},
      my_lut_dump_flag{pset.getUntrackedParameter<bool>("lutDumpFlag")},
      my_lut_btic{static_cast<short int>(pset.getUntrackedParameter<int>("lutBtic"))} {}

void DTTrigProd::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (my_debug)
    cout << "DTTrigProd::beginRun  " << iRun.id().run() << endl;

  ESHandle<DTConfigManager> dtConfig;
  iEventSetup.get<DTConfigManagerRcd>().get(dtConfig);

  my_CCBValid = dtConfig->CCBConfigValidity();

  my_trig.createTUs(iEventSetup);
  if (my_debug)
    cout << "[DTTrigProd] TU's Created" << endl;

  if (my_lut_dump_flag) {
    cout << "Dumping luts...." << endl;
    my_trig.dumpLuts(my_lut_btic, dtConfig.product());
  }
}

void DTTrigProd::produce(Event& iEvent, const EventSetup& iEventSetup) {
  vector<L1MuDTChambPhDigi> outPhi;
  vector<L1MuDTChambThDigi> outTheta;

  // SV check if CCB configuration is valid, otherwise just produce empty collections
  if (!my_CCBValid) {
    if (my_debug)
      cout << "[DTTrigProd] CCB configuration is not valid for this run, empty collection will be produced " << endl;
  } else {
    my_trig.triggerReco(iEvent, iEventSetup);
    // BX offset used to correct DTTPG output
    int bx_offset = my_trig.getBXOffset();

    if (my_debug)
      cout << "[DTTrigProd] Trigger algorithm run for " << iEvent.id() << endl;

    // Convert Phi Segments
    SectCollPhiColl myPhiSegments;
    myPhiSegments = my_trig.SCPhTrigs();

    SectCollPhiColl_iterator SCPCend = myPhiSegments.end();
    for (SectCollPhiColl_iterator it = myPhiSegments.begin(); it != SCPCend; ++it) {
      int step = (*it).step() - bx_offset;  // Shift correct BX to 0 (needed for DTTF data processing)
      int sc_sector = (*it).SCId().sector();
      if (my_DTTFnum == true)
        sc_sector--;  // Modified for DTTF numbering [0-11]
      outPhi.push_back(L1MuDTChambPhDigi(step,
                                         (*it).ChamberId().wheel(),
                                         sc_sector,
                                         (*it).ChamberId().station(),
                                         (*it).phi(),
                                         (*it).phiB(),
                                         (*it).code(),
                                         !(*it).isFirst(),
                                         0));
    }

    // Convert Theta Segments
    SectCollThetaColl myThetaSegments;
    myThetaSegments = my_trig.SCThTrigs();

    SectCollThetaColl_iterator SCTCend = myThetaSegments.end();
    for (SectCollThetaColl_iterator it = myThetaSegments.begin(); it != SCTCend; ++it) {
      int pos[7], qual[7];
      for (int i = 0; i < 7; i++) {
        pos[i] = (*it).position(i);
        qual[i] = (*it).quality(i);
      }
      int step = (*it).step() - bx_offset;  // Shift correct BX to 0 (needed for DTTF data processing)
      int sc_sector = (*it).SCId().sector();
      if (my_DTTFnum == true)
        sc_sector--;  // Modified for DTTF numbering [0-11]
      outTheta.push_back(
          L1MuDTChambThDigi(step, (*it).ChamberId().wheel(), sc_sector, (*it).ChamberId().station(), pos, qual));
    }
  }

  // Write everything into the event (CB write empty collection as default actions if emulator does not run)
  iEvent.emplace(phToken_, std::move(outPhi));
  iEvent.emplace(thToken_, std::move(outTheta));
}

DEFINE_FWK_MODULE(DTTrigProd);
