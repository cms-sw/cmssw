#ifndef DQM_L1TMonitor_L1TdeStage2CaloLayer2
#define DQM_L1TMonitor_L1TdeStage2CaloLayer2

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

namespace calolayer2dedqm {
  struct Histograms {
    // objects to represent individual plots shown in DQM
    ConcurrentMonitorElement agreementSummary;
    ConcurrentMonitorElement jetSummary;
    ConcurrentMonitorElement tauSummary;
    ConcurrentMonitorElement egSummary;
    ConcurrentMonitorElement sumSummary;
    ConcurrentMonitorElement problemSummary;

    // histograms to store the properties of mismatched jets
    ConcurrentMonitorElement jetEtData;
    ConcurrentMonitorElement jetEtaData;
    ConcurrentMonitorElement jetPhiData;
    ConcurrentMonitorElement jetEtEmul;
    ConcurrentMonitorElement jetEtaEmul;
    ConcurrentMonitorElement jetPhiEmul;

    // histograms to store the properties of mismatched non-isolated e/g
    ConcurrentMonitorElement egEtData;
    ConcurrentMonitorElement egEtaData;
    ConcurrentMonitorElement egPhiData;
    ConcurrentMonitorElement egEtEmul;
    ConcurrentMonitorElement egEtaEmul;
    ConcurrentMonitorElement egPhiEmul;

    // histograms to store the properties of mismatched isolated e/g
    ConcurrentMonitorElement isoEgEtData;
    ConcurrentMonitorElement isoEgEtaData;
    ConcurrentMonitorElement isoEgPhiData;
    ConcurrentMonitorElement isoEgEtEmul;
    ConcurrentMonitorElement isoEgEtaEmul;
    ConcurrentMonitorElement isoEgPhiEmul;

    // histograms to store the properties of mismatched non-isolated taus
    ConcurrentMonitorElement tauEtData;
    ConcurrentMonitorElement tauEtaData;
    ConcurrentMonitorElement tauPhiData;
    ConcurrentMonitorElement tauEtEmul;
    ConcurrentMonitorElement tauEtaEmul;
    ConcurrentMonitorElement tauPhiEmul;

    // histograms to store the properties of mismatched isolated taus
    ConcurrentMonitorElement isoTauEtData;
    ConcurrentMonitorElement isoTauEtaData;
    ConcurrentMonitorElement isoTauPhiData;
    ConcurrentMonitorElement isoTauEtEmul;
    ConcurrentMonitorElement isoTauEtaEmul;
    ConcurrentMonitorElement isoTauPhiEmul;

    // histograms for mismatched ett sums
    ConcurrentMonitorElement ettData;
    ConcurrentMonitorElement ettEmul;
    ConcurrentMonitorElement ettHFData;
    ConcurrentMonitorElement ettHFEmul;
    ConcurrentMonitorElement ettEmData;
    ConcurrentMonitorElement ettEmEmul;

    // mismatched htt sums
    ConcurrentMonitorElement httData;
    ConcurrentMonitorElement httEmul;
    ConcurrentMonitorElement httHFData;
    ConcurrentMonitorElement httHFEmul;

    // mismatched met sums
    ConcurrentMonitorElement metEtData;
    ConcurrentMonitorElement metEtEmul;
    ConcurrentMonitorElement metPhiData;
    ConcurrentMonitorElement metPhiEmul;
    ConcurrentMonitorElement metHFEtData;
    ConcurrentMonitorElement metHFEtEmul;
    ConcurrentMonitorElement metHFPhiData;
    ConcurrentMonitorElement metHFPhiEmul;

    // mismatched mht sums
    ConcurrentMonitorElement mhtEtData;
    ConcurrentMonitorElement mhtEtEmul;
    ConcurrentMonitorElement mhtPhiData;
    ConcurrentMonitorElement mhtPhiEmul;
    ConcurrentMonitorElement mhtHFEtData;
    ConcurrentMonitorElement mhtHFEtEmul;
    ConcurrentMonitorElement mhtHFPhiData;
    ConcurrentMonitorElement mhtHFPhiEmul;

    // mismatched min bias sums
    ConcurrentMonitorElement mbhfp0Data;
    ConcurrentMonitorElement mbhfp0Emul;
    ConcurrentMonitorElement mbhfm0Data;
    ConcurrentMonitorElement mbhfm0Emul;
    ConcurrentMonitorElement mbhfp1Data;
    ConcurrentMonitorElement mbhfp1Emul;
    ConcurrentMonitorElement mbhfm1Data;
    ConcurrentMonitorElement mbhfm1Emul;

    // mismatched towercount sum
    ConcurrentMonitorElement towCountData;
    ConcurrentMonitorElement towCountEmul;

  };
}

/**
 * Short class description.
 *
 * Longer class description...
 * ... desc continued.
 */
class L1TdeStage2CaloLayer2 : public DQMGlobalEDAnalyzer<calolayer2dedqm::Histograms> {

 public:
  /**
   * Class constructor
   *
   * Receives the set of parameters, specified by the python configuration file
   * used to initialise the module as a part of a sequence. The contents of the
   * set is used to configure the internal state of the objects of this class.
   * Values from the parameter set are extracted and used to initialise
   * bxcollections for jet, e/g, tau and sum objects reconstructed by firmware
   * (data) and emulator. These collections are the basis of the comparisons
   * performed by this module.
   *
   * @param edm::ParamterSet & ps A pointer to the parameter set used
   */
  L1TdeStage2CaloLayer2 (const edm::ParameterSet & ps);


 protected:

  /**
   * Method to declare or "book" all histograms that will be part of module
   *
   * Histograms that are to be visualised as part of the DQM module should be
   * registered with the ConcurrentBooker object any additional configuration
   * such as title or axis labels and ranges. A good rule of thumb for the
   * amount of configuration is that it should be possible to understnand the
   * contents of the histogram using the configuration received from this
   * method since the plots generated by this module would later be stored
   * into ROOT files for transfer to the DQM system and it should be possible
   * to ...
   *
   * @param DQMStore::ConcurrentBooker& booker Object that handles the creation of plots
   * @param edm::Run const &           Reference to run object
   * @param edm::EventSetup const &    Reference to event configuration object
   * @param calolayer2dedqm::Histograms &  Reference to struct of concurrent monitoring elements
   *
   * @return void
   */
  void bookHistograms (DQMStore::ConcurrentBooker&,
			       const edm::Run&,
			       const edm::EventSetup&,
                               calolayer2dedqm::Histograms&) const override;

  /**
   * Main method where the analysis code resides, executed once for each run
   *
   * The main body of the module code is contained in this method. The different
   * object collections are extracted from the run and are passed to the
   * respective comparison methods for processing of each object type.
   *
   * @param edm::Event const &         Reference to event object
   * @param edm::EventSetup const &    Reference to event configuration object
   * @param calolayer2dedqm::Histograms const &  Reference to struct of concurrent monitoring elements
   *
   * @return void
   */
  void dqmAnalyze (const edm::Event&,
                   const edm::EventSetup&,
                   const calolayer2dedqm::Histograms&) const override;

 private:

  /**
   * Encapsulates the code required for performing a comparison of
   * the jets contained in a given event.
   *
   * Method is called once per each event with the jet collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the jets in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.When
   * the length of data/emul collections is different, all of the objects are
   * "stored" in the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::JetBXCollection>& dataCol Reference to jet
   *    collection from data
   * @param edm::Handle<l1t::JetBXCollection>& emulCol Reference to jet
   *    collection from emulation
   * @param calolayer2dedqm::Histograms const &  Reference to struct of
   *    concurrent monitoring elements
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareJets(const edm::Handle<l1t::JetBxCollection> & dataCol,
                   const edm::Handle<l1t::JetBxCollection> & emulCol,
                   const calolayer2dedqm::Histograms&) const;

  /**
   * Encapsulates the code required for performing a comparison of
   * the e/gs contained in a given event.
   *
   * Method is called once per each event with the e/g collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the e/gs in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.
   * Another distinction is made between isolated and non-isolated e/g
   * candidates and problematic objects are handled accordingly. When the length
   * of data/emul collections is different, all of the objects are "stored" in
   * the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::EGammaBXCollection>& dataCol Reference to e/gamma
   *    collection from data
   * @param edm::Handle<l1t::EGammaBXCollection>& emulCol Reference to e/gamma
   *    collection from emulation
   * @param calolayer2dedqm::Histograms const &  Reference to struct of
   *    concurrent monitoring elements
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareEGs(const edm::Handle<l1t::EGammaBxCollection> & dataCol,
                  const edm::Handle<l1t::EGammaBxCollection> & emulCol,
                  const calolayer2dedqm::Histograms&) const;

  /**
   * Encapsulates the code required for performing a comparison of
   * the taus contained in a given event.
   *
   * Method is called once per each event with the tau collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation checks
   * if the size of collections is the same and when so, compares the taus in
   * the same positions within the data/emul collections. When a discrepancy is
   * found, the properties (Et, eta, phi) of the problematic data/emul objects
   * are stored in dedicated histograms. In cases of differences, a distinction
   * is made between objects whose energy or position are in disagremeent.
   * Another distinction is made between isolated and non-isolated tau
   * candidates and problematic objects are handled accordingly. When the length
   * of data/emul collections is different, all of the objects are "stored" in
   * the histograms dedicated to problematic objects.
   *
   * @param edm::Handle<l1t::TauBXCollection>& dataCol Reference to tau
   *    collection from data
   * @param edm::Handle<l1t::TauBXCollection>& emulCol Reference to tau
   *    collection from emulation
   * @param calolayer2dedqm::Histograms const &  Reference to struct of
   *    concurrent monitoring elements
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareTaus(const edm::Handle<l1t::TauBxCollection> & dataCol,
                   const edm::Handle<l1t::TauBxCollection> & emulCol,
                   const calolayer2dedqm::Histograms&) const;

  /**
   * Encapsulates the code required for performing a comparison of
   * the taus contained in a given event.
   *
   * Method is called once per each event with the sum collections associated
   * with the event being extracted for bx = 0 as the RAW information required
   * to run the emulator is only available for bx 0. The implementation loops
   * over the collection and depending of the sum type, each variant is compared
   * independently. If any disagreement is found, the event is marked a bad and
   * the properties of the sum are stored in the data/emulator problematic
   * histograms.
   *
   * @param edm::Handle<l1t::TauBXCollection>& dataCol Reference to tau
   *    collection from data
   * @param edm::Handle<l1t::TauBXCollection>& emulCol Reference to tau
   *    collection from emulation
   * @param calolayer2dedqm::Histograms const &  Reference to struct of
   *    concurrent monitoring elements
   *
   * @return bool Flag of whether the agreement was perfect
   */
  bool compareSums(const edm::Handle<l1t::EtSumBxCollection> & dataCol,
                   const edm::Handle<l1t::EtSumBxCollection> & emulCol,
                   const calolayer2dedqm::Histograms&) const;

  // Holds the name of directory in DQM where module hostograms will be shown.
  // Value is taken from python configuration file (passed in class constructor)
  std::string monitorDir;

  // collections to hold entities reconstructed from data and emulation
  edm::EDGetTokenT<l1t::JetBxCollection> calol2JetCollectionData;
  edm::EDGetTokenT<l1t::JetBxCollection> calol2JetCollectionEmul;
  edm::EDGetTokenT<l1t::EGammaBxCollection> calol2EGammaCollectionData;
  edm::EDGetTokenT<l1t::EGammaBxCollection> calol2EGammaCollectionEmul;
  edm::EDGetTokenT<l1t::TauBxCollection> calol2TauCollectionData;
  edm::EDGetTokenT<l1t::TauBxCollection> calol2TauCollectionEmul;
  edm::EDGetTokenT<l1t::EtSumBxCollection> calol2EtSumCollectionData;
  edm::EDGetTokenT<l1t::EtSumBxCollection> calol2EtSumCollectionEmul;

  enum summaryBins {
    NEVENTS = 1,      // total number of events
    EVENTGOOD,        // number of good events (100% agreement)
    NJETS_S,          // total number of jets objects found
    JETGOOD_S,        // number of jets in agreement (energy and pos)
    NEGS_S,           // total number of e/g objects found
    EGGOOD_S,         // number of e/g in agremeent (energy and pos)
    NTAUS_S,          // total number of tau objects found
    TAUGOOD_S,        // number of taus in agremenet (energy and pos)
    NSUMS_S,          // total number of sums
    SUMGOOD_S         // number of good sums across all events
  };

  enum problemCauses {
    NEVENTS_P = 1,    // total number of events
    JETCOLLSIZE,      // no. events with different data/emul obj. in jet coll.
    EGCOLLSIZE,       // no. events with different data/emul obj. in eg coll.
    TAUCOLLSIZE,      // no. events with different data/emul obj. in tau coll.
    JETMISMATCH,      // no. events failed due to a jet mismatch
    EGMISMATCH,       // no. events failed due to an e/g mismatch
    TAUMISMATCH,      // no. events failed due to a tau mismatch
    SUMMISMATCH       // no. events failed due to a sum mismatch
  };

  enum jetVars {
    NJETS = 1,
    JETGOOD,
    JETPOSOFF,
    JETETOFF
  };

  enum egVars {
    NEGS = 1,
    EGGOOD,
    EGPOSOFF,
    EGETOFF,
    NISOEGS,
    ISOEGGOOD,
    ISOEGPOSOFF,
    ISOEGETOFF
  };

  enum tauVars {
    NTAUS = 1,
    TAUGOOD,
    TAUPOSOFF,
    TAUETOFF,
    NISOTAUS,
    ISOTAUGOOD,
    ISOTAUPOSOFF,
    ISOTAUETOFF
  };

  enum sumVars {
    NSUMS = 1,
    SUMGOOD,
    NETTSUMS,
    ETTSUMGOOD,
    NHTTSUMS,
    HTTSUMGOOD,
    NMETSUMS,
    METSUMGOOD,
    NMHTSUMS,
    MHTSUMGOOD,
    NMBHFSUMS,
    MBHFSUMGOOD,
    NTOWCOUNTS,
    TOWCOUNTGOOD
  };

  bool verbose;

  // use only bx = 0 since it only contains RAW data (needed for emulator)
  const unsigned int currBx = 0;
};

#endif
