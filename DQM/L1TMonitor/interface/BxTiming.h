#ifndef BxTiming_H
#define BxTiming_H

/*\class BxTiming
 *\description common FED timing DQM module
 *\author N.Leonardo, A.Holzner, T.Christiansen, I.Mikulec
 *\date 08.03
 */

// system, common includes
#include <memory>
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// dqm includes
#include "DQMServices/Core/interface/DQMStore.h"
//dataFormats
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#define nfed_ FEDNumbering::MAXFEDID + 1

class BxTiming : public DQMEDAnalyzer {
public:
  explicit BxTiming(const edm::ParameterSet&);
  ~BxTiming() override;

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;

private:
  // input
  edm::InputTag fedSource_;
  edm::EDGetTokenT<FEDRawDataCollection> fedSource_token_;
  edm::InputTag gtSource_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtSource_token_;

  // debug verbose level
  int verbose_;
  int verbose() { return verbose_; }

  /** calculates the difference (closest distance) between two bunch crossing numbers.
      This is similar to calculating delta phi between two vectors. 
      
      Calculates bx1 - bx2 but makes sure that the value is in the range 
        -num_bx_per_orbit / 2 .. + num_bx_per_orbit / 2 .
  */
  int calcBxDiff(int bx1, int bx2);

  // counters
  int nEvt_;

  // root output file name
  std::string histFile_;

  // dqm histogram folder
  std::string histFolder_;

  // running in filter farm? (use reduced set of me's)
  bool runInFF_;

  // readout l1 systems
  static const int norb_ = 3564;            // bx per orbit
  static const int half_norb_ = norb_ / 2;  // for calculating the difference between two BX numbers

  static const int nbig_ = 10000;  // larger than bx spread
  static const int nttype_ = 6;    // number of trigger types (physics, cal,...)

  std::vector<int> listGtBits_;  // selected gt bit numbers for synch monitoring

  enum nsys { NSYS = 10 };
  enum syslist { PS = 0, ETP, HTP, GCT, CTP, CTF, DTP, DTF, RPC, GLT };
  std::pair<int, int> fedRange_[NSYS];
  int fedRef_;  // reference fed

  // bx spread counters
  static const int nspr_ = 3;  // delta, min, max
  int nBxDiff[nfed_][nspr_];
  int nBxOccy[nfed_][nspr_];

  /// histograms
  MonitorElement* hBxDiffAllFed;        // bx shift wrt reference fed, for all feds
  MonitorElement* hBxDiffSysFed[NSYS];  // bx shift wrt reference fed, per subsystem
  MonitorElement* hBxOccyAllFed;        // bx occupancy, for all fed's
  MonitorElement** hBxOccyOneFed;       // bx occupancy, per each fed

  MonitorElement* hBxDiffAllFedSpread[nspr_];  // bx shift wrt ref fed: mean shift, min, max
  MonitorElement* hBxOccyAllFedSpread[nspr_];  // bx occupancy: mean shift, min, max

  MonitorElement* hBxOccyGtTrigType[nttype_];  // gt bx occupancy per trigger type
  MonitorElement** hBxOccyTrigBit[NSYS];       // subsystem bx occupancy per selected trigger bit
};

#endif
