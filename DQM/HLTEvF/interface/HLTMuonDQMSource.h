#ifndef HLTMuonDQMSource_H
#define HLTMuonDQMSource_H

/** \class HLTMuonDQMSource
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/02/17 11:47:38 $
 *  $Revision: 1.9 $
 *  \author  M. Vander Donckt CERN
 *   
 */
#include <memory>
#include <unistd.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class HLTMuonDQMSource : public edm::EDAnalyzer {
public:
  HLTMuonDQMSource( const edm::ParameterSet& );
  ~HLTMuonDQMSource();

protected:
   
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:
 
  edm::ParameterSet parameters_;

  DQMStore* dbe_;  
  std::string monitorName_;
  std::string outputFile_;
  int counterEvt_;      ///counter
  int nTrig_;		/// mutriggered events
  int prescaleEvt_;     ///every n events
  double coneSize_;
  edm::InputTag l2collectionTag_;
  edm::InputTag l2isolationTag_;
  edm::InputTag l2seedscollectionTag_;
  edm::InputTag l3seedscollectionTag_;
  edm::InputTag l3collectionTag_;
  edm::InputTag l3isolationTag_;
  //  edm::InputTag l3linksTag_;

  std::vector<std::string> theHLTCollectionLabels;
  std::vector<std::string> theHLTCollectionLevel;

  // ----------member data ---------------------------
  bool verbose_;
  static const int NTRIG = 5;
  int nTrigs;

  MonitorElement * hNMu[NTRIG][4];
  MonitorElement * hcharge[NTRIG][4];
  MonitorElement * hchargeconv[NTRIG][3];
  MonitorElement * hpt[NTRIG][4];
  MonitorElement * heta[NTRIG][4];
  MonitorElement * hphi[NTRIG][4];
  MonitorElement * hptphi[NTRIG][4];
  MonitorElement * hpteta[NTRIG][4];
  MonitorElement * hptres[NTRIG][3];
  MonitorElement * hptrespt[NTRIG][3];
  MonitorElement * hetares[NTRIG][3];
  MonitorElement * hetareseta[NTRIG][3];
  MonitorElement * hphires[NTRIG][3];
  MonitorElement * hphiresphi[NTRIG][3];
  MonitorElement * hetaphi[NTRIG][4];
  MonitorElement * hdr[NTRIG][2];
  MonitorElement * hd0[NTRIG][2];
  MonitorElement * hdz[NTRIG][2];
  MonitorElement * hdrphi[NTRIG][2];
  MonitorElement * hd0phi[NTRIG][2];
  MonitorElement * hdzeta[NTRIG][2];
  MonitorElement * herr0[NTRIG][2];
  MonitorElement * hnHits[NTRIG][4];
  MonitorElement * hnValidHits[NTRIG];
  MonitorElement * hnTkValidHits[NTRIG];
  MonitorElement * hnMuValidHits[NTRIG];
  MonitorElement * hdimumass[NTRIG][2];
  MonitorElement * hiso[NTRIG][2];
  MonitorElement * hl1quality[NTRIG];
  MonitorElement * hptfrac[NTRIG][2];
  MonitorElement * hetafrac[NTRIG][2];
  MonitorElement * hphifrac[NTRIG][2];
  float XMIN; float XMAX;

  TH1D *_hpt1[NTRIG][2], *_hpt2[NTRIG][2];
  TH1D *_heta1[NTRIG][2], *_heta2[NTRIG][2];
  TH1D *_hphi1[NTRIG][2], *_hphi2[NTRIG][2];
};

#endif

