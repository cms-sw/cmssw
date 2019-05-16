#ifndef DQMSourceExample_H
#define DQMSourceExample_H

/** \class DQMSourceExample
 * *
 *  DQM Test Client
 *
 *  \author  M. Zanetti CERN
 *
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include "DQMServices/Core/interface/DQMStore.h"

//
// class declaration
//

class DQMSourceExample : public edm::EDAnalyzer {
public:
  DQMSourceExample(const edm::ParameterSet &);
  ~DQMSourceExample() override;

protected:
  // BeginJob
  void beginJob() override;

  // BeginRun
  void beginRun(const edm::Run &r, const edm::EventSetup &c) override;

  // Fake Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  // DQM Client Diagnostic

  // EndRun
  void endRun(const edm::Run &r, const edm::EventSetup &c) override;

  // Endjob
  void endJob() override;

private:
  void initialize();

  edm::ParameterSet parameters_;

  DQMStore *dbe_;
  std::string monitorName_;

  int counterEvt_;
  int counterLS_;

  int prescaleEvt_;  // every n events
  int prescaleLS_;   // units of lumi sections

  // ---------- member data ----------

  int NBINS;
  float XMIN, XMAX;

  // monitor elements for testing of Quality Tests
  MonitorElement *xTrue;
  MonitorElement *xFalse;
  MonitorElement *yTrue;
  MonitorElement *yFalse;

  MonitorElement *wExpTrue;
  MonitorElement *wExpFalse;
  MonitorElement *meanTrue;
  MonitorElement *meanFalse;

  MonitorElement *deadTrue;
  MonitorElement *deadFalse;
  MonitorElement *noisyTrue;
  MonitorElement *noisyFalse;

  // several ME more
  MonitorElement *i1;
  MonitorElement *f1;
  MonitorElement *s1;
  MonitorElement *p1;
  MonitorElement *p2;
  MonitorElement *h1;
  MonitorElement *h1hist;
  MonitorElement *h2;
  MonitorElement *h3;
  MonitorElement *h4;
  MonitorElement *summ;
};

#endif
