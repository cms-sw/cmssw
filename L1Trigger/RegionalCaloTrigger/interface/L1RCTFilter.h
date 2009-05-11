
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

// L1 Extra Electromagnetic Stuff                                              
/* #include "DataFormats/L1Trigger/interface/L1EmParticle.h" */
/* #include "DataFormats/L1Trigger/interface/L1JetParticle.h" */
/* #include "DataFormats/L1Trigger/interface/L1ParticleMap.h" */
/* #include "DataFormats/L1Trigger/interface/L1EtMissParticle.h" */
/* #include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h" */
/* #include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h" */
/* #include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h" */
/* #include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h" */
/* #include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h" */

using std::cerr;
using std::cout;
using std::endl;

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TNtuple.h"
#include "TLorentzVector.h"

using namespace std;
using namespace edm;
//using namespace reco;

class L1RCTFilter : public edm::EDFilter
{
 public:
  explicit L1RCTFilter(edm::ParameterSet const&);
  virtual ~L1RCTFilter();
 
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
  void endJob();

 private:
  double pTMin;
  vector<int> cardNumber;
  edm::InputTag rctDigisLabel;
};
