#ifndef QcdHighPtDQM_H
#define QcdHighPtDQM_H


/** \class QcdHighPtDQM
 *
 *  DQM Physics Module for High Pt QCD group
 *
 *  Based on DQM/SiPixel and DQM/Physics code
 *  Version 1.0, 7/7/09
 *  By Keith Rose
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/MET.h"

class DQMStore;
class MuonServiceProxy;
class MonitorElement;

class QcdHighPtDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  QcdHighPtDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~QcdHighPtDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  void endJob(void);

 private:


  // ----------member data ---------------------------
  
  DQMStore* theDbe;
  MuonServiceProxy *theService;

  //input tags for Jets/MET
  edm::InputTag jetLabel_;
  edm::InputTag metLabel_;

  //map of MEs
  std::map<std::string, MonitorElement*> MEcontainer_;



};
#endif  
