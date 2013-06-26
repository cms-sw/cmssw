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
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"


class DQMStore;
class MonitorElement;

class QcdHighPtDQM : public edm::EDAnalyzer {
 public:

  /// Constructor
  QcdHighPtDQM(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~QcdHighPtDQM();
  
  /// Inizialize parameters for histo binning
  void beginJob();

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  void endJob(void);



 private:


  // ----------member data ---------------------------
  
  DQMStore* theDbe;

  //input tags for Jets/MET
  edm::InputTag jetLabel_;
  edm::InputTag metLabel1_;
  edm::InputTag metLabel2_;
  edm::InputTag metLabel3_;
  edm::InputTag metLabel4_;
  
  //map of MEs
  std::map<std::string, MonitorElement*> MEcontainer_;

  //methods to calculate MET over SumET and MET over Leading Jet Pt
  float movers(const reco::CaloMETCollection& metcollection);
  float moverl(const reco::CaloMETCollection& metcollection, float& ljpt);

};
#endif  
