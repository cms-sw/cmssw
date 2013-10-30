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

  //input tags/Tokens for Jets/MET
  edm::EDGetTokenT<reco::CaloJetCollection> jetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken1_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken2_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken3_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken4_;

  //map of MEs
  std::map<std::string, MonitorElement*> MEcontainer_;

  //methods to calculate MET over SumET and MET over Leading Jet Pt
  float movers(const reco::CaloMETCollection& metcollection);
  float moverl(const reco::CaloMETCollection& metcollection, float& ljpt);

};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
