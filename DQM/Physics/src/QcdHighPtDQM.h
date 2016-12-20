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

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

class DQMStore;
class MonitorElement;

class QcdHighPtDQM : public DQMEDAnalyzer {
 public:
  QcdHighPtDQM(const edm::ParameterSet&);
  virtual ~QcdHighPtDQM();
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  // input tags/Tokens for Jets/MET
  edm::EDGetTokenT<reco::CaloJetCollection> jetToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken1_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken2_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken3_;
  edm::EDGetTokenT<reco::CaloMETCollection> metToken4_;

  // map of MEs
  std::map<std::string, MonitorElement*> MEcontainer_;

  // methods to calculate MET over SumET and MET over Leading Jet Pt
  float movers(const reco::CaloMETCollection& metcollection);
  float moverl(const reco::CaloMETCollection& metcollection, float& ljpt);
};
#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
