#ifndef HLTLeptonTauNonCollProducer_h
#define HLTLeptonTauNonCollProducer_h

/** \class HLTLeptonTauNonCollProducer
 *
 *  
 *  This class is an EDFilter implementing tagged multijet trigger
 *  (e.g., b or tau). It should be run after the normal multijet
 *  trigger.
 *
 *  $Date: 2007/01/31 10:48:44 $
 *  $Revision: 1.1 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <map>
#include <vector>
//
// class declaration
//

class HLTLeptonTauNonCollProducer : public edm::EDProducer {

public:
  explicit HLTLeptonTauNonCollProducer(const edm::ParameterSet&);
  ~HLTLeptonTauNonCollProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc_;
  edm::InputTag leptonTag_;                 // Module label of input JetTagCollection
  double min_dphi_;
  double min_deta_;
  std::string label_;                               // Label of this filter in configuration file.
};

#endif //HLTLeptonTauNonCollProducer_h
