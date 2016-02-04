/*HLTTauRefCombiner
Producer that Combines LV collection to common denominator objects 
for matching in the Tau HLT

*/

#ifndef HLTTauRefCombiner_h
#define HLTTauRefCombiner_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include <string>

class HLTTauRefCombiner : public edm::EDProducer {

public:
  explicit HLTTauRefCombiner(const edm::ParameterSet&);
  ~HLTTauRefCombiner();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  typedef math::XYZTLorentzVectorD LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;

  std::vector<edm::InputTag> inputColl_; //Input LV Collections
  double matchDeltaR_; //Delta R for matching
  std::string outName_; //outputObjectName

 bool match(const LorentzVector&,const LorentzVectorCollection&);//See if this Jet Is Matched


};

#endif
