#ifndef HLTrigger_JetMET_HLTCleanedJetVBFFilter_h
#define HLTrigger_JetMET_HLTCleanedJetVBFFilter_h

/** \class HLTCleanedJetVBFFilter
 *
 *  \author Andrea Benaglia
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"


class HLTCleanedJetVBFFilter : public HLTFilter
{
public:

  //! ctor
  explicit HLTCleanedJetVBFFilter(const edm::ParameterSet&);
  
  //! dotr
  ~HLTCleanedJetVBFFilter();
  
  //! the filter method
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
private:
  
  edm::InputTag inputJetTag_;
  edm::InputTag inputEleTag_;
  
  bool saveTag_;
  double minCleaningDR_;
  double minJetEtHigh_;
  double minJetEtLow_;
  double minJetDeta_;
};

#endif // HLTrigger_JetMET_HLTCleanedJetVBFFilter_h
