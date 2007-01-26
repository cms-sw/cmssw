#ifndef HLTEgammaDoubleEtFilter_h
#define HLTEgammaDoubleEtFilter_h

/** \class HLTEgammaDoubleEtFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//
// class decleration
//

class HLTEgammaDoubleEtFilter : public HLTFilter {

 public:
  explicit HLTEgammaDoubleEtFilter(const edm::ParameterSet&);
  ~HLTEgammaDoubleEtFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);


  

 

 private:
  edm::InputTag candTag_; // input tag identifying product contains filtered candidates
  double etcut1_;           // Et threshold in GeV 
  double etcut2_;           // Et threshold in GeV 
  int    npaircut_;        // number of egammas required
};

#endif //HLTEgammaDoubleEtFilter_h
