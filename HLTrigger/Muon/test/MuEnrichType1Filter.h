#ifndef MuEnrichFltr_h
#define MuEnrichFltr_h
//
// class declaration
//
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class MuEnrichType1Filter : public edm::EDFilter {
public:
  explicit MuEnrichType1Filter(const edm::ParameterSet&);
  ~MuEnrichType1Filter() override;
  
private:
  void beginJob() override ;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  edm::EDGetTokenT<edm::HepMCProduct> theGenToken;
  int nrejected;
  int naccepted;
  int type;
      // ----------member data ---------------------------
};

#endif

