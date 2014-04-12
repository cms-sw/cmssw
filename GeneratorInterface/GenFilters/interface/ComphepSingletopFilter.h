#ifndef ComphepSingletopFilter_h
#define ComphepSingletopFilter_h


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//


class ComphepSingletopFilter : public edm::EDFilter {
public:
    explicit ComphepSingletopFilter(const edm::ParameterSet&);
    ~ComphepSingletopFilter();
private:
    virtual void beginJob() ;
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
private:
    double ptsep;
    unsigned int read22, read23;
    unsigned int pass22, pass23;
    edm::InputTag hepMCProductTag_;
};

#endif
