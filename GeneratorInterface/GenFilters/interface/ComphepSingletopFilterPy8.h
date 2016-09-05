#ifndef ComphepSingletopFilterPy8_h
#define ComphepSingletopFilterPy8_h


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//


class ComphepSingletopFilterPy8 : public edm::EDFilter {
public:
    explicit ComphepSingletopFilterPy8(const edm::ParameterSet&);
    ~ComphepSingletopFilterPy8();
private:
    virtual void beginJob() ;
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
private:
//     edm::InputTag hepMCProductTag;
    double ptsep;
    int read22, read23, pass22, pass23, hardLep;
};

#endif
