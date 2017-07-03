// Name: ptHatFilter.h
// Description:  Class header to filter events in a range of Monte Carlo ptHat.
// Author: R. Harris
// Date:  28 - October - 2008
#ifndef ptHatFilter_h
#define ptHatFilter_h
#include "FWCore/Framework/interface/EDFilter.h"
class ptHatFilter : public edm::EDFilter 
   {
     public:
       ptHatFilter(const edm::ParameterSet&);
       ~ptHatFilter() override;
       void beginJob() override;
       bool filter(edm::Event& e, edm::EventSetup const& iSetup) override;
       void endJob() override;       
     private:
       double ptHatLowerCut;
       double ptHatUpperCut;
       int  totalEvents;
       int  acceptedEvents;
   };
#endif
