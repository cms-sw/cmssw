#ifndef MCPdgIndexFilter_h
#define MCPdgIndexFilter_h
/*
 Description: filter events based on the particle PDG ID at a given
 index in the HepMC::GenEvent record.

 Original Author: Burt Betchart, 2013/08/09
*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class MCPdgIndexFilter : public edm::EDFilter {
   public:
      explicit MCPdgIndexFilter(const edm::ParameterSet&);
      ~MCPdgIndexFilter() {};

      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      bool pass(const edm::Event&);
      const std::string label_;
      const std::vector<int> pdgID;
      const std::vector<unsigned> index;
      const unsigned maxIndex;
      const bool taggingMode;
      const std::string tag;
};
#endif
