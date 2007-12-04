#ifndef HLTAnalFiltNDM_h
#define HLTAnalFiltNDM_h

/** \class HLTAnalFiltNDM
 *
 *  
 *  This class is an EDAnalyzer implementing a very basic HLT filter
 *  product analysis
 *
 *  $Date: 2006/10/04 16:02:42 $
 *  $Revision: 1.13 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class HLTAnalFiltNDM : public edm::EDAnalyzer {

   public:
      explicit HLTAnalFiltNDM(const edm::ParameterSet&);
      ~HLTAnalFiltNDM();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_;  // input tag identifying product to analyze

};

#endif //HLTAnalFiltNDM_h
