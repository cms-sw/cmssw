#ifndef CaloMETAnalyzerBase_H
#define CaloMETAnalyzerBase_H

/** \class CaloMETAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2010/02/24 19:08:53 $
 *  $Revision: 1.6 $
 *  \author F. Chlebana - Fermilab
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class CaloMETAnalyzerBase {
 public:

  /// Constructor
  CaloMETAnalyzerBase() {}
  
  /// Destructor
  virtual ~CaloMETAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(DQMStore * dbe)= 0;

  /// Get the analysis of the muon properties
    void analyze(const edm::Event&, const edm::EventSetup&, 
		 const edm::TriggerResults&,
                 const reco::CaloMET& caloMET){}

 private:
  // ----------member data ---------------------------
};
#endif  
