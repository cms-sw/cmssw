#ifndef PFMETAnalyzerBase_H
#define PFMETAnalyzerBase_H

/** \class PFMETAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2008/09/19 15:05:05 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - The Rockefeller University
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
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class PFMETAnalyzerBase {
 public:

  /// Constructor
  PFMETAnalyzerBase() {}
  
  /// Destructor
  virtual ~PFMETAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(edm::EventSetup const& iSetup,  DQMStore* dbe)= 0;

  /// Get the analysis of the muon properties
    void analyze(const edm::Event&, const edm::EventSetup&, 
		 const edm::TriggerResults&,
                 reco::PFMET& pfMET){}

 private:
  // ----------member data ---------------------------
};
#endif  
