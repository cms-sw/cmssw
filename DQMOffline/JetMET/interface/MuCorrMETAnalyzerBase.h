#ifndef MuCorrMETAnalyzerBase_H
#define MuCorrMETAnalyzerBase_H

/** \class MuCorrMETAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2010/02/17 17:49:47 $
 *  $Revision: 1.5 $
 *  \author A.Apresyan Caltech
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
//#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include <DataFormats/METReco/interface/CaloMET.h>
//#include <DataFormats/METReco/interface/CaloMETCollection.h>
#include "DataFormats/METReco/interface/CaloMETFwd.h"

//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class MuCorrMETAnalyzerBase {
 public:

  /// Constructor
  MuCorrMETAnalyzerBase() {}
  
  /// Destructor
  virtual ~MuCorrMETAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(DQMStore * dbe)= 0;

  /// Get the analysis of the muon properties
    void analyze(const edm::Event&, const edm::EventSetup&, 
		 const edm::TriggerResults&,
                 reco::CaloMET& muCorrMET){}

 private:
  // ----------member data ---------------------------
};
#endif  
