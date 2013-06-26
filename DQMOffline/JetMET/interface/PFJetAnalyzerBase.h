#ifndef PFJetAnalyzerBase_H
#define PFJetAnalyzerBase_H

/** \class PFJetAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2010/01/20 19:12:43 $
 *  $Revision: 1.3 $
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
//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

class PFJetAnalyzerBase {
 public:

  /// Constructor
  PFJetAnalyzerBase() {}
  
  /// Destructor
  virtual ~PFJetAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(DQMStore * dbe)= 0;

  /// Get the analysis of the muon properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::PFJet& jet){}

};
#endif  
