#ifndef JetAnalyzerBase_H
#define JetAnalyzerBase_H

/** \class JetAnalyzerBase
 *
 *  base class for all DQM monitor sources
 *
 *  $Date: 2010/01/18 21:04:05 $
 *  $Revision: 1.2 $
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
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

class JetAnalyzerBase {
 public:

  /// Constructor
  //  JetAnalyzerBase(JetServiceProxy *theServ):theService(theServ){}
  JetAnalyzerBase() {}
  
  /// Destructor
  virtual ~JetAnalyzerBase() {}
  
  /// Inizialize parameters for histo binning
  virtual void beginJob(DQMStore * dbe)= 0;

  /// Get the analysis of the muon properties
  void analyze(const edm::Event&, const edm::EventSetup&, reco::CaloJet& caloJet){}

  //  JetServiceProxy* service() {return theService;}

 private:
  // ----------member data ---------------------------
  //  JetServiceProxy *theService;
};
#endif  
