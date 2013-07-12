#ifndef MuonAlignmentSummary_H
#define MuonAlignmentSummary_H


/** \class MuonAlignmentSummary
 *
 *  DQM client for muon alignment summary
 *
 *  $Date: 2008/12/13 15:31:21 $
 *  $Revision: 1.1 $
 *  \author J. Fernandez - Univ. Oviedo <Javier.Fernandez@cern.ch>
 */

#include <math.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

namespace edm {
    class ParameterSet;
    class EventSetup;
    class InputTag;
}

class TH1F;

class MuonAlignmentSummary : public edm::EDAnalyzer {
public:

    /// Constructor
    MuonAlignmentSummary(const edm::ParameterSet&);
  
    /// Destructor
    virtual ~MuonAlignmentSummary();
  
    /// Inizialize parameters for histo binning
    void beginRun(edm::Run const& run,edm::EventSetup const& iSetup);

    /// Get the analysis
    void analyze(const edm::Event& event, const edm::EventSetup& iSetup){}

    /// Save the histos
    void endRun(edm::Run const& run, edm::EventSetup const& iSetup);

private:
    // ----------member data ---------------------------
  
    DQMStore* dbe;

    MonitorElement *hLocalPositionDT;
    MonitorElement *hLocalPositionRmsDT;
    MonitorElement *hLocalAngleDT;
    MonitorElement *hLocalAngleRmsDT;

    MonitorElement *hLocalXMeanDT;
    MonitorElement *hLocalXRmsDT;
    MonitorElement *hLocalYMeanDT;
    MonitorElement *hLocalYRmsDT;
    MonitorElement *hLocalPhiMeanDT;
    MonitorElement *hLocalPhiRmsDT;
    MonitorElement *hLocalThetaMeanDT;
    MonitorElement *hLocalThetaRmsDT;

    MonitorElement *hLocalPositionCSC;
    MonitorElement *hLocalPositionRmsCSC;
    MonitorElement *hLocalAngleCSC;
    MonitorElement *hLocalAngleRmsCSC;

    MonitorElement *hLocalXMeanCSC;
    MonitorElement *hLocalXRmsCSC;
    MonitorElement *hLocalYMeanCSC;
    MonitorElement *hLocalYRmsCSC;
    MonitorElement *hLocalPhiMeanCSC;
    MonitorElement *hLocalPhiRmsCSC;
    MonitorElement *hLocalThetaMeanCSC;
    MonitorElement *hLocalThetaRmsCSC;

    edm::ParameterSet parameters;

    // Switch for verbosity
    std::string metname;

    // mean and rms histos ranges
    double meanPositionRange,rmsPositionRange,meanAngleRange,rmsAngleRange;
    
    // flags to decide on subdetector and summary histograms
    bool doDT, doCSC;

    // Top folder in root file
    std::string MEFolderName;
    std::stringstream topFolder;

};
#endif  
