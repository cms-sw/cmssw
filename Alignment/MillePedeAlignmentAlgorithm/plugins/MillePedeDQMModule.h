#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h

/**
 * @package   Alignment/MillePedeAlignmentAlgorithm
 * @file      MillePedeDQMModule.h
 *
 * @author    Max Stark (max.stark@cern.ch)
 * @date      Oct 26, 2015
 *
 * @brief     DQM Plotter for PCL-Alignment
 */



/*** system includes ***/
#include <array>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*** DQM ***/
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/*** MillePede ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"



class MillePedeDQMModule : public DQMEDAnalyzer {

  //========================== PUBLIC METHODS ==================================
  public: //====================================================================

    MillePedeDQMModule(const edm::ParameterSet&);
    virtual ~MillePedeDQMModule();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    virtual void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                                edm::EventSetup const&) override;

    virtual void analyze(edm::Event const& e, edm::EventSetup const& es) {}

    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;

  //========================= PRIVATE METHODS ==================================
  private: //===================================================================

    void fillExpertHistos();

    void fillExpertHisto(MonitorElement* histos[],
                         const double cut,
                         std::array<double, 6> obs,
                         std::array<double, 6> obsErr);

  //========================== PRIVATE DATA ====================================
  //============================================================================

    const edm::ParameterSet& mpReaderConfig_;
    MillePedeFileReader mpReader;

    // Signifiance of movement must be above
    double sigCut_;
    // Cutoff in micro-meter & micro-rad
    double Xcut_, tXcut_;
    double Ycut_, tYcut_;
    double Zcut_, tZcut_;
    // maximum movement in micro-meter/rad
    double maxMoveCut_;


    // Histograms
    MonitorElement* h_xPos[4];
    MonitorElement* h_xRot[4];
    MonitorElement* h_yPos[4];
    MonitorElement* h_yRot[4];
    MonitorElement* h_zPos[4];
    MonitorElement* h_zRot[4];

};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeDQMModule);

#endif /* Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h */
