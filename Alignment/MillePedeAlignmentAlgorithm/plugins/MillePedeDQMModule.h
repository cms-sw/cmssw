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

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/*** MillePede ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"




class MillePedeDQMModule : public DQMEDHarvester {

  //========================== PUBLIC METHODS ==================================
  public: //====================================================================

    MillePedeDQMModule(const edm::ParameterSet&);
    virtual ~MillePedeDQMModule();




    
    virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &)  override;
    //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;

  //========================= PRIVATE METHODS ==================================
  private: //===================================================================

    void bookHistograms(DQMStore::IBooker&);

    void fillExpertHistos();

    void fillExpertHisto(MonitorElement* histo,
                         const double cut,
                         const double sigCut,
                         const double maxMoveCut,
                         const double maxErrorCut,
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
    double maxErrorCut_;

    // Histograms
    MonitorElement* h_xPos;
    MonitorElement* h_xRot;
    MonitorElement* h_yPos;
    MonitorElement* h_yRot;
    MonitorElement* h_zPos;
    MonitorElement* h_zRot;

};

// define this as a plug-in
DEFINE_FWK_MODULE(MillePedeDQMModule);

#endif /* Alignment_MillePedeAlignmentAlgorithm_MillePedeDQMModule_h */
