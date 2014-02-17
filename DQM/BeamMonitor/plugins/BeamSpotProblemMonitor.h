#ifndef BeamSpotProblemMonitor_H
#define BeamSpotProblemMonitor_H

/** \class BeamSpotProblemMonitor
 * *
 *  $Date: 2012/05/22 19:44:16 $
 *  $Revision: 1.1 $
 *  \author  Sushil S. Chauhan/UC Davis
 *      
 *
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include <fstream>


//
// class declaration
//

class BeamSpotProblemMonitor : public edm::EDAnalyzer {
  public:

    BeamSpotProblemMonitor( const edm::ParameterSet& );
    ~BeamSpotProblemMonitor();

  protected:

    //The order it runs


    // BeginJob
    void beginJob();

    // BeginRun
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& context) ;
    void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& c);
    // EndRun
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    // Endjob
    void endJob(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);

  
  private:

    void FillPlots(const edm::LuminosityBlock& lumiSeg,int&,int&,int&);
    edm::ParameterSet parameters_;
    std::string monitorName_;
    edm::InputTag scalertag_; // scalar colleciton
    edm::InputTag trkSrc_; //  track collection

    int Ntracks_;
    int nCosmicTrk_;
    int fitNLumi_;
    int intervalInSec_;
    bool debug_;
    bool onlineMode_;
    bool doTest_;
    int  alarmONThreshold_;
    int  alarmOFFThreshold_;

    DQMStore* dbe_;

    int lastlumi_; // previous LS processed
    int nextlumi_; // next LS of Fit
    int nFitElements_;
    bool processed_;

    //Alarm Variable  
    bool ALARM_ON_;
    double BeamSpotStatus_;  
    int BeamSpotFromDB_;
    bool dcsTk[6];

    // MonitorElements:
    std::map<TString, MonitorElement*> hs;
    MonitorElement * BeamSpotError;

};

#endif

