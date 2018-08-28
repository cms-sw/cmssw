#ifndef BeamSpotProblemMonitor_H
#define BeamSpotProblemMonitor_H

/** \class BeamSpotProblemMonitor
 * *
 *  \author  Sushil S. Chauhan/UC Davis
 *      
 *
 */
// C++
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


//
// class declaration
//

class BeamSpotProblemMonitor : public DQMEDAnalyzer {
  public:

    explicit BeamSpotProblemMonitor( const edm::ParameterSet& );
    static void fillDescriptions(edm::ConfigurationDescriptions& );

  protected:

    //The order it runs


    // BeginRun
    void bookHistograms(DQMStore::IBooker& i, const edm::Run& r, const edm::EventSetup& c) override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& context) override;
    void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
        const edm::EventSetup& c) override;
    // EndRun
    void endRun(const edm::Run& r, const edm::EventSetup& c) override;


  private:

    void fillPlots(int&,int&,int);
    std::string monitorName_;
    const edm::EDGetTokenT<DcsStatusCollection> dcsStatus_; // dcs status collection
    const edm::EDGetTokenT<BeamSpotOnlineCollection> scalertag_; // scalar collection
    const edm::EDGetTokenT<reco::TrackCollection> trkSrc_; //  track collection

    int nTracks_;
    const int nCosmicTrk_;
    const int fitNLumi_;
    const bool debug_;
    const bool onlineMode_;
    const bool doTest_;
    const int  alarmONThreshold_;
    const int  alarmOFFThreshold_;

    int lastlumi_; // previous LS processed
    int nextlumi_; // next LS of Fit
    bool processed_;

    //Alarm Variable
    bool alarmOn_;
    double beamSpotStatus_;
    int beamSpotFromDB_;

    // MonitorElements:
    MonitorElement * beamSpotStatusLumi_ = nullptr;
    MonitorElement * beamSpotStatusLumiAll_ = nullptr;
    MonitorElement * beamSpotError_ = nullptr;

};

#endif


// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
