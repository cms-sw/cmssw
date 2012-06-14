#ifndef DQM_L1TMONITORCLIENT_L1TDTTF_H
#define DQM_L1TMONITORCLIENT_L1TDTTF_H

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

//
// class declaration
//

class TH1F;
class TH2F;
class DQMStore;
class MonitorElement;


class L1TDTTFClient: public edm::EDAnalyzer {

public:

  /// Constructor
  L1TDTTFClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1TDTTFClient();
 
protected:

  /// BeginJob
  void beginJob(void);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  void makeSummary();

private:
  
  std::string l1tdttffolder_;
  edm::InputTag dttfSource_;
  bool online_;
  bool verbose_;
  int resetafterlumi_;
  int counterLS_;      ///counter
  TH2F * occupancy_r_;
  DQMStore * dbe_;

  std::string wheel_[6];
  std::string wheelpath_[6];
  std::string inclusivepath_;
  std::string gmtpath_;
  std::string testpath_;

  MonitorElement* dttf_nTracks_integ;
  MonitorElement* dttf_occupancySummary;
  MonitorElement* dttf_bx_summary;
  MonitorElement* dttf_bx_integ;
  MonitorElement* dttf_eta_fine_integ;
  MonitorElement* dttf_quality_integ;
  MonitorElement* dttf_quality_summary;
  MonitorElement* dttf_highQual_Summary;
  MonitorElement* dttf_phi_eta_coarse_integ;
  MonitorElement* dttf_phi_eta_fine_integ;
  MonitorElement* dttf_phi_eta_integ;
  MonitorElement* dttf_eta_fine_fraction;
  MonitorElement* dttf_phi_integ;
  MonitorElement* dttf_pt_integ;
  MonitorElement* dttf_eta_integ;
  MonitorElement* dttf_q_integ;

  MonitorElement* dttf_gmt_matching;
  MonitorElement* dttf_2ndTrack_Summary;

  MonitorElement* dttf_occupancySummary_test;

  MonitorElement* dttf_nTracks_integ_2ndTrack;
  MonitorElement* dttf_occupancySummary_2ndTrack;
  MonitorElement* dttf_bx_summary_2ndTrack;
  MonitorElement* dttf_bx_integ_2ndTrack;
  MonitorElement* dttf_quality_integ_2ndTrack;
  MonitorElement* dttf_quality_summary_2ndTrack;
  MonitorElement* dttf_highQual_Summary_2ndTrack;
  MonitorElement* dttf_phi_eta_integ_2ndTrack;
  MonitorElement* dttf_eta_integ_2ndTrack;
  MonitorElement* dttf_phi_integ_2ndTrack;
  MonitorElement* dttf_pt_integ_2ndTrack;
  MonitorElement* dttf_q_integ_2ndTrack;

  MonitorElement* dttf_nTracks_wheel[6];
  MonitorElement* dttf_bx_wheel_summary[6];
  MonitorElement* dttf_bx_wheel_integ[6];
  MonitorElement* dttf_quality_wheel[6];
  MonitorElement* dttf_quality_summary_wheel[6];
  MonitorElement* dttf_fine_fraction_wh[6];
  MonitorElement* dttf_eta_wheel[6];
  MonitorElement* dttf_phi_wheel[6];
  MonitorElement* dttf_pt_wheel[6];
  MonitorElement* dttf_q_wheel[6];

  MonitorElement* dttf_nTracks_wheel_2ndTrack[6];
  MonitorElement* dttf_bx_wheel_summary_2ndTrack[6];
  MonitorElement* dttf_bx_wheel_integ_2ndTrack[6];

  TH1F * getTH1F(const char * hname);
  TH2F * getTH2F(const char * hname);

  void setMapLabel(MonitorElement *me);

  void buildHighQualityPlot( TH2F * occupancySummary,
			     MonitorElement * highQual_Summary,
			     const std::string & path );

  void buildPhiEtaPlotOFC( MonitorElement * phi_eta_fine_integ,
			   MonitorElement * phi_eta_coarse_integ,
			   MonitorElement * phi_eta_integ,
			   const std::string & path_fine,
			   const std::string & path_coarse,
			   int wh );

  void buildPhiEtaPlotO( MonitorElement * phi_eta_integ,
			 const std::string & path,
			 int wh );

/*  void buildPhiEtaPlot( MonitorElement * phi_eta_integ, */
/* 			const std::string & path, */
/* 			int wh ); */

/*   void buildPhiEtaPlotFC( MonitorElement * phi_eta_fine_integ, */
/* 			  MonitorElement * phi_eta_coarse_integ, */
/* 			  MonitorElement * phi_eta_integ, */
/* 			  const std::string & path_fine, */
/* 			  const std::string & path_coarse, */
/* 			  int wh ); */

  void buildSummaries();
  void setGMTsummary();

  void setWheelLabel(MonitorElement *me);
  void setQualLabel(MonitorElement *me, int axis);

  template<typename T>
    void normalize( T * me )
    {
      double scale = me->Integral();
      if ( scale > 0 ) {
	normalize( me, 1./scale, scale );
      }
    }

  template<typename T>
    void normalize( T * me, const double & scale )
    {
      normalize( me, scale, me->Integral() );
    }


  template<typename T>
    void normalize( T * me, const double & scale, const double & entries )
    {
      me->SetEntries( entries );
      me->Scale( scale );
    }



};

#endif
