#ifndef RPCEfficiencyFromTrack_h
#define RPCEfficiencyFromTrack_h

/**********************************************
 *                                            *
 *           Giuseppe Roselli                 *
 *         INFN, Sezione di Bari              *
 *      Via Amendola 173, 70126 Bari          *
 *         Phone: +390805443218               *
 *      giuseppe.roselli@ba.infn.it           *
 *                                            *
 *                                            *
 **********************************************/

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "FWCore/ParameterSet/interface/InputTag.h"

#include<string>
#include<map>
#include<fstream>

class RPCDetId;
class TFile;
class TH1F;
class TFile;
class TCanvas;
class TH2F;
class Trajectory;
class Propagator;
class GeomDet;
class TrajectoryStateOnSurface;

class RPCEfficiencyFromTrack : public edm::EDAnalyzer {
   public:
      explicit RPCEfficiencyFromTrack(const edm::ParameterSet&);
      ~RPCEfficiencyFromTrack();
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      typedef std::vector<Trajectory> Trajectories;
      std::map<std::string, MonitorElement*> bookDetUnitTrackEff(RPCDetId & detId);
 
   private:

      TFile* fOutputFile;
      TH1F* hRecPt;
      TH1F* hGlobalRes;
      TH1F* histoMean;
      TH1F* hGlobalPull;
      double maxRes;
      bool MeasureEndCap;
      bool MeasureBarrel;
      bool EffSaveRootFile;
      int EffSaveRootFileEventsInterval;
      int ringSelection;
      bool selectwheel;

      std::string EffRootFileName;
      std::string TjInput;
      std::string RPCDataLabel;
      std::string GlobalRootLabel;
      std::map<std::string, std::map<std::string, MonitorElement*> >  meCollection;
      std::string thePropagatorName;
      mutable Propagator* thePropagator;
      DaqMonitorBEInterface * dbe;

      std::vector<std::string> _idList;
      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;
};
#endif
