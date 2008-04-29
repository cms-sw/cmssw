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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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
      TH2F* ExtrapError;
      TH1F* EffGlob1;  TH1F* EffGlob2;  TH1F* EffGlob3;  TH1F* EffGlob4;  TH1F* EffGlob5;  TH1F* EffGlob6; 
      TH1F* EffGlob7;  TH1F* EffGlob8;  TH1F* EffGlob9;  TH1F* EffGlob10;  TH1F* EffGlob11;  TH1F* EffGlob12; 
      int wh;
      bool cosmic;
      int Run;
      time_t aTime;

      ofstream* effres;
      std::string EffRootFileName;
      std::string TjInput;
      std::string RPCDataLabel;
      std::string GlobalRootLabel;
      std::map<std::string, std::map<std::string, MonitorElement*> >  meCollection;
      std::string thePropagatorName;
      mutable Propagator* thePropagator;

      std::vector<std::string> _idList;
      std::vector<std::map<RPCDetId, int> > counter;
      std::vector<int> totalcounter;

      DQMStore * dbe;
};
#endif
