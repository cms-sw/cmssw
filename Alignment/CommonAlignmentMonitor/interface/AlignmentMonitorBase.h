#ifndef CommonAlignmentMonitor_AlignmentMonitorBase_h
#define CommonAlignmentMonitor_AlignmentMonitorBase_h
// -*- C++ -*-
//
// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorBase
// 
/**\class AlignmentMonitorBase AlignmentMonitorBase.h Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar 30 12:21:02 CDT 2007
// $Id: AlignmentMonitorBase.h,v 1.2 2007/05/09 07:06:32 fronga Exp $
//

// system include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <stdio.h>
#include <map>
#include <string>

// user include files
#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TObject.h"
#include "TList.h"
#include "TIterator.h"
#include "TKey.h"

// forward declarations

class AlignmentMonitorBase {
   public:
      typedef std::pair<const Trajectory*, const reco::Track*> ConstTrajTrackPair; 
      typedef std::vector<ConstTrajTrackPair>  ConstTrajTrackPairCollection;

      /// Constructor
      AlignmentMonitorBase(const edm::ParameterSet &cfg);
      
      /// Destructor
      virtual ~AlignmentMonitorBase() {}

      /// Called at beginning of job: don't reimplement
      void beginOfJob(AlignableTracker *pTracker, AlignableMuon *pMuon,
		      AlignmentParameterStore *pStore);

      /// Called at beginning of loop: don't reimplement
      void startingNewLoop();

      /// Called for each event: don't reimplement
      void duringLoop(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection &iTrajTracks);

      /// Called at end of loop: don't reimplement
      void endOfLoop(const edm::EventSetup &iSetup);

      /// Called at end of processing: don't implement
      void endOfJob();

      /////////////////////////////////////////////////////

      /// Book or retrieve histograms; MUST be reimplemented
      virtual void book() = 0;

      /// Called for each event (by "run()"): may be reimplemented
      virtual void event(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection &iTrajTracks) { };

      /// Called after updating AlignableTracker and AlignableMuon (by
      /// "endOfLoop()"): may be reimplemented
      virtual void afterAlignment(const edm::EventSetup &iSetup) { };

   protected:
      /// Use this every time you book a histogram (so that
      /// AlignmentMonitorBase can find your histograms in a
      /// collector (parallel-processing) job)
      TObject *add(std::string dir, TObject *obj);

      int                     iteration()    { return m_iteration; };
      AlignableTracker        *pTracker()    { return mp_tracker; };
      AlignableMuon           *pMuon()       { return mp_muon; };
      AlignmentParameterStore *pStore()      { return mp_store; };
      AlignableNavigator      *pNavigator()  { return mp_navigator; };

   private:
      AlignmentMonitorBase(const AlignmentMonitorBase&); // stop default
      const AlignmentMonitorBase& operator=(const AlignmentMonitorBase&); // stop default

      TDirectory *getDirectoryFromMap(const std::string path, const bool isIter);
      int iterationNumber(const std::string &path);
      void collectAllHists(const TDirectory *dir, std::map<std::string, std::vector<TH1*> > &allHists, int &highestIter);
      void collect();

      // ---------- member data --------------------------------

      int m_iteration;
      AlignableTracker         *mp_tracker;
      AlignableMuon            *mp_muon;
      AlignmentParameterStore  *mp_store;
      AlignableNavigator       *mp_navigator;

      std::string m_outpath, m_outfile;
      bool m_collectorActive;
      int m_collectorNJobs;
      std::string m_collectorPath;
      bool m_collectorDone;

      TFile *mp_file;
      TDirectory *mp_iterDir;

      std::vector<TObject*> m_inSlashDir, m_inIterDir;
};


#endif
