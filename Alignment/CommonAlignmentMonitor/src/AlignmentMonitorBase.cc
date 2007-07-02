// -*- C++ -*-
//
// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Fri Mar 30 12:21:07 CDT 2007
// $Id: AlignmentMonitorBase.cc,v 1.1 2007/04/23 22:19:14 pivarski Exp $
//

// system include files

// user include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

// AlignmentMonitorBase::AlignmentMonitorBase(const AlignmentMonitorBase& rhs)
// {
//    // do actual copying here;
// }

AlignmentMonitorBase::AlignmentMonitorBase(const edm::ParameterSet& cfg): m_iteration(0), mp_tracker(0), mp_muon(0), mp_store(0), m_collectorActive(false), m_collectorNJobs(0), m_collectorDone(false), mp_file(NULL), mp_iterDir(NULL) {
   m_outpath = cfg.getParameter<std::string>("outpath");
   assert(m_outpath.at(m_outpath.size()-1) == '/');
   m_outfile = cfg.getParameter<std::string>("outfile");

   m_collectorActive = cfg.getParameter<bool>("collectorActive");
   m_collectorNJobs = cfg.getParameter<int>("collectorNJobs");
   m_collectorPath = cfg.getParameter<std::string>("collectorPath");
}

//
// assignment operators
//
// const AlignmentMonitorBase& AlignmentMonitorBase::operator=(const AlignmentMonitorBase& rhs)
// {
//   //An exception safe implementation is
//   AlignmentMonitorBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void AlignmentMonitorBase::beginOfJob(AlignableTracker *pTracker, AlignableMuon *pMuon,
				      AlignmentParameterStore *pStore) {
   mp_tracker = pTracker;
   mp_muon = pMuon;
   mp_store = pStore;

   if (!pMuon)          mp_navigator = new AlignableNavigator(pTracker);
   else if (!pTracker)  mp_navigator = new AlignableNavigator(pMuon);
   else                 mp_navigator = new AlignableNavigator(pTracker, pMuon);
}

void AlignmentMonitorBase::startingNewLoop() {
   if (m_collectorActive) {
      if (!m_collectorDone) collect();
      m_collectorDone = true;
   }
   else {
      if (!mp_file) mp_file = new TFile((m_outpath + m_outfile).c_str(), "update");
      assert(mp_file);

      m_iteration = 0;
      char iterStr[10];
      do {
	 m_iteration++;
	 sprintf(iterStr, "iter%d", m_iteration);
      } while (mp_file->Get(iterStr));

      mp_iterDir = mp_file->mkdir(iterStr, iterStr);

      gROOT->cd();
      book();
   }
}

void AlignmentMonitorBase::duringLoop(const edm::EventSetup &iSetup, const ConstTrajTrackPairCollection &iTrajTracks) {
   if (!m_collectorActive) event(iSetup, iTrajTracks);
}

void AlignmentMonitorBase::endOfLoop(const edm::EventSetup &iSetup) {
   if (!m_collectorActive) afterAlignment(iSetup);
   if (mp_file) {
      edm::LogInfo("AlignmentMonitorBase") << "Writing histograms for iteration " << iteration() << " to file." << std::endl;
      mp_file->Write(NULL, TObject::kWriteDelete);
   }
}

void AlignmentMonitorBase::endOfJob() {
   if (mp_file) mp_file->Close();
}

TObject *AlignmentMonitorBase::add(std::string dir, TObject *obj) {
   assert(dir.at(0) == '/'  &&  dir.at(dir.size()-1) == '/');

   TObject *output = obj;
   if (dir.substr(0, 7) == std::string("/iterN/")) {
      std::string subdir = dir.substr(6, dir.size());
      getDirectoryFromMap(subdir, true)->Append(obj);
   }
   else {
      TDirectory *tdir = getDirectoryFromMap(dir, false);
      if (tdir->Get(obj->GetName())) {
	 output = tdir->Get(obj->GetName());
      }
      else {
	 tdir->Append(obj);
      }
   }

   gROOT->cd();
   return output;
}


TDirectory *AlignmentMonitorBase::getDirectoryFromMap(const std::string path, const bool isIter) {
   if (path == std::string("/")) {
      if (isIter) return mp_iterDir;
      return mp_file;
   }

   TObject *candidate;

   if (isIter && (candidate = mp_iterDir->Get(path.substr(1, path.size()-2).c_str()))) {
      if (std::string(candidate->ClassName()) == std::string("TDirectory")) {
	 return (TDirectory*)(candidate);
      }
      else throw cms::Exception("BadConfig") << "Root directory " << path << " isn't a TDirectory, it's a " << candidate->ClassName();
   }

   if (!isIter && (candidate = mp_file->Get(path.substr(1, path.size()-2).c_str()))) {
      if (std::string(candidate->ClassName()) == std::string("TDirectory")) {
	 return (TDirectory*)(candidate);
      }
      else throw cms::Exception("BadConfig") << "Root directory " << path << " isn't a TDirectory, it's a " << candidate->ClassName();
   }

   int slash = path.rfind("/", path.size()-2)+1;
   std::string parentPath(path.substr(0, slash));
   std::string name(path.substr(slash, path.size() - slash-1));
   return getDirectoryFromMap(parentPath, isIter)->mkdir(name.c_str());
}

int AlignmentMonitorBase::iterationNumber(const std::string &path) {
   if (path.size() < 4  ||  path.substr(0, 4) != std::string("iter")) {
      return 0;
   }

   int slash = path.find("/", 0);
   std::string number = path.substr(4, slash-4);

   int output = 0;
   for (unsigned int place = 0;  place < number.size();  place++) {
      output *= 10;
      output += number.at(place) - '0';
   }
   return output;
}

void AlignmentMonitorBase::collectAllHists(const TDirectory *dir, std::map<std::string, std::vector<TH1*> > &allHists, int &highestIter) {
   TIterator *iter = dir->GetListOfKeys()->MakeIterator();
   while (TKey *key = (TKey*)(iter->Next())) {
      TObject *obj = key->ReadObj();

      if (obj->InheritsFrom("TDirectory")) collectAllHists((TDirectory*)(obj), allHists, highestIter);

      else if (obj->InheritsFrom("TH1")) {
	 std::string path(dir->GetPath());
	 int colon = path.rfind(":") + 2;
	 std::string formatted = path.substr(colon, path.size()-colon);

	 if (formatted == std::string("")) formatted = std::string(obj->GetName());
	 else formatted += std::string("/") + std::string(obj->GetName());

	 if (allHists.find(formatted) == allHists.end()) {
	    allHists[formatted] = std::vector<TH1*>();
	 }
	 allHists[formatted].push_back((TH1*)(obj));

	 int num = iterationNumber(formatted);
	 if (num > highestIter) highestIter = num;
      }
   }
}

void AlignmentMonitorBase::collect() {
   std::map<std::string, std::vector<TH1*> > allHists;

   int highestIter = 0;
   char iterStr[10];
   for (int job = 1;  job <= m_collectorNJobs;  job++) {
      sprintf(iterStr, "%d", job);
      std::string fileName = m_collectorPath + std::string("/job") + std::string(iterStr) + std::string("/") + m_outfile;

      collectAllHists(new TFile(fileName.c_str()), allHists, highestIter);
   }

   mp_file = new TFile((m_outpath + m_outfile).c_str(), "recreate");
   assert(mp_file);
   
   for (m_iteration = highestIter;  m_iteration >= 0;  m_iteration--) {
      if (m_iteration != 0) {
	 sprintf(iterStr, "iter%d", m_iteration);
	 mp_iterDir = mp_file->mkdir(iterStr, iterStr);
      }

      gROOT->cd();
      book();

      for (std::map<std::string, std::vector<TH1*> >::const_iterator mit = allHists.begin();  mit != allHists.end();  ++mit) {
	 if (iterationNumber(mit->first) == m_iteration) {
	    TH1 *h = dynamic_cast<TH1*>(mp_file->Get(mit->first.c_str()));
	    assert(h);

	    TList tl;
	    for (std::vector<TH1*>::const_iterator vit = mit->second.begin();  vit != mit->second.end();  ++vit) {
	       tl.Add(*vit);
	    }

	    h->Merge(&tl);
	 }
      }

      edm::LogInfo("AlignmentMonitorBase") << "Writing all histograms to file.  This can take many minutes if you booked a LOT of histograms." << std::endl;
      mp_file->Write();
      mp_file->Close();
      if (m_iteration > 0) mp_file = new TFile((m_outpath + m_outfile).c_str(), "update");
   }

   mp_file = NULL;
   mp_iterDir = NULL;
}

//
// const member functions
//

//
// static member functions
//
