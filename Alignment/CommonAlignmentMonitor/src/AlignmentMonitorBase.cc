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
// $Id: AlignmentMonitorBase.cc,v 1.3 2007/07/09 12:35:23 pivarski Exp $
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
   if (m_outpath.at(m_outpath.size()-1) != '/') {
      throw cms::Exception("BadConfig") << "outpath must end in a slash";
   }
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
      if (!mp_file) {
	 throw cms::Exception("FileAccess") << "could not open \"" << (m_outpath + m_outfile) << "\"";
      }

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
      edm::LogWarning("AlignmentMonitorBase") << "Writing histograms for iteration " << iteration() << " to file.  This can take many minutes if you booked a LOT of histograms.";
      mp_file->Write(NULL, TObject::kWriteDelete);
   }

   edm::LogWarning("AlignmentMonitorBase") << "Deleting TObject pointers for histograms that are redrawn for each iteration.";
   for (std::vector<TObject*>::const_iterator iter = m_inIterDir.begin();  iter != m_inIterDir.end();  ++iter) {
      delete *iter;
   }
   m_inIterDir.clear();
   edm::LogWarning("AlignmentMonitorBase") << "Done with iteration " << iteration() << "!";
}

void AlignmentMonitorBase::endOfJob() {
   if (mp_file) mp_file->Close();

// Apparently, ROOT deletes these histograms when it Closes a file
//    std::cout << "endofjob right before detetes" << std::endl;
//    for (std::vector<TObject*>::const_iterator iter = m_inSlashDir.begin();  iter != m_inSlashDir.end();  ++iter) {
//       std::cout << "endofjob detete " << (*iter)->GetName() << std::endl;
//       delete *iter;
//    }
//    std::cout << "endofjob right after detetes" << std::endl;
//    m_inSlashDir.clear();
//    std::cout << "endofjob right after clear" << std::endl;
}

TObject *AlignmentMonitorBase::add(std::string dir, TObject *obj) {
   if (dir.at(0) != '/'  ||  dir.at(dir.size()-1) != '/') {
      throw cms::Exception("BadConfig") << "ROOT directory must begin and end with slashes in call to add()";
   }

   TObject *output = obj;

   if (dir.substr(0, 7) == std::string("/iterN/")) {
      m_inIterDir.push_back(obj);

      std::string subdir = dir.substr(6, dir.size());
      getDirectoryFromMap(subdir, true)->Append(obj);
   }
   else {
      TDirectory *tdir = getDirectoryFromMap(dir, false);

      if (tdir->Get(obj->GetName())) {
	 output = tdir->Get(obj->GetName());
	 delete obj;
      }
      else {
	 m_inSlashDir.push_back(obj);
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
   if (!mp_file) {
      throw cms::Exception("FileAccess") << "could not open \"" << (m_outpath + m_outfile) << "\"";
   }
   
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

	    if (!h) {
	       throw cms::Exception("FileAccess") << "ROOT object \"" << mp_file->Get(mit->first.c_str())->GetName() << "\" is not a TH1";
	    }

	    TList tl;
	    for (std::vector<TH1*>::const_iterator vit = mit->second.begin();  vit != mit->second.end();  ++vit) {
	       tl.Add(*vit);
	    }

	    h->Merge(&tl);
	 }
      }

      edm::LogWarning("AlignmentMonitorBase") << "Writing all histograms to file.  This can take many minutes if you booked a LOT of histograms." << std::endl;
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
