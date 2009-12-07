#include "Fireworks/Core/interface/FWFileEntry.h"
#include "TFile.h"
#include "TError.h"
#include "TTree.h"

FWFileEntry::FWFileEntry(const std::string& name) :
   m_name(name),m_file(0), m_eventTree(0), m_event(0)
{
   openFile();
}

bool FWFileEntry::openFile(){
   gErrorIgnoreLevel = 3000; // suppress warnings about missing dictionaries
   TFile *newFile = TFile::Open(m_name.c_str());
   if (newFile == 0 || newFile->IsZombie() || !newFile->Get("Events")) {
      std::cout << "Invalid file. Ignored." << std::endl;
      return false;
   }
   gErrorIgnoreLevel = -1;
   m_file = newFile;
   m_event = new fwlite::Event(m_file);
   m_eventTree = dynamic_cast<TTree*>(m_file->Get("Events"));
   assert(m_eventTree!=0 && "Cannot find TTree 'Events' in the data file");
   return true;
}

void FWFileEntry::closeFile()
{
   if (m_file) {
      m_file->Close();
      delete m_file;
   }
   if (m_event) delete m_event;
}
