#ifndef Subsystem_Package_FWTEveViewer_h
#define Subsystem_Package_FWTEveViewer_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FWTEveViewer
// 
/**\class FWTEveViewer FWTEveViewer.h "FWTEveViewer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue, 03 Feb 2015 21:46:04 GMT
//

// system include files

#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>

// user include files

#include "TEveViewer.h"

// forward declarations

class FWTGLViewer;


class FWTEveViewer : public TEveViewer
{

public:
   FWTEveViewer(const char* n="FWTEveViewer", const char* t="");
   virtual ~FWTEveViewer();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   static bool SavePng(const TString& file, UChar_t* xx, int ww, int hh);
   static bool SaveJpg(const TString& file, UChar_t* xx, int ww, int hh);

   // ---------- member functions ---------------------------

   FWTGLViewer* fwGlViewer() { return m_fwGlViewer; }

   FWTGLViewer* SpawnFWTGLViewer();

   std::future<int> CaptureAndSaveImage(const TString& file, int height=-1);

private:
   FWTEveViewer(const FWTEveViewer&); // stop default

   const FWTEveViewer& operator=(const FWTEveViewer&); // stop default

   void spawn_image_thread();

   // ---------- member data --------------------------------

   FWTGLViewer *m_fwGlViewer;

   std::vector<unsigned char> m_imgBuffer;

   TString                 m_name;
   int                     m_ww, m_hh;
   bool                    m_thr_exit = false;
   std::thread            *m_thr = 0;
   std::promise<int>       m_prom;
   std::mutex              m_moo;
   std::condition_variable m_cnd;
};


#endif
