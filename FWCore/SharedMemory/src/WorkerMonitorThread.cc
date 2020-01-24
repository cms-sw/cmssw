// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     WorkerMonitorThread
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <signal.h>
#include <iostream>
#include <unistd.h>

// user include files
#include "FWCore/SharedMemory/interface/WorkerMonitorThread.h"

//
// constants, enums and typedefs
//
using namespace edm::shared_memory;

//
// static data member definitions
//
std::atomic<bool> WorkerMonitorThread::s_signal_happened = false;
std::atomic<bool> WorkerMonitorThread::s_helperThreadDone = false;
int WorkerMonitorThread::s_sig = 0;

//
// constructors and destructor
//

//
// member functions
//
void WorkerMonitorThread::run() {
  //std::cerr << "Started cleanup thread\n";
  sigset_t ensemble;

  sigemptyset(&ensemble);
  sigaddset(&ensemble, SIGABRT);
  sigaddset(&ensemble, SIGILL);
  sigaddset(&ensemble, SIGBUS);
  sigaddset(&ensemble, SIGSEGV);
  sigaddset(&ensemble, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &ensemble, NULL);

  //std::cerr << "Start loop\n";
  helperReady_ = true;
  while (not stopRequested_.load()) {
    sleep(60);
    if (s_signal_happened) {
      if (actionSet_) {
        action_();
      }
      std::cerr << "Worker: SIGNAL CAUGHT " << s_sig << "\n";
      s_helperThreadDone = true;
      break;
    } else {
      //std::cerr << "SIGNAL woke\n";
    }
  }
  //std::cerr << "Ending cleanup thread\n";
}

void WorkerMonitorThread::startThread() {
  {
    //Setup watchdog thread for crashing signals

    //Need to use signal handler since signals generated
    // from within a program are thread specific which can
    // only be handed by a signal handler
    setupSignalHandling();

    std::thread t(&WorkerMonitorThread::run, this);
    t.detach();
    helperThread_ = std::move(t);
  }
  while (helperReady_.load() == false) {
  }
}

void WorkerMonitorThread::setupSignalHandling() {
  struct sigaction act;
  act.sa_sigaction = sig_handler;
  act.sa_flags = SA_SIGINFO;
  sigemptyset(&act.sa_mask);
  sigaction(SIGABRT, &act, nullptr);
  sigaction(SIGILL, &act, nullptr);
  sigaction(SIGBUS, &act, nullptr);
  sigaction(SIGSEGV, &act, nullptr);
  sigaction(SIGTERM, &act, nullptr);
}

//
// const member functions
//

//
// static member functions
//
void WorkerMonitorThread::sig_handler(int sig, siginfo_t*, void*) {
  s_sig = sig;
  s_signal_happened = true;
  while (not s_helperThreadDone) {
  };
  signal(sig, SIG_DFL);
  raise(sig);
}
