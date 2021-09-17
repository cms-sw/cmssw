// ----------------------------------------------------------------
// STANDALONE (non-CMSSW) producer for L1 tracking
// ----------------------------------------------------------------
// ROOT includes
#include <TMath.h>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TBranch.h>
#include <TSystem.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TLegend.h>
#include <TLatex.h>

#include <iostream>

#include "../interface/IMATH_TrackletCalculator.h"
#include "../interface/IMATH_TrackletCalculatorDisk.h"
#include "../interface/IMATH_TrackletCalculatorOverlap.h"

#include "../interface/SLHCEvent.h"
#include "../interface/Track.h"
#include "../interface/Settings.h"
#include "../interface/TrackletEventProcessor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <iomanip>
#include <fstream>
#include <string>

//Uncomment if you want root output
//#define USEROOT

// Include file to define ROOT-Tree
// --------------------------------
#ifdef USEROOT
#include "FPGAEvent.h"
#endif
// --------------------------------

using namespace trklet;
using namespace std;

int main(const int argc, const char **argv) {
  trklet::Settings settings;

  // ---------------------------------------------------------
  // these are options that are read from python configuration files for the CMSSW running, set manually for the standalone version

  settings.setFitPatternFile("../data/fitpattern.txt");
  settings.setProcessingModulesFile("../data/processingmodules_" + settings.geomext() + ".dat");
  settings.setMemoryModulesFile("../data/memorymodules_" + settings.geomext() + ".dat");
  settings.setWiresFile("../data/wires_" + settings.geomext() + ".dat");

  if (settings.extended()) {
    settings.setTableTEDFile("../data/table_TED/table_TED_Dummy.txt");
    settings.setTableTREFile("../data/table_TRE/table_TRE_Dummy.txt");
  }

  edm::LogVerbatim("Tracklet") << "fit pattern :     " << settings.fitPatternFile();
  edm::LogVerbatim("Tracklet") << "process modules : " << settings.processingModulesFile();
  edm::LogVerbatim("Tracklet") << "memory modules :  " << settings.memoryModulesFile();
  edm::LogVerbatim("Tracklet") << "wires          :  " << settings.wiresFile();

  // ---------------------------------------------------------

  TrackletEventProcessor eventProcessor;
  eventProcessor.init(settings);

  if (argc < 3)
    edm::LogVerbatim("Tracklet") << "Need to specify the input ascii file and the number of events to run on!";

  int nevents = atoi(argv[2]);

  ifstream infile;
  istream *in = &cin;
  if (strcmp(argv[1], "stdin")) {
    infile.open(argv[1]);
    in = &infile;
  }

  ofstream outpars;
  if (settings.writeMonitorData("Pars"))
    outpars.open("trackpars.txt");

    // ---------------------------------------------------------
    // Open file to hold ROOT-Tree
#ifdef USEROOT
  TFile *hfile = new TFile("myTest.root", "RECREATE", "Simple ROOT Ntuple");
  TTree *trackTree = new TTree("FPGAEvent", "L1Track Tree");
  FPGAEvent *fpgaEvent = new FPGAEvent;
  fpgaEvent->reset();
  trackTree->Branch("Event", &fpgaEvent);
#endif
  // ---------------------------------------------------------

  if (settings.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::out);
    fout.close();
  }

  for (int eventnum = 0; eventnum < nevents && !in->eof(); eventnum++) {
    SLHCEvent ev(*in);

    //auto simtrk = ev.simtrack(0);
    //if (std::abs(std::abs(simtrk.eta())-1.3)>0.1) continue;

    // -----------------------------------------------------------------
    // setup ROOT Tree and Add Monte Carlo tracks to the ROOT-Tree Event
#ifdef USEROOT
    fpgaEvent->reset();
    fpgaEvent->nevt = eventnum;
    for (int nst = 0; nst < ev.nsimtracks(); nst++) {
      simtrk = ev.simtrack(nst);
      FPGAEventMCTrack *mcTrack = new FPGAEventMCTrack(
          simtrk.type(), simtrk.pt(), simtrk.eta(), simtrk.phi(), simtrk.vx(), simtrk.vy(), simtrk.vz());
      fpgaEvent->mcTracks.push_back(*mcTrack);
    }
#endif
    // ------------------------------------------------------------------

    if (settings.writeMonitorData("Seeds")) {
      ofstream fout("seeds.txt", ofstream::app);
      fout << "======== Event " << eventnum << " ========" << endl;
      for (unsigned nst = 0; nst < ev.nsimtracks(); nst++) {
        const L1SimTrack &simtrk = ev.simtrack(nst);
        fout << "SimTrk " << simtrk.pt() << " " << simtrk.eta() << " " << simtrk.phi() << " " << simtrk.d0() << " ";

        vector<string> hitPattern;
        for (int i = 0; i < ev.nstubs(); i++) {
          const L1TStub &stub = ev.stub(i);
          if (!stub.tpmatch(simtrk.trackid()))
            continue;
          if (stub.layer() < 999) {
            switch (stub.layer()) {
              case 0:
                hitPattern.emplace_back("L1");
                break;
              case 1:
                hitPattern.emplace_back("L2");
                break;
              case 2:
                hitPattern.emplace_back("L3");
                break;
              case 3:
                hitPattern.emplace_back("L4");
                break;
              case 4:
                hitPattern.emplace_back("L5");
                break;
              case 5:
                hitPattern.emplace_back("L6");
                break;
              default:
                edm::LogVerbatim("Tracklet") << "Stub layer: " << stub.layer();
                assert(0);
            }
          } else {
            string d = (stub.isPSmodule() ? "D" : "d");
            switch (abs(stub.disk())) {
              case 1:
                hitPattern.push_back(d + "1");
                break;
              case 2:
                hitPattern.push_back(d + "2");
                break;
              case 3:
                hitPattern.push_back(d + "3");
                break;
              case 4:
                hitPattern.push_back(d + "4");
                break;
              case 5:
                hitPattern.push_back(d + "5");
                break;
              default:
                edm::LogVerbatim("Tracklet") << "Stub disk: " << stub.disk();
                assert(0);
            }
          }
        }
        bool (*compare)(const string &, const string &) = [](const string &a, const string &b) -> bool {
          if (a.at(0) == 'L' && b.at(0) == 'D')
            return true;
          else if (a.at(0) == 'D' && b.at(0) == 'L')
            return false;
          else
            return a.at(1) < b.at(1);
        };
        sort(hitPattern.begin(), hitPattern.end(), compare);
        hitPattern.erase(unique(hitPattern.begin(), hitPattern.end()), hitPattern.end());
        for (const auto &stub : hitPattern)
          fout << stub;
        if (hitPattern.empty())
          fout << "XX";
        fout << endl;
      }
      fout.close();
    }

    edm::LogVerbatim("Tracklet") << "Process event: " << eventnum << " with " << ev.nstubs() << " stubs and "
                                 << ev.nsimtracks() << " simtracks";

    eventProcessor.event(ev);

    const std::vector<Track> &tracks = eventProcessor.tracks();

    // ---------------------------------------------------------
    // Block for producing ROOT-Tree
#ifdef USEROOT
#include "FPGATree.icc"
#endif
    // ---------------------------------------------------------

    if (settings.writeMonitorData("MatchEff")) {
      static ofstream out("matcheff.txt");
      int nsim = 0;
      for (unsigned int isimtrack = 0; isimtrack < ev.nsimtracks(); isimtrack++) {
        L1SimTrack simtrack = ev.simtrack(isimtrack);
        if (simtrack.pt() < 2.0)
          continue;
        if (fabs(simtrack.eta()) > 2.4)
          continue;
        if (fabs(simtrack.vz()) > 15.0)
          continue;
        if (hypot(simtrack.vx(), simtrack.vy()) > 0.1)
          continue;
        bool electron = (abs(simtrack.type()) == 11);
        bool muon = (abs(simtrack.type()) == 13);
        bool pion = (abs(simtrack.type()) == 211);
        bool kaon = (abs(simtrack.type()) == 321);
        bool proton = (abs(simtrack.type()) == 2212);
        if (!(electron || muon || pion || kaon || proton))
          continue;
        int nlayers = 0;
        int ndisks = 0;
        int simeventid = simtrack.eventid();
        int simtrackid = simtrack.trackid();
        ev.layersHit(simtrackid, nlayers, ndisks);
        if (nlayers + ndisks < 4)
          continue;
        nsim++;
        for (int seed = -1; seed < 12; seed++) {
          bool eff = false;
          bool effloose = false;
          int itrackmatch = -1;
          for (unsigned int itrack = 0; itrack < tracks.size(); itrack++) {
            const std::vector<L1TStub> &stubs = tracks[itrack].stubs();
            if (seed == -1) {
              if (tracks[itrack].duplicate())
                continue;
            } else {
              if (seed != tracks[itrack].seed())
                continue;
            }

            unsigned int nmatch = 0;
            for (auto stub : stubs) {
              if (stub.tpmatch(simtrackid)) {
                nmatch++;
              }
            }

            if (nmatch == stubs.size()) {
              eff = true;
              itrackmatch = itrack;
            }
            if (nmatch >= stubs.size() - 1) {
              effloose = true;
              if (!eff)
                itrackmatch = itrack;
            }
          }
          double dpt = -999;
          double dphi = -999;
          double deta = -999;
          double dz0 = -999;
          int q = 1;
          if (simtrack.type() == 11 || simtrack.type() == 13 || simtrack.type() == -211 || simtrack.type() == -321 ||
              simtrack.type() == -2212) {
            q = -1;
          }

          if (itrackmatch >= 0) {
            dpt = tracks[itrackmatch].pt(settings) - q * simtrack.pt();
            dphi = tracks[itrackmatch].phi0(settings) - simtrack.phi();
            if (dphi > M_PI)
              dphi -= 2 * M_PI;
            if (dphi < -M_PI)
              dphi += 2 * M_PI;
            deta = tracks[itrackmatch].eta(settings) - simtrack.eta();
            dz0 = tracks[itrackmatch].z0(settings) - simtrack.vz();
          }

          out << eventnum << " " << simeventid << " " << seed << " " << simtrackid << " " << simtrack.type() << " "
              << simtrack.pt() << " " << simtrack.eta() << " " << simtrack.phi() << " " << simtrack.vx() << " "
              << simtrack.vy() << " " << simtrack.vz() << " " << eff << " " << effloose << " " << dpt << " " << dphi
              << " " << deta << " " << dz0 << endl;
        }
      }
    }

    int ntrack = 0;
    int i = 0;
    for (auto &track : tracks) {
      i++;
      if (settings.writeMonitorData("Pars")) {
        outpars << track.duplicate() << " " << track.eta(settings) << " " << track.phi0(settings) << " "
                << track.z0(settings) << " " << angle0to2pi::make0To2pi(track.phi0(settings)) / (2 * M_PI / N_SECTOR)
                << " " << track.rinv(settings);
      }
      if (!track.duplicate()) {
        ntrack++;
      }
    }

    edm::LogVerbatim("Tracklet") << "Number of found tracks : " << tracks.size() << " unique " << ntrack << " ";
  }

  eventProcessor.printSummary();
}
