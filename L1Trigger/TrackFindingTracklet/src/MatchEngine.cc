#include "L1Trigger/TrackFindingTracklet/interface/MatchEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CandidateMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <filesystem>

using namespace std;
using namespace trklet;

MatchEngine::MatchEngine(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global), luttable_(settings) {
  layerdisk_ = initLayerDisk(3);

  barrel_ = layerdisk_ < N_LAYER;

  luttable_.initBendMatch(layerdisk_);

  nrinv_ = NRINVBITS;
}

void MatchEngine::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "matchout") {
    auto* tmp = dynamic_cast<CandidateMatchMemory*>(memory);
    assert(tmp != nullptr);
    candmatches_ = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find output: " << output;
}

void MatchEngine::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "vmstubin") {
    auto* tmp = dynamic_cast<VMStubsMEMemory*>(memory);
    assert(tmp != nullptr);
    vmstubs_ = tmp;
    return;
  }
  if (input == "vmprojin") {
    auto* tmp = dynamic_cast<VMProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    vmprojs_ = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find input: " << input;
}

void MatchEngine::execute(unsigned int iSector) {
  unsigned int countall = 0;
  unsigned int countpass = 0;

  //bool print = (iSector == 3 && getName() == "ME_L3PHIC20");
  bool print = false;

  constexpr unsigned int kNBitsBuffer = 3;

  int writeindex = 0;
  int readindex = 0;
  std::pair<int, int> projbuffer[1 << kNBitsBuffer];  //iproj zbin

  //The next projection to read, the number of projections and flag if we have more projections to read
  int iproj = 0;
  int nproj = vmprojs_->nTracklets();
  bool moreproj = iproj < nproj;

  //Projection that is read from the buffer and compared to stubs
  int rzbin = 0;
  int projfinerz = 0;
  int projfinerzadj = 0;
  unsigned int projfinephi = 0;

  int projindex = 0;
  int projrinv = 0;
  bool isPSseed = false;

  //Number of stubs for current zbin and the stub being processed on this clock
  int nstubs = 0;
  int istub = 0;

  //Main processing loops starts here
  for (unsigned int istep = 0; istep < settings_.maxStep("ME"); istep++) {
    countall++;

    int writeindexplus = (writeindex + 1) % (1 << kNBitsBuffer);
    int writeindexplusplus = (writeindex + 2) % (1 << kNBitsBuffer);

    //Determine if buffer is full - or near full as a projection
    //can point to two z bins we might fill two slots in the buffer
    bool bufferfull = (writeindexplus == readindex) || (writeindexplusplus == readindex);

    //Determin if buffer is empty
    bool buffernotempty = (writeindex != readindex);

    //If we have more projections and the buffer is not full we read
    //next projection and put in buffer if there are stubs in the
    //memory the projection points to

    if ((!moreproj) && (!buffernotempty))
      break;

    if (moreproj && (!bufferfull)) {
      Tracklet* proj = vmprojs_->getTracklet(iproj);

      int iprojtmp = iproj;

      iproj++;
      moreproj = iproj < nproj;

      unsigned int rzfirst = proj->proj(layerdisk_).fpgarzbin1projvm().value();
      unsigned int rzlast = rzfirst;

      bool second = proj->proj(layerdisk_).fpgarzbin2projvm().value();

      if (second)
        rzlast += 1;

      //Check if there are stubs in the memory
      int nstubfirst = vmstubs_->nStubsBin(rzfirst);
      int nstublast = vmstubs_->nStubsBin(rzlast);
      bool savefirst = nstubfirst != 0;
      bool savelast = second && (nstublast != 0);

      int writeindextmp = writeindex;
      int writeindextmpplus = (writeindex + 1) % (1 << kNBitsBuffer);

      if (savefirst && savelast) {
        writeindex = writeindexplusplus;
      } else if (savefirst || savelast) {
        writeindex = writeindexplus;
      }

      if (savefirst) {  //TODO for HLS (make code logic simpler)
        std::pair<int, int> tmp(iprojtmp, rzfirst);
        projbuffer[writeindextmp] = tmp;
      }
      if (savelast) {
        std::pair<int, int> tmp(iprojtmp, rzlast + 100);  //TODO for HLS (fix flagging that this is second bin)
        if (savefirst) {
          projbuffer[writeindextmpplus] = tmp;
        } else {
          projbuffer[writeindextmp] = tmp;
        }
      }
    }

    //If the buffer is not empty we have a projection that we need to process.

    if (buffernotempty) {
      int istubtmp = istub;

      //New projection
      if (istub == 0) {
        projindex = projbuffer[readindex].first;
        rzbin = projbuffer[readindex].second;
        bool second = false;
        if (rzbin >= 100) {
          rzbin -= 100;
          second = true;
        }

        Tracklet* proj = vmprojs_->getTracklet(projindex);

        FPGAWord fpgafinephi = proj->proj(layerdisk_).fpgafinephivm();

        projfinephi = fpgafinephi.value();

        nstubs = vmstubs_->nStubsBin(rzbin);

        projfinerz = proj->proj(layerdisk_).fpgafinerzvm().value();

        projrinv = barrel_ ? ((1 << (nrinv_ - 1)) + ((-2 * proj->proj(layerdisk_).fpgaphiprojder().value()) >>
                                                     (proj->proj(layerdisk_).fpgaphiprojder().nbits() - (nrinv_ - 1))))
                           : proj->proj(layerdisk_).getBendIndex().value();
        assert(projrinv >= 0);
        if (settings_.extended() && projrinv == (1 << nrinv_)) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << "Extended tracking, projrinv:" << projrinv;
          }
          projrinv = (1 << nrinv_) - 1;
        }
        assert(projrinv < (1 << nrinv_));

        isPSseed = proj->PSseed();

        //Calculate fine z position
        if (second) {
          projfinerzadj = projfinerz - (1 << NFINERZBITS);
        } else {
          projfinerzadj = projfinerz;
        }
        if (nstubs == 1) {
          istub = 0;
          readindex = (readindex + 1) % (1 << kNBitsBuffer);
        } else {
          istub++;
        }
      } else {
        //Check if last stub, if so, go to next buffer entry
        if (istub + 1 >= nstubs) {
          istub = 0;
          readindex = (readindex + 1) % (1 << kNBitsBuffer);
        } else {
          istub++;
        }
      }

      //Read vmstub memory and extract data fields
      const VMStubME& vmstub = vmstubs_->getVMStubMEBin(rzbin, istubtmp);

      int stubfinerz = vmstub.finerz().value();

      bool isPSmodule;

      if (barrel_) {
        isPSmodule = layerdisk_ < N_PSLAYER;
      } else {
        if (layerdisk_ < N_LAYER + 2) {
          isPSmodule = ((rzbin & 7) < 3) || ((rzbin & 7) == 3 && stubfinerz <= 3);
        } else {
          isPSmodule = ((rzbin & 7) < 3) || ((rzbin & 7) == 3 && stubfinerz <= 2);
        }
      }

      assert(isPSmodule == vmstub.isPSmodule());

      int nbits = isPSmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

      int deltaphi = projfinephi - vmstub.finephi().value();

      constexpr int mindeltaphicut = 3;
      constexpr int maxdeltaphicut = 5;
      bool passphi = (std::abs(deltaphi) < mindeltaphicut) || (std::abs(deltaphi) > maxdeltaphicut);

      unsigned int index = (projrinv << nbits) + vmstub.bend().value();
      if (!barrel_ && isPSmodule) {
        index += (1 << (nrinv_ + N_BENDBITS_2S));
      }

      //Check if stub z position consistent
      int idrz = stubfinerz - projfinerzadj;
      bool passz;

      if (barrel_) {
        if (isPSseed) {
          constexpr int drzcut = 1;
          passz = std::abs(idrz) <= drzcut;
        } else {
          constexpr int drzcut = 5;
          passz = std::abs(idrz) <= drzcut;
        }
      } else {
        if (isPSmodule) {
          constexpr int drzcut = 1;
          passz = std::abs(idrz) <= drzcut;
        } else {
          constexpr int drzcut = 3;
          passz = std::abs(idrz) <= drzcut;
        }
      }

      if (print) {
        edm::LogVerbatim("Tracklet") << "istep index : " << istep << " " << index << " " << vmstub.bend().value()
                                     << " rzbin istubtmp : " << rzbin << " " << istubtmp << " dz " << stubfinerz << " "
                                     << projfinerzadj << "  dphi: " << deltaphi;
      }

      //Check if stub bend and proj rinv consistent
      if (passz && passphi) {
        if (luttable_.lookup(index)) {
          Tracklet* proj = vmprojs_->getTracklet(projindex);
          std::pair<Tracklet*, int> tmp(proj, vmprojs_->getAllProjIndex(projindex));
          if (settings_.writeMonitorData("Seeds")) {
            ofstream fout("seeds.txt", ofstream::app);
            fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << proj->getISeed() << endl;
            fout.close();
          }
          candmatches_->addMatch(tmp, vmstub.stub());
          countpass++;
        }
      }
    }
  }

  if (settings_.writeMonitorData("ME")) {
    globals_->ofstream("matchengine.txt") << getName() << " " << countall << " " << countpass << endl;
  }
}
