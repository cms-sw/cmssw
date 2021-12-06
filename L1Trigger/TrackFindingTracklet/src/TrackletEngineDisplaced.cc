#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

TrackletEngineDisplaced::TrackletEngineDisplaced(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {
  stubpairs_.clear();
  firstvmstubs_.clear();
  secondvmstubs_ = nullptr;
  layer1_ = 0;
  layer2_ = 0;
  disk1_ = 0;
  disk2_ = 0;
  string name1 = name.substr(1);  //this is to correct for "TED" having one more letter then "TE"
  if (name1[3] == 'L') {
    layer1_ = name1[4] - '0';
  }
  if (name1[3] == 'D') {
    disk1_ = name1[4] - '0';
  }
  if (name1[11] == 'L') {
    layer2_ = name1[12] - '0';
  }
  if (name1[11] == 'D') {
    disk2_ = name1[12] - '0';
  }
  if (name1[12] == 'L') {
    layer2_ = name1[13] - '0';
  }
  if (name1[12] == 'D') {
    disk2_ = name1[13] - '0';
  }

  iSeed_ = -1;
  if (layer1_ == 3 && layer2_ == 4)
    iSeed_ = 8;
  if (layer1_ == 5 && layer2_ == 6)
    iSeed_ = 9;
  if (layer1_ == 2 && layer2_ == 3)
    iSeed_ = 10;
  if (disk1_ == 1 && disk2_ == 2)
    iSeed_ = 11;

  firstphibits_ = settings_.nfinephi(0, iSeed_);
  secondphibits_ = settings_.nfinephi(1, iSeed_);
}

TrackletEngineDisplaced::~TrackletEngineDisplaced() { table_.clear(); }

void TrackletEngineDisplaced::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "stubpairout") {
    StubPairsMemory* tmp = dynamic_cast<StubPairsMemory*>(memory);
    assert(tmp != nullptr);
    stubpairs_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TrackletEngineDisplaced::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "firstvmstubin") {
    VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    firstvmstubs_.push_back(tmp);
    return;
  }
  if (input == "secondvmstubin") {
    VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    secondvmstubs_ = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletEngineDisplaced::execute() {
  if (!settings_.useSeed(iSeed_))
    return;

  if (table_.empty() && (settings_.enableTripletTables() && !settings_.writeTripletTables()))
    readTables();

  unsigned int countall = 0;
  unsigned int countpass = 0;
  unsigned int nInnerStubs = 0;

  for (unsigned int iInnerMem = 0; iInnerMem < firstvmstubs_.size();
       nInnerStubs += firstvmstubs_.at(iInnerMem)->nVMStubs(), iInnerMem++)
    ;

  assert(!firstvmstubs_.empty());
  assert(secondvmstubs_ != nullptr);

  for (auto& iInnerMem : firstvmstubs_) {
    assert(iInnerMem->nVMStubs() == iInnerMem->nVMStubs());
    for (unsigned int i = 0; i < iInnerMem->nVMStubs(); i++) {
      const VMStubTE& firstvmstub = iInnerMem->getVMStubTE(i);
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "In " << getName() << " have first stub";
      }

      if ((layer1_ == 3 && layer2_ == 4) || (layer1_ == 5 && layer2_ == 6)) {
        int lookupbits = firstvmstub.vmbits().value() & 1023;
        int zdiffmax = (lookupbits >> 7);
        int newbin = (lookupbits & 127);
        int bin = newbin / 8;

        int zbinfirst = newbin & 7;

        int start = (bin >> 1);
        int last = start + (bin & 1);

        assert(last < 8);

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Will look in zbins " << start << " to " << last;
        }
        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int j = 0; j < secondvmstubs_->nVMStubsBinned(ibin); j++) {
            if (settings_.debugTracklet()) {
              edm::LogVerbatim("Tracklet") << "In " << getName() << " have second stub(1) " << ibin << " " << j;
            }

            if (countall >= settings_.maxStep("TE"))
              break;
            countall++;
            const VMStubTE& secondvmstub = secondvmstubs_->getVMStubTEBinned(ibin, j);

            int zbin = (secondvmstub.vmbits().value() & 7);
            if (start != ibin)
              zbin += 8;
            if (zbin < zbinfirst || zbin - zbinfirst > zdiffmax) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << "Stubpair rejected because of wrong zbin";
              }
              continue;
            }

            assert(firstphibits_ != -1);
            assert(secondphibits_ != -1);

            FPGAWord iphifirstbin = firstvmstub.finephi();
            FPGAWord iphisecondbin = secondvmstub.finephi();

            unsigned int index = (iphifirstbin.value() << secondphibits_) + iphisecondbin.value();

            FPGAWord firstbend = firstvmstub.bend();
            FPGAWord secondbend = secondvmstub.bend();

            index = (index << firstbend.nbits()) + firstbend.value();
            index = (index << secondbend.nbits()) + secondbend.value();

            if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                (index >= table_.size() || table_.at(index).empty())) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet")
                    << "Stub pair rejected because of stub pt cut bends : "
                    << settings_.benddecode(firstvmstub.bend().value(), layer1_ - 1, firstvmstub.isPSmodule()) << " "
                    << settings_.benddecode(secondvmstub.bend().value(), layer2_ - 1, secondvmstub.isPSmodule());
              }

              //FIXME temporarily commented out until stub bend table fixed
              //if (!settings_.writeTripletTables())
              //  continue;
            }

            if (settings_.debugTracklet())
              edm::LogVerbatim("Tracklet") << "Adding layer-layer pair in " << getName();
            for (unsigned int isp = 0; isp < stubpairs_.size(); ++isp) {
              if (!settings_.enableTripletTables() || settings_.writeTripletTables() || table_.at(index).count(isp)) {
                if (settings_.writeMonitorData("Seeds")) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubpairs_.at(isp)->addStubPair(firstvmstub, secondvmstub, index, getName());
              }
            }

            countpass++;
          }
        }

      } else if (layer1_ == 2 && layer2_ == 3) {
        int lookupbits = firstvmstub.vmbits().value() & 1023;
        int zdiffmax = (lookupbits >> 7);
        int newbin = (lookupbits & 127);
        int bin = newbin / 8;

        int zbinfirst = newbin & 7;

        int start = (bin >> 1);
        int last = start + (bin & 1);

        assert(last < 8);

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Will look in zbins " << start << " to " << last;
        }
        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int j = 0; j < secondvmstubs_->nVMStubsBinned(ibin); j++) {
            if (settings_.debugTracklet()) {
              edm::LogVerbatim("Tracklet") << "In " << getName() << " have second stub(2) ";
            }

            if (countall >= settings_.maxStep("TE"))
              break;
            countall++;

            const VMStubTE& secondvmstub = secondvmstubs_->getVMStubTEBinned(ibin, j);

            int zbin = (secondvmstub.vmbits().value() & 7);
            if (start != ibin)
              zbin += 8;
            if (zbin < zbinfirst || zbin - zbinfirst > zdiffmax) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << "Stubpair rejected because of wrong zbin";
              }
              continue;
            }

            assert(firstphibits_ != -1);
            assert(secondphibits_ != -1);

            FPGAWord iphifirstbin = firstvmstub.finephi();
            FPGAWord iphisecondbin = secondvmstub.finephi();

            unsigned int index = (iphifirstbin.value() << secondphibits_) + iphisecondbin.value();

            FPGAWord firstbend = firstvmstub.bend();
            FPGAWord secondbend = secondvmstub.bend();

            index = (index << firstbend.nbits()) + firstbend.value();
            index = (index << secondbend.nbits()) + secondbend.value();

            if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                (index >= table_.size() || table_.at(index).empty())) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet")
                    << "Stub pair rejected because of stub pt cut bends : "
                    << settings_.benddecode(firstvmstub.bend().value(), layer1_ - 1, firstvmstub.isPSmodule()) << " "
                    << settings_.benddecode(secondvmstub.bend().value(), layer2_ - 1, secondvmstub.isPSmodule());
              }
              continue;
            }

            if (settings_.debugTracklet())
              edm::LogVerbatim("Tracklet") << "Adding layer-layer pair in " << getName();
            for (unsigned int isp = 0; isp < stubpairs_.size(); ++isp) {
              if ((!settings_.enableTripletTables() || settings_.writeTripletTables()) ||
                  (index < table_.size() && table_.at(index).count(isp))) {
                if (settings_.writeMonitorData("Seeds")) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubpairs_.at(isp)->addStubPair(firstvmstub, secondvmstub, index, getName());
              }
            }

            countpass++;
          }
        }

      } else if (disk1_ == 1 && disk2_ == 2) {
        if (settings_.debugTracklet())
          edm::LogVerbatim("Tracklet") << getName() << " Disk-disk pair";

        int lookupbits = firstvmstub.vmbits().value() & 511;
        bool negdisk = firstvmstub.stub()->disk().value() < 0;
        int rdiffmax = (lookupbits >> 6);
        int newbin = (lookupbits & 63);
        int bin = newbin / 8;

        int rbinfirst = newbin & 7;

        int start = (bin >> 1);
        if (negdisk)
          start += 4;
        int last = start + (bin & 1);
        assert(last < 8);
        for (int ibin = start; ibin <= last; ibin++) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << getName() << " looking for matching stub in " << secondvmstubs_->getName()
                                         << " in bin = " << ibin << " with " << secondvmstubs_->nVMStubsBinned(ibin)
                                         << " stubs";
          }
          for (unsigned int j = 0; j < secondvmstubs_->nVMStubsBinned(ibin); j++) {
            if (countall >= settings_.maxStep("TE"))
              break;
            countall++;

            const VMStubTE& secondvmstub = secondvmstubs_->getVMStubTEBinned(ibin, j);

            int rbin = (secondvmstub.vmbits().value() & 7);
            if (start != ibin)
              rbin += 8;
            if (rbin < rbinfirst)
              continue;
            if (rbin - rbinfirst > rdiffmax)
              continue;

            unsigned int irsecondbin = secondvmstub.vmbits().value() >> 2;

            FPGAWord iphifirstbin = firstvmstub.finephi();
            FPGAWord iphisecondbin = secondvmstub.finephi();

            unsigned int index = (irsecondbin << (secondphibits_ + firstphibits_)) +
                                 (iphifirstbin.value() << secondphibits_) + iphisecondbin.value();

            FPGAWord firstbend = firstvmstub.bend();
            FPGAWord secondbend = secondvmstub.bend();

            index = (index << firstbend.nbits()) + firstbend.value();
            index = (index << secondbend.nbits()) + secondbend.value();

            if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                (index >= table_.size() || table_.at(index).empty())) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet")
                    << "Stub pair rejected because of stub pt cut bends : "
                    << settings_.benddecode(firstvmstub.bend().value(), disk1_ + 5, firstvmstub.isPSmodule()) << " "
                    << settings_.benddecode(secondvmstub.bend().value(), disk2_ + 5, secondvmstub.isPSmodule());
              }
              continue;
            }

            if (settings_.debugTracklet())
              edm::LogVerbatim("Tracklet") << "Adding disk-disk pair in " << getName();

            for (unsigned int isp = 0; isp < stubpairs_.size(); ++isp) {
              if ((!settings_.enableTripletTables() || settings_.writeTripletTables()) ||
                  (index < table_.size() && table_.at(index).count(isp))) {
                if (settings_.writeMonitorData("Seeds")) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubpairs_.at(isp)->addStubPair(firstvmstub, secondvmstub, index, getName());
              }
            }
            countpass++;
          }
        }
      }
    }
  }
  if (countall > 5000) {
    edm::LogVerbatim("Tracklet") << "In TrackletEngineDisplaced::execute : " << getName() << " " << nInnerStubs << " "
                                 << secondvmstubs_->nVMStubs() << " " << countall << " " << countpass;
    for (auto& iInnerMem : firstvmstubs_) {
      for (unsigned int i = 0; i < iInnerMem->nVMStubs(); i++) {
        const VMStubTE& firstvmstub = iInnerMem->getVMStubTE(i);
        edm::LogVerbatim("Tracklet") << "In TrackletEngineDisplaced::execute first stub : "
                                     << firstvmstub.stub()->l1tstub()->r() << " "
                                     << firstvmstub.stub()->l1tstub()->phi() << " "
                                     << firstvmstub.stub()->l1tstub()->r() * firstvmstub.stub()->l1tstub()->phi() << " "
                                     << firstvmstub.stub()->l1tstub()->z();
      }
    }
    for (unsigned int i = 0; i < secondvmstubs_->nVMStubs(); i++) {
      const VMStubTE& secondvmstub = secondvmstubs_->getVMStubTE(i);
      edm::LogVerbatim("Tracklet") << "In TrackletEngineDisplaced::execute second stub : "
                                   << secondvmstub.stub()->l1tstub()->r() << " "
                                   << secondvmstub.stub()->l1tstub()->phi() << " "
                                   << secondvmstub.stub()->l1tstub()->r() * secondvmstub.stub()->l1tstub()->phi() << " "
                                   << secondvmstub.stub()->l1tstub()->z();
    }
  }

  if (settings_.writeMonitorData("TED")) {
    globals_->ofstream("trackletenginedisplaces.txt") << getName() << " " << countall << " " << countpass << endl;
  }
}

void TrackletEngineDisplaced::readTables() {
  ifstream fin;
  string tableName, line, word;

  string tablePath = settings_.tableTEDFile();
  unsigned int finddir = tablePath.find("table_TED_");
  tableName = tablePath.substr(0, finddir) + "table_" + name_ + ".txt";

  fin.open(tableName, ifstream::in);
  if (!fin) {
    throw cms::Exception("BadConfig") << "TripletEngine::readTables, file " << tableName << " not known";
  }

  while (getline(fin, line)) {
    istringstream iss(line);
    table_.resize(table_.size() + 1);

    while (iss >> word)
      table_[table_.size() - 1].insert(memNameToIndex(word));
  }
  fin.close();
}

short TrackletEngineDisplaced::memNameToIndex(const string& name) {
  for (unsigned int isp = 0; isp < stubpairs_.size(); ++isp)
    if (stubpairs_.at(isp)->getName() == name)
      return isp;
  return -1;
}
