#include "L1Trigger/TrackFindingTracklet/interface/TripletEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

using namespace std;
using namespace trklet;

TripletEngine::TripletEngine(string name, Settings const &settings, Globals *global)
    : ProcessBase(name, settings, global) {
  stubpairs_.clear();
  thirdvmstubs_.clear();
  layer1_ = 0;
  layer2_ = 0;
  layer3_ = 0;
  disk1_ = 0;
  disk2_ = 0;
  disk3_ = 0;
  dct1_ = 0;
  dct2_ = 0;
  dct3_ = 0;
  phi1_ = 0;
  phi2_ = 0;
  phi3_ = 0;
  z1_ = 0;
  z2_ = 0;
  z3_ = 0;
  r1_ = 0;
  r2_ = 0;
  r3_ = 0;

  if (name_[4] == 'L')
    layer1_ = name_[5] - '0';
  if (name_[4] == 'D')
    disk1_ = name_[5] - '0';
  if (name_[7] == 'L')
    layer2_ = name_[8] - '0';
  if (name_[7] == 'D')
    disk2_ = name_[8] - '0';

  if (layer1_ == 3 && layer2_ == 4) {
    layer3_ = 2;
    iSeed_ = 8;
  } else if (layer1_ == 5 && layer2_ == 6) {
    layer3_ = 4;
    iSeed_ = 9;
  } else if (layer1_ == 2 && layer2_ == 3) {
    disk3_ = 1;
    iSeed_ = 10;
  } else if (disk1_ == 1 && disk2_ == 2) {
    layer3_ = 2;
    iSeed_ = 11;
  } else
    throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";

  if ((layer2_ == 4 && layer3_ == 2) || (layer2_ == 6 && layer3_ == 4)) {
    secondphibits_ = settings_.nfinephi(1, iSeed_);
    thirdphibits_ = settings_.nfinephi(2, iSeed_);
  }
  if ((layer2_ == 3 && disk3_ == 1) || (disk2_ == 2 && layer3_ == 2)) {
    secondphibits_ = settings_.nfinephi(1, iSeed_);
    thirdphibits_ = settings_.nfinephi(2, iSeed_);
  }
  if (settings_.enableTripletTables() && !settings_.writeTripletTables())
    readTables();
}

TripletEngine::~TripletEngine() {
  if (settings_.writeTripletTables())
    writeTables();
}

void TripletEngine::addOutput(MemoryBase *memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "stubtripout") {
    auto *tmp = dynamic_cast<StubTripletsMemory *>(memory);
    assert(tmp != nullptr);
    stubtriplets_ = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TripletEngine::addInput(MemoryBase *memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "thirdvmstubin") {
    auto *tmp = dynamic_cast<VMStubsTEMemory *>(memory);
    assert(tmp != nullptr);
    thirdvmstubs_.push_back(tmp);
    return;
  }
  if (input.substr(0, 8) == "stubpair") {
    auto *tmp = dynamic_cast<StubPairsMemory *>(memory);
    assert(tmp != nullptr);
    stubpairs_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TripletEngine::execute() {
  unsigned int countall = 0;
  unsigned int countpass = 0;
  unsigned int nThirdStubs = 0;
  count_ = 0;

  for (unsigned int iThirdMem = 0; iThirdMem < thirdvmstubs_.size();
       nThirdStubs += thirdvmstubs_.at(iThirdMem)->nVMStubs(), iThirdMem++)
    ;

  assert(!thirdvmstubs_.empty());
  assert(!stubpairs_.empty());

  bool print = false && (getName().substr(0, 10) == "TRE_L2cL3c");

  print = print && nThirdStubs > 0;

  int hacksum = 0;
  if (print) {
    edm::LogVerbatim("Tracklet") << "In TripletEngine::execute : " << getName() << " " << nThirdStubs << ":";
    for (unsigned int i = 0; i < thirdvmstubs_.size(); ++i) {
      edm::LogVerbatim("Tracklet") << thirdvmstubs_.at(i)->getName() << " " << thirdvmstubs_.at(i)->nVMStubs();
    }
    int s = 0;
    std::string oss = "";
    for (unsigned int i = 0; i < stubpairs_.size(); ++i) {
      oss += std::to_string(stubpairs_.at(i)->nStubPairs());
      oss += " ";
      s += stubpairs_.at(i)->nStubPairs();
    }
    hacksum += nThirdStubs * s;
    edm::LogVerbatim("Tracklet") << oss;
    for (unsigned int i = 0; i < stubpairs_.size(); ++i) {
      edm::LogVerbatim("Tracklet") << "                                          " << stubpairs_.at(i)->getName();
    }
  }

  tmpSPTable_.clear();

  for (unsigned int i = 0; i < stubpairs_.size(); ++i) {
    for (unsigned int j = 0; j < stubpairs_.at(i)->nStubPairs(); ++j) {
      if (print)
        edm::LogVerbatim("Tracklet") << "     *****    " << stubpairs_.at(i)->getName() << " "
                                     << stubpairs_.at(i)->nStubPairs();

      auto firstvmstub = stubpairs_.at(i)->getVMStub1(j);
      auto secondvmstub = stubpairs_.at(i)->getVMStub2(j);

      if ((layer2_ == 4 && layer3_ == 2) || (layer2_ == 6 && layer3_ == 4)) {
        constexpr unsigned int vmbitshift = 10;
        int lookupbits = (int)((firstvmstub.vmbits().value() >> vmbitshift) & 1023);  //1023=2^vmbitshift-1
        int newbin = (lookupbits & 127);
        int bin = newbin / 8;

        int start = (bin >> 1);
        int last = start + (bin & 1);

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int k = 0; k < thirdvmstubs_.size(); k++) {
            string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
            vmsteSuffix = vmsteSuffix.substr(0, vmsteSuffix.find_last_of('n'));
            if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
              continue;
            for (unsigned int l = 0; l < thirdvmstubs_.at(k)->nVMStubsBinned(ibin); l++) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << "In " << getName() << " have third stub";
              }

              if (countall >= settings_.maxStep("TRE"))
                break;
              countall++;

              const VMStubTE &thirdvmstub = thirdvmstubs_.at(k)->getVMStubTEBinned(ibin, l);

              assert(secondphibits_ != -1);
              assert(thirdphibits_ != -1);

              unsigned int nvmsecond = settings_.nallstubs(layer2_ - 1) * settings_.nvmte(1, iSeed_);
              unsigned int nvmbitssecond = nbits(nvmsecond);

              FPGAWord iphisecondbin = secondvmstub.stub()->iphivmFineBins(nvmbitssecond, secondphibits_);

              //currently not using same number of bits as in the TED
              //assert(iphisecondbin==(int)secondvmstub.finephi());
              FPGAWord iphithirdbin = thirdvmstub.finephi();

              unsigned int index = (iphisecondbin.value() << thirdphibits_) + iphithirdbin.value();

              FPGAWord secondbend = secondvmstub.bend();
              FPGAWord thirdbend = thirdvmstub.bend();

              index = (index << secondbend.nbits()) + secondbend.value();
              index = (index << thirdbend.nbits()) + thirdbend.value();

              if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                  (index >= table_.size() || !table_[index])) {
                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet")
                      << "Stub pair rejected because of stub pt cut bends : "
                      << settings_.benddecode(secondvmstub.bend().value(), layer2_ - 1, secondvmstub.isPSmodule())
                      << " " << settings_.benddecode(thirdvmstub.bend().value(), layer3_ - 1, thirdvmstub.isPSmodule());
                }

                //FIXME temporarily commented out until bend table fixed
                //if (!settings_.writeTripletTables())
                //  continue;
              }
              if (settings_.writeTripletTables()) {
                if (index >= table_.size())
                  table_.resize(index + 1, false);
                table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize(spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back(stubpairs_.at(i)->getName());
              }

              if (settings_.debugTracklet())
                edm::LogVerbatim("Tracklet") << "Adding layer-layer pair in " << getName();
              if (settings_.writeMonitorData("Seeds")) {
                ofstream fout("seeds.txt", ofstream::app);
                fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                fout.close();
              }
              stubtriplets_->addStubs(thirdvmstub.stub(),
                                      (stubpairs_.at(i))->getVMStub1(j).stub(),
                                      (stubpairs_.at(i))->getVMStub2(j).stub());

              countpass++;
            }
          }
        }

      }

      else if (disk2_ == 2 && layer3_ == 2) {
        int lookupbits = (int)((firstvmstub.vmbits().value() >> 10) & 1023);
        int newbin = (lookupbits & 127);
        int bin = newbin / 8;

        int start = (bin >> 1);
        int last = start + (bin & 1);

        if (firstvmstub.stub()->disk().value() < 0) {  //TODO - negative disk should come from memory
          start = settings_.NLONGVMBINS() - last - 1;
          last = settings_.NLONGVMBINS() - start - 1;
        }

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int k = 0; k < thirdvmstubs_.size(); k++) {
            string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
            vmsteSuffix = vmsteSuffix.substr(0, vmsteSuffix.find_last_of('n'));
            if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
              continue;

            for (unsigned int l = 0; l < thirdvmstubs_.at(k)->nVMStubsBinned(ibin); l++) {
              if (countall >= settings_.maxStep("TRE"))
                break;
              countall++;

              const VMStubTE &thirdvmstub = thirdvmstubs_.at(k)->getVMStubTEBinned(ibin, l);

              assert(secondphibits_ != -1);
              assert(thirdphibits_ != -1);

              FPGAWord iphisecondbin = secondvmstub.finephi();
              FPGAWord iphithirdbin = thirdvmstub.finephi();

              unsigned int index = (iphisecondbin.value() << thirdphibits_) + iphithirdbin.value();

              FPGAWord secondbend = secondvmstub.bend();
              FPGAWord thirdbend = thirdvmstub.bend();

              index = (index << secondbend.nbits()) + secondbend.value();
              index = (index << thirdbend.nbits()) + thirdbend.value();

              if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                  (index >= table_.size() || !table_[index])) {
                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet")
                      << "Stub triplet rejected because of stub pt cut bends : "
                      << settings_.benddecode(secondvmstub.bend().value(), disk2_ + 5, secondvmstub.isPSmodule()) << " "
                      << settings_.benddecode(thirdvmstub.bend().value(), layer3_ - 1, thirdvmstub.isPSmodule());
                }
                continue;
              }
              if (settings_.writeTripletTables()) {
                if (index >= table_.size())
                  table_.resize(index + 1, false);
                table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize(spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back(stubpairs_.at(i)->getName());
              }

              if (settings_.debugTracklet())
                edm::LogVerbatim("Tracklet") << "Adding layer-disk pair in " << getName();
              if (settings_.writeMonitorData("Seeds")) {
                ofstream fout("seeds.txt", ofstream::app);
                fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                fout.close();
              }
              stubtriplets_->addStubs(thirdvmstub.stub(),
                                      (stubpairs_.at(i))->getVMStub1(j).stub(),
                                      (stubpairs_.at(i))->getVMStub2(j).stub());
              countpass++;
            }
          }
        }
      }

      else if (layer2_ == 3 && disk3_ == 1) {
        int lookupbits = (int)((firstvmstub.vmbits().value() >> 10) & 1023);

        int newbin = (lookupbits & 127);
        int bin = newbin / 8;

        int start = (bin >> 1);
        int last = start + (bin & 1);

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int k = 0; k < thirdvmstubs_.size(); k++) {
            string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
            vmsteSuffix = vmsteSuffix.substr(0, vmsteSuffix.find_last_of('n'));
            if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
              continue;
            for (unsigned int l = 0; l < thirdvmstubs_.at(k)->nVMStubsBinned(ibin); l++) {
              if (countall >= settings_.maxStep("TRE"))
                break;
              countall++;

              const VMStubTE &thirdvmstub = thirdvmstubs_.at(k)->getVMStubTEBinned(ibin, l);

              assert(secondphibits_ != -1);
              assert(thirdphibits_ != -1);

              unsigned int nvmsecond;

              nvmsecond = settings_.nallstubs(layer2_ - 1) * settings_.nvmte(1, iSeed_);
              unsigned int nvmbitssecond = nbits(nvmsecond);

              FPGAWord iphisecondbin = secondvmstub.stub()->iphivmFineBins(nvmbitssecond, secondphibits_);

              //currentlty not using same number of bits as in the TED
              //assert(iphisecondbin==(int)secondvmstub.finephi());
              FPGAWord iphithirdbin = thirdvmstub.finephi();

              unsigned int index = (iphisecondbin.value() << thirdphibits_) + iphithirdbin.value();

              FPGAWord secondbend = secondvmstub.bend();
              FPGAWord thirdbend = thirdvmstub.bend();

              index = (index << secondbend.nbits()) + secondbend.value();
              index = (index << thirdbend.nbits()) + thirdbend.value();

              if ((settings_.enableTripletTables() && !settings_.writeTripletTables()) &&
                  (index >= table_.size() || !table_[index])) {
                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet")
                      << "Stub pair rejected because of stub pt cut bends : "
                      << settings_.benddecode(secondvmstub.bend().value(), layer2_ - 1, secondvmstub.isPSmodule())
                      << " " << settings_.benddecode(thirdvmstub.bend().value(), disk3_ + 5, thirdvmstub.isPSmodule());
                }
                continue;
              }
              if (settings_.writeTripletTables()) {
                if (index >= table_.size())
                  table_.resize(index + 1, false);
                table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize(spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back(stubpairs_.at(i)->getName());
              }

              if (settings_.debugTracklet())
                edm::LogVerbatim("Tracklet") << "Adding layer-disk pair in " << getName();
              if (settings_.writeMonitorData("Seeds")) {
                ofstream fout("seeds.txt", ofstream::app);
                fout << __FILE__ << ":" << __LINE__ << " " << name_ << " " << iSeed_ << endl;
                fout.close();
              }
              stubtriplets_->addStubs(thirdvmstub.stub(),
                                      (stubpairs_.at(i))->getVMStub1(j).stub(),
                                      (stubpairs_.at(i))->getVMStub2(j).stub());
              countpass++;
            }
          }
        }
      }
    }
  }

  for (const auto &tedName : tmpSPTable_) {
    for (unsigned spIndex = 0; spIndex < tedName.second.size(); spIndex++) {
      if (tedName.second.at(spIndex).empty())
        continue;
      vector<string> entry(tedName.second.at(spIndex));
      sort(entry.begin(), entry.end());
      entry.erase(unique(entry.begin(), entry.end()), entry.end());
      const string &spName = entry.at(0);

      if (!spTable_.count(tedName.first))
        spTable_[tedName.first];
      if (spIndex >= spTable_.at(tedName.first).size())
        spTable_.at(tedName.first).resize(spIndex + 1);
      if (!spTable_.at(tedName.first).at(spIndex).count(spName))
        spTable_.at(tedName.first).at(spIndex)[spName] = 0;
      spTable_.at(tedName.first).at(spIndex)[spName]++;
    }
  }

  if (settings_.writeMonitorData("TRE")) {
    globals_->ofstream("tripletengine.txt") << getName() << " " << countall << " " << countpass << endl;
  }
}

void TripletEngine::readTables() {
  ifstream fin;
  string tableName, word;
  unsigned num;

  string tablePath = settings_.tableTREFile();
  unsigned int finddir = tablePath.find("table_TRE_");
  tableName = tablePath.substr(0, finddir) + "table_" + name_ + ".txt";

  fin.open(tableName, ifstream::in);
  if (!fin) {
    throw cms::Exception("BadConfig") << "TripletEngine::readTables, file " << tableName << " not known";
  }
  while (!fin.eof()) {
    fin >> word;
    num = atoi(word.c_str());
    table_.push_back(num > 0 ? true : false);
  }
  fin.close();
}

void TripletEngine::writeTables() {
  ofstream fout;
  stringstream tableName;

  tableName << "table/table_" << name_ << ".txt";

  fout.open(tableName.str(), ofstream::out);
  for (const auto entry : table_)
    fout << entry << endl;
  fout.close();

  for (const auto &tedName : spTable_) {
    tableName.str("");
    tableName << "table/table_" << tedName.first << "_" << name_ << ".txt";

    fout.open(tableName.str(), ofstream::out);
    for (const auto &entry : tedName.second) {
      for (const auto &spName : entry)
        fout << spName.first << ":" << spName.second << " ";
      fout << endl;
    }
    fout.close();
  }
}
