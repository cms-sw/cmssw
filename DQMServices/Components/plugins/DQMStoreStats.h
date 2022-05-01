#ifndef DQMStoreStats_H
#define DQMStoreStats_H

/** \class DQMStoreStats
 * *
 *  DQM Test Client
 *
 *  \author Andreas Meyer CERN
 *  \author Jan Olzem DESY
 *   
 */

#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <utility>
#include <fstream>
#include <sstream>

#include "TFile.h"
#include "TTree.h"

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

//
// class declarations
//

using dqm::legacy::DQMStore;
using dqm::legacy::MonitorElement;

///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsSubfolder {
public:
  DQMStoreStatsSubfolder() {
    totalHistos_ = 0;
    totalBins_ = 0;
    totalMemory_ = 0;
    totalEmptyBins_ = 0;
  }
  std::string subfolderName_;
  unsigned int totalHistos_;
  unsigned int totalBins_;
  unsigned int totalEmptyBins_;
  unsigned int totalMemory_;
  void AddBinsF(unsigned int nBins, unsigned int nEmptyBins) {
    ++totalHistos_;
    totalBins_ += nBins;
    totalEmptyBins_ += nEmptyBins;
    totalMemory_ += (nBins *= sizeof(float));
  }
  void AddBinsS(unsigned int nBins, unsigned int nEmptyBins) {
    ++totalHistos_;
    totalBins_ += nBins;
    totalEmptyBins_ += nEmptyBins;
    totalMemory_ += (nBins *= sizeof(short));
  }
  void AddBinsD(unsigned int nBins, unsigned int nEmptyBins) {
    ++totalHistos_;
    totalBins_ += nBins;
    totalEmptyBins_ += nEmptyBins;
    totalMemory_ += (nBins *= sizeof(double));
  }
  void AddBinsI(unsigned int nBins, unsigned int nEmptyBins) {
    ++totalHistos_;
    totalBins_ += nBins;
    totalEmptyBins_ += nEmptyBins;
    totalMemory_ += (nBins *= sizeof(int));
  }
};

///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsSubsystem : public std::vector<DQMStoreStatsSubfolder> {
public:
  DQMStoreStatsSubsystem() = default;
  std::string subsystemName_;
};

///
/// DQMStoreStats helper class for
/// storing subsystem results
///
class DQMStoreStatsTopLevel : public std::vector<DQMStoreStatsSubsystem> {
public:
  DQMStoreStatsTopLevel() = default;
};

template <class Item>
class Iterator {
public:
  virtual ~Iterator() = default;
  virtual void First() = 0;
  virtual void Next() = 0;
  virtual bool IsDone() const = 0;
  virtual Item CurrentItem() const = 0;

protected:
  Iterator() = default;
};

template <class Item>
class VIterator : public Iterator<Item> {
public:
  VIterator(const std::vector<Item>* aVector) : vector_(aVector), index(0) {}
  ~VIterator() override = default;
  void First() override { index = 0; }
  void Next() override { ++index; }
  virtual int size() { return vector_->size(); }
  virtual int getIndex() { return (int)index; }

  bool IsDone() const override {
    if (index < (unsigned int)vector_->size())
      return false;
    return true;
  }

  Item CurrentItem() const override { return vector_->operator[](index); }

private:
  const std::vector<Item>* vector_;
  unsigned int index;
};

static unsigned int getId() {
  static unsigned int id = 10;
  return ++id;
}

class Folder {
public:
  Folder(std::string name)
      : totalHistos_(0),
        totalBins_(0),
        totalEmptyBins_(0),
        totalMemory_(0),
        id_(10),
        level_(0),
        folderName_(std::move(name)),
        father_(nullptr) {}

  ~Folder() {
    for (auto& subfolder : subfolders_)
      delete subfolder;
  }

  void setFather(Folder* e) { father_ = e; }
  Folder* getFather() { return father_; }
  const std::string& name() { return folderName_; }

  Folder* cd(const std::string& name) {
    for (auto& subfolder : subfolders_)
      if (subfolder->name() == name)
        return subfolder;
    auto* tmp = new Folder(name);
    this->add(tmp);
    return tmp;
  }

  void setId(unsigned int id) { id_ = id; }
  unsigned int id() { return id_; }
  void setLevel(unsigned int value) { level_ = value; }
  unsigned int level() { return level_; }

  void add(Folder* f) {
    f->setFather(this);
    subfolders_.push_back(f);
    f->setLevel(level_ + 1);
    f->setId(getId());
  }

  unsigned int getHistos() {
    unsigned int result = totalHistos_;
    for (auto& subfolder : subfolders_)
      result += subfolder->getHistos();
    return result;
  }
  unsigned int getBins() {
    unsigned int result = totalBins_;
    for (auto& subfolder : subfolders_)
      result += subfolder->getBins();
    return result;
  }
  unsigned int getEmptyBins() {
    unsigned int result = totalEmptyBins_;
    for (auto& subfolder : subfolders_)
      result += subfolder->getEmptyBins();
    return result;
  }
  unsigned int getMemory() {
    unsigned int result = totalMemory_;
    for (auto& subfolder : subfolders_)
      result += subfolder->getMemory();
    return result;
  }
  void update(unsigned int bins, unsigned int empty, unsigned int memory) {
    totalHistos_ += 1;
    totalBins_ += bins;
    totalEmptyBins_ += empty;
    totalMemory_ += memory;
  }
  void dump(std::string indent) {
    indent.append(" ");
    std::cout << indent << "I'm a " << name() << " whose father is " << getFather() << " with ID: " << id_
              << " Histo: " << getHistos() << " Bins: " << getBins() << " EmptyBins: " << getEmptyBins()
              << " Memory: " << getMemory() << " and my children are: " << std::endl;
    for (auto& subfolder : subfolders_)
      subfolder->dump(indent);
  }
  VIterator<Folder*> CreateIterator() { return VIterator<Folder*>(&subfolders_); }

  void mainrows(std::string& sql_statement) {
    std::stringstream s("");
    s << "INSERT INTO mainrows(id, symbol_id, self_count, cumulative_count, kids, self_calls, total_calls, self_paths, "
         "total_paths, pct)"
         " VALUES("
      << id_ << ", " << id_ << ", " << getMemory() << ", " << getMemory() << ", " << subfolders_.size() << ", "
      << getBins() - getEmptyBins() << ", " << getBins() << ", " << getHistos() << ", " << getHistos() << ", 0.0);\n";
    sql_statement.append(s.str());
    for (auto& subfolder : subfolders_)
      subfolder->mainrows(sql_statement);
  }

  void symbols(std::string& sql_statement) {
    unsigned int parentid = this->getFather() ? this->getFather()->id() : id_;
    std::stringstream s("");
    s << "INSERT INTO symbols(id, name, filename_id) VALUES (" << id_ << ",\"" << folderName_ << "\", " << parentid
      << ");\n";
    sql_statement.append(s.str());
    for (auto& subfolder : subfolders_)
      subfolder->symbols(sql_statement);
  }

  void parents(std::string& sql_statement) {
    unsigned int parentid = this->getFather() ? this->getFather()->id() : id_;
    std::stringstream s("");
    s << "INSERT INTO parents(self_id, child_id, to_child_count, to_child_calls, to_child_paths, pct) VALUES("
      << parentid << "," << id_ << "," << totalMemory_ << "," << totalBins_ << "," << totalHistos_ << ",0"
      << ");\n";
    sql_statement.append(s.str());
    for (auto& subfolder : subfolders_)
      subfolder->parents(sql_statement);
  }

  void children(std::string& sql_statement) {
    unsigned int parentid = this->getFather() ? this->getFather()->id() : id_;
    std::stringstream s("");
    s << "INSERT INTO children(self_id, parent_id, from_parent_count, from_parent_calls, from_parent_paths, pct) "
         "VALUES("
      << id_ << "," << parentid << "," << getMemory() << "," << getBins() - getEmptyBins() << "," << totalHistos_
      << ",0"
      << ");\n";
    sql_statement.append(s.str());
    for (auto& subfolder : subfolders_)
      subfolder->children(sql_statement);
  }

  void mainrows_cumulative(std::string& sql_statement) {
    std::stringstream s("");
    s << "INSERT INTO mainrows(id, symbol_id, self_count, cumulative_count, kids, self_calls, total_calls, self_paths, "
         "total_paths, pct)"
      << " VALUES(" << id_ << "," << id_ << "," << 0 << "," << getMemory() << ", 0," << getBins() - getEmptyBins()
      << "," << getBins() << ", 0, " << getHistos() << ", 0);\n";
    sql_statement.append(s.str());
  }

  void summary(std::string& sql_statement) {
    std::stringstream s("");
    s << "INSERT INTO summary(counter, total_count, total_freq, tick_period) VALUES (\"BINS_LIVE\"," << getMemory()
      << "," << getBins() << ", 1);\n";
    sql_statement.append(s.str());
  }

  void files(std::string& sql_statement) {
    std::stringstream s("");
    s << "INSERT INTO files(id, name) VALUES(" << id_ << ",\"" << folderName_ << "\");\n";
    sql_statement.append(s.str());
  }

private:
  unsigned int totalHistos_;
  unsigned int totalBins_;
  unsigned int totalEmptyBins_;
  unsigned int totalMemory_;
  unsigned int id_;
  unsigned int level_;
  std::string folderName_;
  Folder* father_;
  std::vector<Folder*> subfolders_;
};

///
/// DQMStoreStats itself
///
class DQMStoreStats : public edm::EDAnalyzer {
public:
  DQMStoreStats(const edm::ParameterSet&);
  ~DQMStoreStats() override;

  enum statsMode { considerAllME = 0, considerOnlyLumiProductME = 1 };

protected:
  // BeginJob
  void beginJob() override;

  // BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  // Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;

  // EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;

  // Endjob
  void endJob() override;

private:
  int calcstats(int);
  void calcIgProfDump(Folder&);
  void dumpMemoryProfile();
  std::pair<unsigned int, unsigned int> readMemoryEntry() const;
  void print();

  DQMStore* dbe_;
  edm::ParameterSet parameters_;

  std::string subsystem_;
  std::string subfolder_;
  int nbinsglobal_;
  int nbinssubsys_;
  int nmeglobal_;
  int nmesubsys_;
  int maxbinsglobal_;
  int maxbinssubsys_;
  std::string maxbinsmeglobal_;
  std::string maxbinsmesubsys_;

  int statsdepth_;
  std::string pathnamematch_;
  int verbose_;

  std::vector<std::pair<time_t, unsigned int> > memoryHistoryVector_;
  time_t startingTime_;
  bool isOpenProcFileSuccessful_;
  std::stringstream procFileName_;

  bool runonendrun_;
  bool runonendjob_;
  bool runonendlumi_;
  bool runineventloop_;
  bool dumpMemHistory_;
  bool dumpToFWJR_;

  // ---------- member data ----------
};

#endif
