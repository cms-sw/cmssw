#ifndef DataFormats_Histograms_MonitorElementCollection_h
#define DataFormats_Histograms_MonitorElementCollection_h
// -*- C++ -*-
//
// Package:     DataFormats/Histograms
// Class  :     MonitorElementCollection
//
/**\class MonitorElementCollection MonitorElementCollection.h "DataFormats/Histograms/interface/MonitorElementCollection.h"

 Description: Product to represent DQM data in LuminosityBlocks and Runs.
 The MonitorElements are represented by a simple struct that only contains the 
 required fields to represent a ME. The only opration allowed on these objects
 is merging, which is a important part of the DQM functionality and should be
 handled by EDM.

 Usage: This product should only be handled by the DQMStore, which provides 
 access to the MEs inside. The DQMStore will wrap the MonitorElementData in
 real MonitorElements, which allow various operations on the underlying 
 histograms, depending on the current stage of processing: In the RECO step,
 only filling is allowed, while in HARVESTING, the same data will be wrapped in
 a MonitorElement that also allows access to the ROOT objects.

 Currently, the product types are not used as products and all data is passed
 through the edm::Service<DQMStore>.

*/
//
// Original Author:  Marcel Schneider
//         Created:  2018-05-02
//
//
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <regex>

#include "TH1.h"

struct MonitorElementData {
  // This is technically a union, but the struct is safer.
  struct Scalar {
    int64_t num = 0;
    double real = 0;
    std::string str;
  };

  // Quality test result types.
  // These are inherited from DQMNet/old DQMStore, and left unchanged to avoid
  // another layer of wrapping. The APIs are used in some places in subsystem
  // code, and could be changed, but not removed.
  class QReport {
  public:
    struct QValue {
      int code;
      float qtresult;
      std::string message;
      std::string qtname;
      std::string algorithm;
    };
    struct DQMChannel {
      int binx;       //< bin # in x-axis (or bin # for 1D histogram)
      int biny;       //< bin # in y-axis (for 2D or 3D histograms)
      int binz;       //< bin # in z-axis (for 3D histograms)
      float content;  //< bin content
      float RMS;      //< RMS of bin content

      int getBin() { return getBinX(); }
      int getBinX() { return binx; }
      int getBinY() { return biny; }
      int getBinZ() { return binz; }
      float getContents() { return content; }
      float getRMS() { return RMS; }

      DQMChannel(int bx, int by, int bz, float data, float rms) {
        binx = bx;
        biny = by;
        binz = bz;
        content = data;
        RMS = rms;
      }

      DQMChannel() {
        binx = 0;
        biny = 0;
        binz = 0;
        content = 0;
        RMS = 0;
      }
    };

    /// access underlying value
    QValue& getValue() { return qvalue_; };
    QValue const& getValue() const { return qvalue_; };

    /// get test status
    int getStatus() const { return qvalue_.code; }

    /// get test result i.e. prob value
    float getQTresult() const { return qvalue_.qtresult; }

    /// get message attached to test
    const std::string& getMessage() const { return qvalue_.message; }

    /// get name of quality test
    const std::string& getQRName() const { return qvalue_.qtname; }

    /// get quality test algorithm
    const std::string& getAlgorithm() const { return qvalue_.algorithm; }

    /// get vector of channels that failed test
    /// (not relevant for all quality tests!)
    const std::vector<DQMChannel>& getBadChannels() const { return badChannels_; }

    void setBadChannels(std::vector<DQMChannel> badChannels) { badChannels_ = badChannels; }

    QReport(QValue value) : qvalue_(value) {}

  private:
    QValue qvalue_;                        //< Pointer to the actual data.
    std::vector<DQMChannel> badChannels_;  //< Bad channels from QCriterion.
  };

  // These values are compatible to DQMNet, but DQMNet is not likely to exist
  // in the future.
  enum class Kind {
    INVALID = 0x0,
    INT = 0x1,
    REAL = 0x2,
    STRING = 0x3,
    TH1F = 0x10,
    TH1S = 0x11,
    TH1D = 0x12,
    TH2F = 0x20,
    TH2S = 0x21,
    TH2D = 0x22,
    TH3F = 0x30,
    TPROFILE = 0x40,
    TPROFILE2D = 0x41
  };

  // Which window of time the ME is supposed to cover.
  // There is space for a granularity level between runs and lumisections,
  // maybe blocks of 10LS or some fixed number of events or integrated
  // luminosity. We also want to be able to change the granularity centrally
  // depending on the use case. That is what the default is for, and it should
  // be used unless some specific granularity is really required.
  // We'll also need to switch the default to JOB for multi-run harvesting.
  enum Scope { JOB = 1, RUN = 2, LUMI = 3 /*, BLOCK = 4 */ };

  // The main ME data. We don't keep references/QTest results, instead we use
  // only the fields stored in DQMIO files.
  struct Value {
    Scalar scalar_;
    edm::propagate_const<std::unique_ptr<TH1>> object_;
    std::vector<QReport> qreports_;
  };

  struct Path {
  private:
    // We could use pointers to interned strings here to save some space.
    std::string dirname_;
    std::string objname_;

  public:
    enum class Type { DIR, DIR_AND_NAME };

    std::string const& getDirname() const { return dirname_; }
    std::string const& getObjectname() const { return objname_; }
    std::string getFullname() const { return dirname_ + objname_; }

    // Clean up the path and normalize it to preserve certain invariants.
    // Instead of reasoning about whatever properties of paths, we just parse
    // the thing and build a normalized instance with no slash in the beginning
    // and a slash in the end.
    // Type of string `path` could be just directory name, or
    // directory name followed by the name of the monitor element
    void set(std::string path, Path::Type type) {
      std::string in(path);
      std::vector<std::string> buf;
      static std::regex const dir("^/*([^/]+)");
      std::smatch m;

      while (std::regex_search(in, m, dir)) {
        if (m[1] == "..") {
          if (!buf.empty()) {
            buf.pop_back();
          }
        } else {
          buf.push_back(m[1]);
        }
        in = m.suffix().str();
      }

      // Construct dirname_ and object_name
      dirname_ = "";
      objname_ = "";
      int numberOfItems = buf.size();
      for (int i = 0; i < numberOfItems; i++) {
        if (i == numberOfItems - 1) {
          // Processing last component...
          if (type == Path::Type::DIR_AND_NAME) {
            objname_ = buf[i];
          } else if (type == Path::Type::DIR) {
            dirname_ += buf[i] + "/";
          }
        } else {
          dirname_ += buf[i] + "/";
        }
      }
    }

    bool operator==(Path const& other) const {
      return this->dirname_ == other.dirname_ && this->objname_ == other.objname_;
    }
  };

  // Metadata about the ME. The range is included here in case we have e.g.
  // multiple per-lumi histograms in one collection. For a logical comparison,
  // one should look only at the name.
  struct Key {
    Path path_;

    // Run number (and optionally lumi number) that the ME belongs to.
    edm::LuminosityBlockID id_;
    Scope scope_;
    Kind kind_;

    bool operator<(Key const& other) const {
      auto makeKeyTuple = [](Key const& k) {
        return std::make_tuple(
            k.path_.getDirname(), k.path_.getObjectname(), k.scope_, k.id_.run(), k.id_.luminosityBlock());
      };

      return makeKeyTuple(*this) < makeKeyTuple(other);
    }
  };

  bool operator<(MonitorElementData const& other) const { return this->key_ < other.key_; }

  // The only non class/struct members
  Key key_;
  Value value_;
};

// For now, no additional (meta-)data is needed apart from the MEs themselves.
// The framework will take care of tracking the plugin and LS/run that the MEs
// belong to.
// Unused for now.
class MonitorElementCollection {
  std::vector<std::unique_ptr<const MonitorElementData>> data_;

public:
  void push_back(std::unique_ptr<const MonitorElementData> value) {
    // enforce ordering
    assert(data_.empty() || data_[data_.size() - 1] <= value);
    data_.push_back(std::move(value));
  }

  void swap(MonitorElementCollection& other) { data_.swap(other.data_); }

  auto begin() const { return data_.begin(); }

  auto end() const { return data_.end(); }

  bool mergeProduct(MonitorElementCollection const& product) {
    // discussion: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePerRunAndPerLumiBlockData#Merging_Run_and_Luminosity_Block
    assert(!"Not implemented yet.");
    return false;
    // Things to decide:
    // - Should we allow merging collections of different sets of MEs? (probably not.) [0]
    // - Should we assume the MEs to be ordered? (probably yes.)
    // - How to handle incompatible MEs (different binning)? (fail hard.) [1]
    // - Can multiple MEs with same (dirname, objname) exist? (probably yes.) [2]
    // - Shall we modify the (immutable?) ROOT objects? (probably yes.)
    //
    // [0] Merging should increase the statistics, but not change the number of
    // MEs, at least with the current workflows. It might be convenient to
    // allow it, but for the beginning, it would only mask errors.
    // [1] The DQM framework should guarantee same booking parameters as long
    // as we stay within the Scope of the MEs.
    // [2] To implement e.g. MEs covering blocks of 10LS, we'd store them in a
    // run product, but have as many MEs with same name but different range as
    // needed to perserve the wanted granularity. Merging then can merge or
    // concatenate as possible/needed.
    // Problem: We need to keep copies in memory until the end of run, even
    // though we could save them to the output file as soon as it is clear that
    // the nexe LS will not fall into the same block. Instead, we could drop
    // them into the next lumi block we see; the semantics would be weird (the
    // MEs in the lumi block don't actually correspond to the lumi block they
    // are in) but the DQMIO output should be able to handle that.
  }
};

#endif
