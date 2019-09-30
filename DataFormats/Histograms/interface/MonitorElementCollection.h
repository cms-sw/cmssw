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
 Once a MonitorElement enters this product, it is immutable. During event
 processing, the ROOT objects need to be protectd by locks. These locks are not
 present in this structure: Any potential modification needs to be done as a
 copy-on-write and create a new MonitorElement.
 The file IO for these objects should still be handled by the DQMIO classes
 (DQMRootOutputModule and DQMRootSource), so persistent=false would be ok for
 this class. However, if we can get EDM IO cheaply, it could be useful to 
 handle corner cases like MEtoEDM more cleanly.
 TODO: We use persistent=false now, since ROOT really wants to be able to copy
 things (which is not compatible with unique_ptr and mutex).

 Usage: This product should only be handled by the DQMStore, which provides 
 access to the MEs inside. The DQMStore will wrap the MonitorElementData in
 real MonitorElements, which allow various operations on the underlying 
 histograms, depending on the current stage of processing: In the RECO step,
 only filling is allowed, while in HARVESTING, the same data will be wrapped in
 a MonitorElement that also allows access to the ROOT objects.
 We only use pointers to MonitorElementData, to allow replacing it with a
 variable-size templated version later. That could eliminate one level of 
 indirection in accessing histograms.

*/
//
// Original Author:  Marcel Schneider
//         Created:  2018-05-02
//
//
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <mutex>
#include <regex>

#include "TH1.h"

struct MonitorElementData {
  // This is technically a union, but the struct is safer.
  struct Scalar {
    int64_t num = 0;
    double real = 0;
    std::string str;
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

  // Which window of time the ME is supposed to cover. How much data is actually
  // covered is tracked separately; these values define when things should be
  // merged.
  // There is space for a granularity level between runs and lumisections,
  // maybe blocks of 10LS or some fixed number of events or integrated
  // luminosity. We also want to be able to change the granularity centrally
  // depending on the use case. That is what the DEFAULT is for, and it should
  // be used unless some specific granularity is really required.
  // We'll also need to switch the DEFAULT to JOB for multi-run harvesting.
  enum Scope { JOB = 1, RUN = 2, LUMI = 3, DEFAULT = RUN };

  // The main ME data. We don't keep references/QTest results, instead we use
  // only the fields stored in DQMIO files.
  struct Value {
    // The lock protects the data while filling. There is no point in having it
    // here except for convenience.
    // "mutable" to allow thread-safe "Fill" to change it without a const-cast
    // const-casting would be more logically correct, but risks UB.
    // TODO: It should be ok to use a TH1 value here, but we'd have to template
    // TODO: ConcurrentME used a tbb::spin_lock, check if we can do that here as
    // well (dependencies!)
    // TODO: This really, really, should be unique_ptr. But ROOT wants to copy it
    // for serialization.
    // TODO: The lock_ should of course not be serialized.
  private:
    mutable Scalar scalar_;
    mutable std::unique_ptr<TH1> object_;
    mutable std::mutex lock_;

  public:
    // To access the data, including for initializing it, make a
    // MonitorElementData::Value::Access instance and use its fields. It should
    // hold the lock as long as it exists.
    struct Access {
      std::scoped_lock<std::mutex> lock;
      Scalar& scalar;
      std::unique_ptr<TH1>& object;
      Access(MonitorElementData::Value const& value)
          : lock(value.lock_), scalar(value.scalar_), object(value.object_){};
    };
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

    // Clean up the path and normalize it to preserve certain invariants.
    // Instead of reasoning about whatever properties of paths, we just parse
    // the thing and build a normalized instance with no slash in the beginning
    // and a slash in the end.
    // Type of string `path` could be just directory name, or
    // directory name followed by the name of the monitor element
    void set(std::string path, Path::Type type) {
      std::string in(path);
      std::vector<std::string> buf;
      std::regex dir("^/*([^/]+)");
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

    // The range from the first to the last event that actually went into this
    // histogram. When merging, we extend this range; merging overlapping but not
    // identical ranges should probably be an error, see the Mergable Products
    // discussion: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePerRunAndPerLumiBlockData#Merging_Run_and_Luminosity_Block
    // We could also keep event numbers, to make it easier to see if there is
    // double counting for debugging.
    edm::LuminosityBlockRange coveredrange_;
    Scope scope_;
    Kind kind_;

    bool operator<(Key const& other) const {
      auto makeKeyTuple = [](Key const& k) {
        return std::make_tuple(k.path_.getDirname(),
                               k.path_.getObjectname(),
                               k.scope_,
                               k.coveredrange_.startRun(),
                               k.coveredrange_.startLumi(),
                               k.coveredrange_.endRun(),
                               k.coveredrange_.endLumi());
      };

      return makeKeyTuple(*this) < makeKeyTuple(other);
    }
  };

  bool operator<(MonitorElementData const& other) const { return this->key_ < other.key_; }

  // The only non class/struct members
  Key key_;
  Value value_;
};

// TODO: We should not use edm::OwnVector once we can, then this can go away
struct FakeMEDataClone {
  static MonitorElementData* clone(MonitorElementData const&) {
    assert(!"This is to make EDM happy.");
    return nullptr;
  };
};

// For now, no additional (meta-)data is needed apart from the MEs themselves.
// The framework will take care of tracking the plugin and LS/run that the MEs
// belong to.
// TODO: move away from OwnVector once we can.
// TODO: what about mergeProduct? Maybe we need a class here, after all.
using MonitorElementCollection = edm::OwnVector<const MonitorElementData, FakeMEDataClone>;

// Only to hold the mergeProduct placeholder for now.
class MonitorElementCollectionHelper {
public:
  bool mergeProduct(MonitorElementCollection const& product) {
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
