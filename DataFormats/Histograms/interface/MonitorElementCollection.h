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

 Usage: This product should only be handled by the DQMStore, which provides 
 access to the MEs inside. The DQMStore will wrap the MonitorElementData in
 real MonitorElements, which allow various operations on the underlying 
 histograms, depending on the current stage of processing: In the RECO step,
 only filling is allowed, while in HARVESTING, the same data will be wrapped in
 a MonitorElement that also allows access to the ROOT objects.

*/
//
// Original Author:  Marcel Schneider
//         Created:  2018-05-02
//
//
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

#include <cstdint>
#include <cassert>
#include <vector>
#include <string>

#include "TH1.h"


struct MonitorElementData
{
  // This is technically a union, but the struct is safer.
  struct Scalar
  {
    int64_t             num;
    double              real;
    std::string         str;
  };

  // These values are compatible to DQMNet, but DQMNet is not likely to exist
  // in the future.
  // Maybe this declaration should be moved somewhere else, MonitorElement::Kind
  // is used in a lot of places. Can one `using` an enum?
  enum Kind
  {
    DQM_KIND_INVALID    = 0x0,
    DQM_KIND_INT        = 0x1,
    DQM_KIND_REAL       = 0x2,
    DQM_KIND_STRING     = 0x3,
    DQM_KIND_TH1F       = 0x10,
    DQM_KIND_TH1S       = 0x11,
    DQM_KIND_TH1D       = 0x12,
    DQM_KIND_TH2F       = 0x20,
    DQM_KIND_TH2S       = 0x21,
    DQM_KIND_TH2D       = 0x22,
    DQM_KIND_TH3F       = 0x30,
    DQM_KIND_TPROFILE   = 0x40,
    DQM_KIND_TPROFILE2D = 0x41
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
  enum Scope
  {
    DQM_SCOPE_JOB = 1,
    DQM_SCOPE_RUN = 2,
    DQM_SCOPE_LUMI = 3,
    DQM_SCOPE_DEFAULT = 4 /* = DQM_SCOPE_RUN? */
  };

  // The main ME data. We don't keep references/QTest results, instead we use
  // only the fields stored in DQMIO files.
  Kind kind;
  Scalar scalar;
  TH1* object;
  // ROOT will serialize that correctly, I hope? or do we need to do the 
  // template dance as in MEtoEDM? 

  // Metadata about the ME.
  // We could use pointers to interned strings here to save some space.
  std::string dirname;
  std::string objname;

  // The range from the first to the last event that actually went into this
  // histogram. When merging, we extend this range; merging overlapping but not
  // identical ranges should probably be an error, see the Mergable Products 
  // discussion: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePerRunAndPerLumiBlockData#Merging_Run_and_Luminosity_Block
  // We could also keep event numbers, to make it easier to see if there is
  // double counting for debugging.
  edm::LuminosityBlockRange coveredrange;
  Scope scope;
  
  // Copying this stucture would be dangerous due to the ROOT object pointer,
  // but moving should be fine.
  MonitorElementData() = default;
  MonitorElementData(MonitorElementData const&) = delete;
  MonitorElementData(MonitorElementData&&) = default;
  // We don't delete the ROOT object at destruction so that it is easier/safer
  // to use this as a base class for the actual, mutable ME classes.
  ~MonitorElementData() = default;
  
  // TODO: We'll probably need a total order on the MEs for any sort of 
  // efficient data structure. Would not hurt to define it here, to avoid 
  // confusion. It should probably include:
  // (dirname, objname, scope, beginrun, beginlumi, endrun, endlumi)
  // where the latter items come from the coveredrange.
};


// For now, no additional (meta-)data is needed apart from the MEs themselves.
// The framework will take care of tracking the plugin and LS/run that the MEs
// belong to.
// TODO: would it be legal/better to use MonitorElementData* here?
// The ROOT objects hang on pointers anyways. And the MonitorElementCollection
// owns these objects, once they are put in here. We also need to agree with
// ROOT on that topic.
// TODO: we could use a set or map keyed by the (dirname, objname), but that
// seems to be not really required here. We use a more advanced structure in
// the DQMStore, while this type is only exported/imported there.
class MonitorElementCollection : public std::vector<MonitorElementData>
{
   public:
  MonitorElementCollection() {}
  ~MonitorElementCollection() 
  {
    for (auto& me : *this) {
      if (me.object) {
        delete me.object;
        me.object = nullptr;
      }
    }
  }

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
