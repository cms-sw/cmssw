#ifndef L1Trigger_TrackerDTC_DTC_h
#define L1Trigger_TrackerDTC_DTC_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/StubFE.h"
#include "L1Trigger/TrackerDTC/interface/StubGL.h"
#include "L1Trigger/TrackerDTC/interface/StubDTC.h"

#include <vector>
#include <string>
#include <sstream>

#include <TProfile.h>
#include <TH1F.h>
#include <TH2F.h>

namespace trackerDTC {

  /*! \class  trackerDTC::DTC
   *  \brief  Class to represent an outer tracker DTC board
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class DTC {
  public:
    // configuration
    struct Config {
      // enables comparison of s/w with f/w
      bool enable;
      // modelsim simulation time in us
      double runTime;
      // number of 8 bx boxcars played in one test
      int num8BX;
      // number of 18 bx boxcars played in one test
      int num18BX;
      // path to ipbb proj area
      std::string pathIPBB;
    };
    DTC(const DTC&) = delete;
    DTC(DTC&&) = default;
    DTC(const Setup*, const Config&, int, std::vector<TH1F*>&, std::vector<TProfile*>&, TH2F*, TH2F*);
    ~DTC() = default;
    // process single bx
    void consume(const edm::Handle<TTStubDetSetVec>&, int);
    // process 8 bx boxcars
    void produce(int);
    // compare s/w with f/w
    void analyze();

  private:
    // create emulation input
    void produce(std::vector<tt::Stream>&);
    // emulate output
    void produce(const tt::Streams&, tt::Streams&);
    // apply mpa rules, 4 stubs per 2 bx row priority, up to 2 stubs with same pos but different bend
    void mpa(const std::deque<const StubFE*>&, std::deque<const StubFE*>&, int);
    // apply cbc rules, 3 stubs per bx row priority
    void cbc(const std::deque<const StubFE*>&, std::deque<const StubFE*>&, int);
    // read in input data
    void convert(const tt::Streams&, std::vector<std::deque<const StubFE*>>&) const;
    // convert front end stubs to global stubs
    void produce(const std::vector<std::deque<const StubFE*>>&,
                 std::vector<StubGL>&,
                 std::vector<std::deque<const StubGL*>>&) const;
    // convert stubs to streams
    void convert(const std::vector<std::deque<const StubDTC*>>&, tt::Streams&);
    // emualte routing 9 links -> 8 single bx streams
    void unbox(const std::vector<std::deque<const StubGL*>>&, std::vector<std::deque<const StubGL*>>&) const;
    // emulate 8BX -> 12BX repacking and 12BX -> 18BX
    void repack(
        const std::vector<std::deque<const StubGL*>>&, std::vector<std::deque<const StubGL*>>&, int, int, int) const;
    // emulate routing 2 -> 2 phi region splitting
    void produce(const std::vector<std::deque<const StubGL*>>&,
                 std::vector<StubDTC>&,
                 std::vector<std::deque<const StubDTC*>>&) const;
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    template <class Stub>
    void merge(std::vector<std::deque<const Stub*>>&, std::deque<const Stub*>&, bool = false) const;
    // fill occopancy histos
    template <class Stub>
    void fill(std::vector<std::deque<const Stub*>>&, int);
    // fill lost stubs histo
    void lost(const std::deque<const StubFE*>&, int);
    // fill lost stubs histo
    void lost(const std::deque<const StubGL*>&, int);
    // fill lost stubs histo
    void lost(const std::deque<const StubDTC*>&, int);
    // emulate truncation
    template <class Stub>
    void truncate(std::vector<std::deque<const Stub*>>&, int);
    // pop_front function which additionally returns copy of deleted front
    template <class Stub>
    const Stub* pop_front(std::deque<const Stub*>& Stream) const;
    // helper class to store configurations
    const Setup* setup_;
    // configuration
    const Config* config_;
    // DTC id  of this DTC
    int dtcId_;
    // channel occupancy histograms
    std::vector<TH1F*>& his_;
    // channel occupancy profiles
    std::vector<TProfile*>& prof_;
    // stub counts rz plane
    TH2F* hisRZStubs_;
    // lost stub counts rz plane
    TH2F* hisRZLost_;
    // stub storage
    std::vector<std::vector<StubFE>> stubsFE_;
    // structured input from out to in: module, stubs
    std::vector<std::deque<const StubFE*>> streamsFE_;
    // dtc input streams
    std::vector<tt::Stream> streamsIn_;
    // dtc output streams
    std::vector<tt::Stream> streamsOut_;
    // questasim input text file header
    std::stringstream headerIn_;
    // questasim predicted output text file header
    std::stringstream headerPre_;
    // path to questasim input text
    std::string pathIn_;
    // path to questasim predicted output text
    std::string pathPre_;
    // path questasim output text
    std::string pathOut_;
    // path to diff of questasim predicted output text file vs output text file
    std::string pathDiff_;
    // command string calling qustasim and comparing text files
    std::stringstream cmd_;
  };

}  // namespace trackerDTC

#endif
