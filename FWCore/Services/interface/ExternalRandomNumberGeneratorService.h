#ifndef FWCore_Services_ExternalRandomNumberGeneratorService_h
#define FWCore_Services_ExternalRandomNumberGeneratorService_h

/** \class edm::ExternalRandomNumberGenerator

  Description: Interface for obtaining random number engines.

  Usage:
*/

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

namespace edm {

  class ExternalRandomNumberGeneratorService : public RandomNumberGenerator {
  public:
    ExternalRandomNumberGeneratorService();
    ExternalRandomNumberGeneratorService(ExternalRandomNumberGeneratorService const&) = delete;
    ExternalRandomNumberGeneratorService const& operator=(ExternalRandomNumberGeneratorService const&) = delete;

    void setState(std::vector<unsigned long> const&, long seed);
    std::vector<unsigned long> getState() const;

    CLHEP::HepRandomEngine& getEngine(StreamID const&) final;
    CLHEP::HepRandomEngine& getEngine(LuminosityBlockIndex const&) final;
    std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(LuminosityBlockIndex const&) final;
    std::uint32_t mySeed() const final;

    // The following functions should not be used by general users.  They
    // should only be called by Framework code designed to work with the
    // service while it is saving the engine states or restoring them.
    // The first two are called by the EventProcessor at special times.
    // The next two are called by a dedicated producer module (RandomEngineStateProducer).

    void preBeginLumi(LuminosityBlock const& lumi) final;
    void postEventRead(Event const& event) final;

    void setLumiCache(LuminosityBlockIndex, std::vector<RandomEngineState> const& iStates) final;
    void setEventCache(StreamID, std::vector<RandomEngineState> const& iStates) final;

    std::vector<RandomEngineState> const& getEventCache(StreamID const&) const final;
    std::vector<RandomEngineState> const& getLumiCache(LuminosityBlockIndex const&) const final;

    void consumes(ConsumesCollector&& iC) const final;

    /// For debugging purposes only.
    void print(std::ostream& os) const final;

  private:
    std::unique_ptr<CLHEP::HepRandomEngine> createFromState(std::vector<unsigned long> const&, long seed) const;

    std::unique_ptr<CLHEP::HepRandomEngine> engine_;
  };
}  // namespace edm
#endif
