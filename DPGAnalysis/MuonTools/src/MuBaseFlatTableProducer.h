#ifndef Mu_MuBaseFlatTableProducer_h
#define Mu_MuBaseFlatTableProducer_h

/** \class MuBaseFlatTableProducer MuBaseFlatTableProducer.h DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h
 *  
 * Helper class defining the generic interface of a FlatTableProducer
 *
 * \author C. Battilana (INFN BO)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuNtupleUtils.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include <string>

class MuBaseFlatTableProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit MuBaseFlatTableProducer(const edm::ParameterSet &);

  /// Configure event setup for each run
  void beginRun(const edm::Run &run, const edm::EventSetup &config) final;

  /// Fill ntuples event by event
  void produce(edm::Event &, const edm::EventSetup &) final;

  /// Empty, needed by interface
  void endRun(const edm::Run &, const edm::EventSetup &) final {}

protected:
  /// The label name of the FlatTableProducer
  std::string m_name;

  /// Get info from the ES by run
  virtual void getFromES(const edm::Run &run, const edm::EventSetup &environment) {}

  /// Get info from the ES for a given event
  virtual void getFromES(const edm::EventSetup &environment) {}

  /// Fill ntuple
  virtual void fillTable(edm::Event &ev) = 0;

  /// Definition of default values for int variables
  static constexpr int DEFAULT_INT_VAL{-999};

  /// Definition of default values for int8 variables
  static constexpr int8_t DEFAULT_INT8_VAL{-99};

  /// Definition of default values for positive int variables
  static constexpr int DEFAULT_INT_VAL_POS{-1};

  /// Definition of default values for float variables
  static constexpr double DEFAULT_DOUBLE_VAL{-999.0};

  /// Definition of default values for positive float variables
  static constexpr double DEFAULT_DOUBLE_VAL_POS{-1.0};

  template <typename T>
  void addColumn(std::unique_ptr<nanoaod::FlatTable> &table,
                 const std::string name,
                 const std::vector<T> &vec,
                 const std::string descr) {
    table->template addColumn<T, std::vector<T>>(name, vec, descr);
  }
};

#endif
