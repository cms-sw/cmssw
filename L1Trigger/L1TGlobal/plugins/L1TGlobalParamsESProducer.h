#ifndef L1Trigger_L1TGlobalParamsESProducer_h
#define L1Trigger_L1TGlobalParamsESProducer_h

// This is the ES producer for the L1T Global Stable Parameters.
// This version is used for the updated uGT emulator of April 2016.
// It deprecates StableParametersTrivialProducer.

// system include files
#include <memory>
#include <iostream>
#include <vector>

#include "boost/shared_ptr.hpp"
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "L1Trigger/L1TGlobal/interface/GlobalParamsHelper.h"

// forward declarations

// class declaration
class L1TGlobalParamsESProducer : public edm::ESProducer
{

public:

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    /// constructor
    L1TGlobalParamsESProducer(const edm::ParameterSet&);

    /// destructor
    ~L1TGlobalParamsESProducer();

    /// public methods

    typedef boost::shared_ptr<L1TGlobalParameters> ReturnType;

    /// L1 GT parameters
    ReturnType produce(const L1TGlobalParametersRcd&);

private:

    l1t::GlobalParamsHelper data_;
};

#endif
