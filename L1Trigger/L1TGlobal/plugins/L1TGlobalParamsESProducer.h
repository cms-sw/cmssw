#ifndef L1Trigger_L1TGlobal_StableParametersTrivialProducer_h
#define L1Trigger_L1TGlobal_StableParametersTrivialProducer_h

/**
 * \class L1TGlobalParamsESProducer
 * 
 * 
 * Description: ESProducer for L1 GT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

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
