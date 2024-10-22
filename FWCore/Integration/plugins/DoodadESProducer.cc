// -*- C++ -*-
//
// Package:    DoodadESProducer
// Class:      DoodadESProducer
//
/**\class DoodadESProducer DoodadESProducer.h test/DoodadESProducer/interface/DoodadESProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 14:33:04 EDT 2005
//
//

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "Doodad.h"
#include "GadgetRcd.h"

// system include files
#include <memory>
#include <stdexcept>

//
// class decleration
//
namespace edmtest {

  class DoodadESProducer : public edm::ESProducer {
  public:
    DoodadESProducer(edm::ParameterSet const&);
    ~DoodadESProducer() override;

    typedef std::unique_ptr<Doodad> ReturnType;

    ReturnType produce(GadgetRcd const&);

  private:
    // ----------member data ---------------------------
    bool exceptionInformation_;
  };

  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions

  //
  // constructors and destructor
  //
  DoodadESProducer::DoodadESProducer(edm::ParameterSet const& iConfig)
      : exceptionInformation_(iConfig.getUntrackedParameter<bool>("throwException", false)) {
    //the following line is needed to tell the framework what
    // data is being produced
    setWhatProduced(this);

    //now do what ever other initialization is needed
  }

  DoodadESProducer::~DoodadESProducer() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  DoodadESProducer::ReturnType DoodadESProducer::produce(GadgetRcd const& /*iRecord*/) {
    if (exceptionInformation_) {
      throw std::runtime_error("test exception catching");
    }

    using namespace edmtest;

    auto pDoodad = std::make_unique<Doodad>();

    pDoodad->a = 1;

    return pDoodad;
  }
}  // namespace edmtest

using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(DoodadESProducer);
