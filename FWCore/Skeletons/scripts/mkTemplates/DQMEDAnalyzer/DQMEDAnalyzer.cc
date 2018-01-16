// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
//
/**\class __class__ __class__.cc __subsys__/__pkgname__/plugins/__class__.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//

#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class __class__ : public DQMEDAnalyzer {
   public:
      explicit __class__(const edm::ParameterSet&);
      ~__class__();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void bookHistograms(DQMStore::IBooker &,
                                  edm::Run const&,
                                  edm::EventSetup const&) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      std::string folder_;
      MonitorElement * example_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
__class__::__class__(const edm::ParameterSet& iConfig)
  : folder_(iConfig.getUntrackedParameter<std::string>("folder"))
{
   //now do what ever initialization is needed
}


__class__::~__class__()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void
__class__::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
}

void
__class__::bookHistograms(DQMStore::IBooker & ibook,
                          edm::Run const& run,
                          edm::EventSetup const & iSetup)
{
  ibook.setCurrentFolder(folder_);

  ibook.book1D("EXAMPLE", "EXAMPLE", 10, 0., 10.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
__class__::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "MY_FOLDER");
  descriptions.add("__class_lowercase__", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(__class__);
