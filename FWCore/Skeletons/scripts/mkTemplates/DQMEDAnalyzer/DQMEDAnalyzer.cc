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
@example_stream#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
@example_global#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

@example_stream#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

@example_globalstruct Histograms___class__ {
@example_global ConcurrentMonitorElement histo_;
@example_global};

@example_streamclass __class__ : public DQMEDAnalyzer {
@example_globalclass __class__ : public DQMGlobalEDAnalyzer<Histograms___class__> {
   public:
      explicit __class__(const edm::ParameterSet&);
      ~__class__();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
@example_stream      virtual void bookHistograms(DQMStore::IBooker &,
@example_stream                                  edm::Run const&,
@example_stream                                  edm::EventSetup const&) override;
@example_global      virtual void bookHistograms(DQMStore::ConcurrentBooker &,
@example_global                                  edm::Run const&,
@example_global                                  edm::EventSetup const&,
@example_global                                  Histograms___class__&) const override;

@example_stream      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
@example_global      virtual void dqmAnalyze(edm::Event const&,
@example_global                              edm::EventSetup const&,
@example_global                              Histograms___class__ const&) const override;

      // ----------member data ---------------------------
      std::string folder_;
@example_stream      MonitorElement * example_;
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
@example_streamvoid
@example_stream__class__::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
@example_stream{
@example_stream   using namespace edm;
@example_stream}
@example_stream
@example_streamvoid
@example_stream__class__::bookHistograms(DQMStore::IBooker & ibook,
@example_stream                          edm::Run const& run,
@example_stream                          edm::EventSetup const & iSetup)
@example_stream{
@example_stream  ibook.setCurrentFolder(folder_);
@example_stream
@example_stream  ibook.book1D("EXAMPLE", "EXAMPLE", 10, 0., 10.);
@example_stream}

@example_globalvoid
@example_global__class__::dqmAnalyze(edm::Event const& iEvent, edm::EventSetup const& iSetup,
@example_global                      Histograms___class__ const & histos) const
@example_global{
@example_global}
@example_global

@example_globalvoid
@example_global__class__::bookHistograms(DQMStore::ConcurrentBooker & ibook,
@example_global                          edm::Run const & run,
@example_global                          edm::EventSetup const & iSetup,
@example_global                          Histograms___class__ & histos) const
@example_global{
@example_global  ibook.setCurrentFolder(folder_);
@example_global  histos.histo_ = ibook.book1D("EXAMPLE", "EXAMPLE", 10, 0., 10.);
@example_global}

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
