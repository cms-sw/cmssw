// -*- C++ -*-
//
// Package:    HcalLaserHFFilter2012
// Class:      HcalLaserHFFilter2012
// 
/**\class HcalLaserHFFilter2012 HcalLaserHFFilter2012.cc UserCode/HcalLaserHFFilter2012/src/HcalLaserHFFilter2012.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Fri Oct 19 13:15:44 EDT 2012
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"


//
// class declaration
//

class HcalLaserHFFilter2012 : public edm::EDFilter {
public:
  explicit HcalLaserHFFilter2012(const edm::ParameterSet&);
  ~HcalLaserHFFilter2012();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual bool beginRun(edm::Run&, edm::EventSetup const&);
  virtual bool endRun(edm::Run&, edm::EventSetup const&);
  virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  
  // ----------member data ---------------------------
  bool verbose_;  // if set to true, then the run:LS:event for any event failing the cut will be printed out
  std::string prefix_;  // prefix will be printed before any event if verbose mode is true, in order to make searching for events easier
  int minCalibChannelsHFLaser_; // set minimum number of HF Calib events that causes an event to be considered a bad (i.e., HF laser) event
  edm::InputTag     digiLabel_;

  bool WriteBadToFile_;
  bool forceFilterTrue_;
  std::ofstream outfile_;
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
HcalLaserHFFilter2012::HcalLaserHFFilter2012(const edm::ParameterSet& ps)
{
   //now do what ever initialization is needed
  verbose_ = ps.getUntrackedParameter<bool>("verbose",false);
  prefix_  = ps.getUntrackedParameter<std::string>("prefix","");
  minCalibChannelsHFLaser_=ps.getUntrackedParameter<int>("minCalibChannelsHFLaser",10);
  edm::InputTag digi_default("hcalDigis");
  digiLabel_     = ps.getUntrackedParameter<edm::InputTag>("digiLabel",digi_default);
  WriteBadToFile_=ps.getUntrackedParameter<bool>("WriteBadToFile",false);
  if (WriteBadToFile_)
    outfile_.open("badHcalLaserList_hffilter.txt");
  forceFilterTrue_=ps.getUntrackedParameter<bool>("forceFilterTrue",false);

} // HcalLaserHFFilter2012::HcalLaserHFFilter2012  constructor


HcalLaserHFFilter2012::~HcalLaserHFFilter2012()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HcalLaserHFFilter2012::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Step 1:: try to get calib digi collection.
  // Return true if collection not found?  Or false?  What should default behavior be?
  edm::Handle<HcalCalibDigiCollection> calib_digi;
   if (!(iEvent.getByLabel(digiLabel_,calib_digi)))
    {
      edm::LogWarning("HcalLaserHFFilter2012")<< digiLabel_<<" calib_digi not available";
      return true;
    }
   
   if (!(calib_digi.isValid()))
     {
       edm::LogWarning("HcalLaserHFFilter2012")<< digiLabel_<<" calib_digi is not valid";
       return true;
    }

  // Step 2:  Count HF digi calib channels
  int ncalibHF=0; // this will track number of HF digi channels

  
  for (HcalCalibDigiCollection::const_iterator Calibiter = calib_digi->begin();
       Calibiter != calib_digi->end(); ++ Calibiter)
     {
       const HcalCalibDataFrame digi = (const HcalCalibDataFrame)(*Calibiter);
       HcalCalibDetId myid=(HcalCalibDetId)digi.id();
       if (myid.hcalSubdet()!=HcalForward) continue;
       ++ncalibHF;
       if (ncalibHF>=minCalibChannelsHFLaser_)
	 {
	   if (verbose_) std::cout <<prefix_<<iEvent.id().run()<<":"<<iEvent.luminosityBlock()<<":"<<iEvent.id().event()<<std::endl;
	   if (WriteBadToFile_)
	     outfile_<<iEvent.id().run()<<":"<<iEvent.luminosityBlock()<<":"<<iEvent.id().event()<<std::endl;	  
	   if (forceFilterTrue_) return true; // if special input boolean set, always return true, regardless of filter decision
	   else return false;
	 }
     }

   return true;
}  // HcalLaserHFFilter2012::filter
 
// ------------ method called once each job just before starting event loop  ------------
void 
HcalLaserHFFilter2012::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalLaserHFFilter2012::endJob() {
  if (WriteBadToFile_) outfile_.close();
}

// ------------ method called when starting to processes a run  ------------
bool 
HcalLaserHFFilter2012::beginRun(edm::Run&, edm::EventSetup const&)
{ 
  return true;
}

// ------------ method called when ending the processing of a run  ------------
bool 
HcalLaserHFFilter2012::endRun(edm::Run&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when starting to processes a luminosity block  ------------
bool 
HcalLaserHFFilter2012::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method called when ending the processing of a luminosity block  ------------
bool 
HcalLaserHFFilter2012::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
  return true;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HcalLaserHFFilter2012::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalLaserHFFilter2012);
