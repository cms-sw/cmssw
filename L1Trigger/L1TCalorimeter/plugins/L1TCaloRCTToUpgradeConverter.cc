// ------------ method called to produce the data  ------------
void
L1TCaloRCTToUpgradeConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloRCTToUpgradeConverter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloRCTToUpgradeConverter::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TCaloRCTToUpgradeConverter::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TCaloRCTToUpgradeConverter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TCaloRCTToUpgradeConverter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup cons
t&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TCaloRCTToUpgradeConverter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&
)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TCaloRCTToUpgradeConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloRCTToUpgradeConverter);
