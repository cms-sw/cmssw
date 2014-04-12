#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include <iostream>
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"

class CSCTFFilter : public edm::EDFilter {
   public:
      explicit CSCTFFilter(const edm::ParameterSet&);
      ~CSCTFFilter();


   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      std::vector<unsigned> modes;
      std::vector<unsigned>::const_iterator mode;
      edm::InputTag inputTag;
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
CSCTFFilter::CSCTFFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  modes = iConfig.getUntrackedParameter<std::vector<unsigned> >("modes");
  inputTag = iConfig.getUntrackedParameter<edm::InputTag >("inputTag");

}
CSCTFFilter::~CSCTFFilter()
{
}

// member functions
//

// ------------ method called to for each event  ------------
bool
CSCTFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  using namespace std;

  edm::Handle<L1CSCTrackCollection> trackFinderTracks;
  iEvent.getByLabel(inputTag,trackFinderTracks);
  L1CSCTrackCollection::const_iterator BaseTFTrk;
  for(BaseTFTrk=trackFinderTracks->begin();BaseTFTrk != trackFinderTracks->end(); BaseTFTrk++)
  {
    for(mode=modes.begin(); mode!=modes.end(); mode++)
    {
	if(BaseTFTrk->first.mode()== (*mode))
	{
		//cout << "mode: "<< *mode << endl;
		return true;
	}
    }
  }
  return false;
}


// ------------ method called once each job just before starting event loop  ------------
void 
CSCTFFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CSCTFFilter::endJob() 
{
}



//define this as a plug-in
DEFINE_FWK_MODULE(CSCTFFilter);
