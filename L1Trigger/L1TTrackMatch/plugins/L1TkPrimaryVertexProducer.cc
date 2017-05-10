// -*- C++ -*-
//
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// $Id$
//
//

// -------------------------------------------------------------------------------------------------------
//
//	********  OLD CODE   ********
//
//	********  The latest producer for the primary vertex is  L1TkFastVertexProducer.cc      ********
//
// --------------------------------------------------------------------------------------------------------



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"


//
// class declaration
//

class L1TkPrimaryVertexProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
   typedef std::vector< L1TTTrackType >  L1TTTrackCollectionType;

      explicit L1TkPrimaryVertexProducer(const edm::ParameterSet&);
      ~L1TkPrimaryVertexProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


      float MaxPtVertex(const edm::Handle<L1TTTrackCollectionType> & L1TTTrackHandle,
                float& sum,
                int nmin, int nPSmin, float ptmin, int imode,
		const TrackerTopology* topol) ;

      float SumPtVertex(const edm::Handle<L1TTTrackCollectionType> & L1TTTrackHandle,
                float z, int nmin, int nPSmin, float ptmin, int imode,
		const TrackerTopology* topol);


   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------

	float ZMAX;	// in cm
	float DeltaZ;	// in cm
	float CHI2MAX;
	float PTMINTRA ; 	// in GeV

	int nStubsmin ;		// minimum number of stubs 
	int nStubsPSmin ;	// minimum number of stubs in PS modules 
	bool SumPtSquared;

        const edm::EDGetTokenT< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;

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
L1TkPrimaryVertexProducer::L1TkPrimaryVertexProducer(const edm::ParameterSet& iConfig) :
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{
   //register your products
   //now do what ever other initialization is needed
  
  ZMAX = (float)iConfig.getParameter<double>("ZMAX");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");

  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  nStubsPSmin = iConfig.getParameter<int>("nStubsPSmin");

  SumPtSquared = iConfig.getParameter<bool>("SumPtSquared");

  produces<L1TkPrimaryVertexCollection>();

}


L1TkPrimaryVertexProducer::~L1TkPrimaryVertexProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TkPrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

 std::unique_ptr<L1TkPrimaryVertexCollection> result(new L1TkPrimaryVertexCollection);

  
  ////////////////////////
  // GET MAGNETIC FIELD //
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  if ( mMagneticFieldStrength < 0) std::cout << "mMagneticFieldStrength < 0 " << std::endl;  // for compil when not used

  //////////////////////
  // Tracker Topology //
  edm::ESHandle<TrackerTopology> tTopoHandle_;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle_);
  const TrackerTopology* tTopo = tTopoHandle_.product();


  edm::Handle<L1TTTrackCollectionType> L1TTTrackHandle;
  iEvent.getByToken(trackToken, L1TTTrackHandle);   


 if( !L1TTTrackHandle.isValid() )
        {
          LogError("L1TkPrimaryVertexProducer")
            << "\nWarning: LTTkTrackCollection not found in the event. Exit"
            << std::endl;
 	    return;
        }



   float sum1 = -999;
   int nmin = nStubsmin;
   int nPSmin = nStubsPSmin ;
   float ptmin = PTMINTRA ;
   int imode = 2;	// max(Sum PT2)
   if (! SumPtSquared)  imode = 1;   // max(Sum PT)

   float z1 = MaxPtVertex( L1TTTrackHandle, sum1, nmin, nPSmin, ptmin, imode, tTopo );
   L1TkPrimaryVertex vtx1( z1, sum1 );

 result -> push_back( vtx1 );

 iEvent.put( std::move(result) );
}


float L1TkPrimaryVertexProducer::MaxPtVertex(const edm::Handle<L1TTTrackCollectionType> & L1TTTrackHandle,
 		float& Sum,
		int nmin, int nPSmin, float ptmin, int imode,
		const TrackerTopology* topol) {
        // return the zvtx corresponding to the max(SumPT)
        // of tracks with at least nPSmin stubs in PS modules
   
      float sumMax = 0;
      float zvtxmax = -999;
      int nIter = (int)(ZMAX * 10. * 2.) ;
      for (int itest = 0; itest <= nIter; itest ++) {
	
        //float z = -100 + itest;         // z in mm
	float z = -ZMAX * 10 + itest ;  	// z in mm
        z = z/10.  ;   // z in cm
        float sum = SumPtVertex(L1TTTrackHandle, z, nmin, nPSmin, ptmin, imode, topol);
        if (sumMax >0 && sum == sumMax) {
          //cout << " Note: Several vertices have the same sum " << zvtxmax << " " << z << " " << sumMax << endl;
        }
   
        if (sum > sumMax) {
           sumMax = sum;
           zvtxmax = z;
        }
       }  // end loop over tested z 
   
 Sum = sumMax;
 return zvtxmax;
}  


float L1TkPrimaryVertexProducer::SumPtVertex(const edm::Handle<L1TTTrackCollectionType> & L1TTTrackHandle,
		float z, int nmin, int nPSmin, float ptmin, int imode,
		const TrackerTopology* topol) {

        // sumPT of tracks with >= nPSmin stubs in PS modules
        // z in cm
 float sumpt = 0;


  L1TTTrackCollectionType::const_iterator trackIter;

  for (trackIter = L1TTTrackHandle->begin(); trackIter != L1TTTrackHandle->end(); ++trackIter) {

    float pt = trackIter->getMomentum().perp();
    float chi2 = trackIter->getChi2();
    float ztr  = trackIter->getPOCA().z();

    if (pt < ptmin) continue;
    if (fabs(ztr) > ZMAX ) continue;
    if (chi2 > CHI2MAX) continue;
    if ( fabs(ztr - z) > DeltaZ) continue;   // eg DeltaZ = 1 mm


	// get the number of stubs and the number of stubs in PS layers
    float nPS = 0.;     // number of stubs in PS modules
    float nstubs = 0;

      // get pointers to stubs associated to the L1 track
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

      int tmp_trk_nstub = (int) theStubs.size();
      if ( tmp_trk_nstub < 0) {
	std::cout << " ... could not retrieve the vector of stubs in L1TkPrimaryVertexProducer::SumPtVertex " << std::endl;
	continue;
      }

      // loop over the stubs
      for (unsigned int istub=0; istub<(unsigned int)theStubs.size(); istub++) {
        //bool genuine = theStubs.at(istub)->isGenuine();
        //if (genuine) {
           nstubs ++;
	   bool isPS = false;
	   DetId detId( theStubs.at(istub)->getDetId() );
	   if (detId.det() == DetId::Detector::Tracker) {
	     if (detId.subdetId() == StripSubdetector::TOB && topol->tobLayer(detId) <= 3)  isPS = true;
	     else if (detId.subdetId() == StripSubdetector::TID && topol->tidRing(detId) <= 9)  isPS = true;
	   }
	   if (isPS) nPS ++;
	   //if (isPS) cout << " this is a stub in a PS module " << endl;
           if (isPS) nPS ++;
	//} // endif genuine
       } // end loop over stubs

        if (imode == 1 || imode == 2 ) {
            if (nPS < nPSmin) continue;
        }
	if ( nstubs < nmin) continue;

        if (imode == 2) sumpt += pt*pt;
        if (imode == 1) sumpt += pt;

  } // end loop over the tracks

 return sumpt;

}



// ------------ method called once each job just before starting event loop  ------------
void 
L1TkPrimaryVertexProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TkPrimaryVertexProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TkPrimaryVertexProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{


}
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkPrimaryVertexProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkPrimaryVertexProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkPrimaryVertexProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkPrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkPrimaryVertexProducer);
