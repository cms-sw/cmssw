// -*- C++ -*-
//
// Package:    EcalBadScFilter
// Class:      EcalBadScFilter
//
/**\class EcalBadSCFilter EcalBadScFilter.cc

 Description: <one line class summary>
 Event filtering to remove events with anomalous energy in EE supercrystals
*/
//
// Original Authors:  K. Theofilatos and D. Petyt
//


// include files
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "TVector3.h"

class EEBadScFilter : public edm::EDFilter {

  public:

    explicit EEBadScFilter(const edm::ParameterSet & iConfig);
    ~EEBadScFilter() {}

  private:

  // main filter function

  virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

  // function to calculate 5x5 energy and check rechit flags

  virtual void scan5x5(const DetId & det, const edm::Handle<EcalRecHitCollection> &hits, const edm::ESHandle<CaloTopology>  &caloTopo, const edm::ESHandle<CaloGeometry>  &geometry, int &nHits, float & totEt);

  // input parameters

  // ee rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  eeRHSrcToken_;

  //config parameters (defining the cuts on the bad SCs)
  const double Emin_;               // rechit energy threshold (check for !kGood rechit flags)
  const double EtminSC_;            // et threshold for the supercrystal
  const int side_;                  // supercrystal size  (default = 5x5 crystals)
  const int nBadHitsSC_;            // number of bad hits in the SC to reject the event
  const std::vector<int> badsc_;    // crystal coordinates of the bad SCs (central xtal in the 5x5)

  const bool taggingMode_;
  const bool debug_;                // prints out debug info if set to true

};

// read the parameters from the config file
EEBadScFilter::EEBadScFilter(const edm::ParameterSet & iConfig)
  : eeRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , Emin_        (iConfig.getParameter<double>("EminHit"))
  , EtminSC_     (iConfig.getParameter<double>("EtminSC"))
  , side_        (iConfig.getParameter<int>("SCsize"))
  , nBadHitsSC_  (iConfig.getParameter<int>("nBadHitsSC"))
  , badsc_       (iConfig.getParameter<std::vector<int> >("badscEE"))
  , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
  , debug_       (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}


void EEBadScFilter::scan5x5(const DetId & det, const edm::Handle<EcalRecHitCollection> &hits, const edm::ESHandle<CaloTopology>  &caloTopo, const edm::ESHandle<CaloGeometry>  &geometry, int &nHits, float & totEt)
{

  // function to compute:  total transverse energy in a given supercrystal (totEt)
  //                       number of hits with E>Emin_ and rechit flag != kGood (nHits)
  // bad events have large totEt and many high energy hits with rechit flags !kGood

  nHits = 0;
  totEt = 0;

  // navigator to define a 5x5 region around the input DetId

  CaloNavigator<DetId> cursor = CaloNavigator<DetId>(det,caloTopo->getSubdetectorTopology(det));

  // loop over a 5x5 array centered on the input DetId

  for(int j=side_/2; j>=-side_/2; --j)
    {
      for(int i=-side_/2; i<=side_/2; ++i)
	{
	  cursor.home();
	  cursor.offsetBy(i,j);
	  if(hits->find(*cursor)!=hits->end()) // if hit exists in the rechit collection
	    {
	      EcalRecHit tmpHit = *hits->find(*cursor); // get rechit with detID at cursor


	      const GlobalPoint p ( geometry->getPosition(*cursor) ) ; // calculate Et of the rechit
	      TVector3 hitPos(p.x(),p.y(),p.z());
	      hitPos *= 1.0/hitPos.Mag();
	      hitPos *= tmpHit.energy();
	      float rechitEt =  hitPos.Pt();

	      //--- add rechit E_t to the total for this supercrystal
	      totEt += rechitEt;

	      // increment nHits if E>Emin and rechit flag is not kGood
	      if(tmpHit.energy()>Emin_ && !tmpHit.checkFlag(EcalRecHit::kGood))nHits++;

	    }
	}
    }


}





bool EEBadScFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {


  // load required collections


  // EE rechit collection
  edm::Handle<EcalRecHitCollection> eeRHs;
  iEvent.getByToken(eeRHSrcToken_, eeRHs);

  // Calo Geometry - needed for computing E_t
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);

  // Calo Toplology - needed for navigating the 5x5 xtal array around the centre of a SC
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);

  // by default the event is OK
  bool pass = true;


  // set discriminating variables to zero
  int nhits=0;
  float totEt=0.0;


  // define detid variables and ix,iy,iz coordinates

  EEDetId scdet;
  DetId det;

  int ix,iy,iz;

  ix=0,iy=0,iz=0;


  // loop over the list of bad SCs (defined in the python file)

  for (std::vector<int>::const_iterator scit = badsc_.begin(); scit != badsc_.end(); ++ scit) {



    // unpack the SC coordinates from the python file into ix,iy,iz

    iz=int(*scit/1000000);
    iy=*scit%100*iz;
    ix=int((*scit-iy-1000000*iz)/1000)*iz;

    // make the DetId from these coordinates
    scdet=EEDetId(ix,iy,iz);
    det=scdet;

    // loop over the 5x5 SC centered on this DetId and fill discriminating variables

    scan5x5(det,eeRHs,pTopology,pG,nhits,totEt);

    // print some debug info

    if (debug_) {
      std::cout << "EEBadScFilter.cc:  SCID=" <<  *scit << std::endl;
      std::cout << "EEBadScFilter.cc:  ix=" << ix << " iy=" << iy << " iz=" << iz << std::endl;
      std::cout << "EEBadScFilter.cc:  Et(5x5)=" << totEt << " nbadhits=" << nhits << std::endl;
    }


    // if 5x5 transverse energy is above threshold and number of bad hits above threshold
    // event is bad

    if (totEt>EtminSC_ && nhits>=nBadHitsSC_) pass=false;


  }

  // print the decision if event is bad
  if (pass==false && debug_) std::cout << "EEBadScFilter.cc:  REJECT EVENT!!!" << std::endl;


  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  // return the decision

  return taggingMode_ || pass;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EEBadScFilter);
