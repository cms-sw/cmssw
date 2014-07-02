#include "RecoParticleFlow/PFClusterProducer/plugins/PFArborLinker.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitCreator.h"

using namespace std;
using namespace edm;


PFArborLinker::PFArborLinker(const edm::ParameterSet& iConfig):
  hits_(consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<reco::PFClusterCollection>("allLinks");
}


void PFArborLinker::produce(edm::Event& iEvent, 
			    const edm::EventSetup& iSetup) {

  //Read seeds
  edm::Handle<reco::PFRecHitCollection> hits; 
  iEvent.getByToken(hits_,hits);
  std::auto_ptr<reco::PFClusterCollection> out(new reco::PFClusterCollection);

  std::vector<bool> seed;
  seed.clear();

//  cout<<"AAAAAAAAAAAAAAA"<<endl; 

  for(unsigned int j = 0; j < (hits->size()); ++j) 
  {
	  bool maximum=true;

//	cout<<"BBBBBBBBBBBBB"<<enld;

	  for (const auto& neighbour : hits->at(j).neighbours8()) {
		  if (hits->at(j).energy()<neighbour->energy()) {
			  maximum=false;
			  break;
		  }
	  }
	  if (maximum)
	          seed.push_back(true);
	  else
		  seed.push_back(false);

	cout<<seed[j]<<endl; 
  }


  std::vector<TVector3> inputArborHits;
  std::vector<TVector3> ArborIntegralHits; 
  TVector3 currhitPos, hitID; 
  branchcoll ArborBranch; 
  // float cellsize = 25;
  // float layerthickness = 80; 

  for (unsigned int i=0;i<hits->size();++i) {

	 //  if(seed[i])
	  {
		  HcalDetId a( hits->at(i).detId() );
		  hitID.SetXYZ( a.ieta(), a.iphi(), a.depth() );
		  ArborIntegralHits.push_back(hitID);	
	  }
  }

  ArborBranch = Arbor(ArborIntegralHits, 3, 1);
  int NBranch = ArborBranch.size();
  int BranchSize = 0; 
  int currhitindex = 0;
  std::vector<int> TouchedHits;
  TouchedHits.clear();

  for(int j = 0; j < int(hits->size()); ++j)
  {

	  if( find(TouchedHits.begin(), TouchedHits.end(), j) == TouchedHits.end() )
	  {

		  reco::PFCluster c(PFLayer::HCAL_BARREL1,hits->at(j).energy(),hits->at(j).position().x(),
				  hits->at(j).position().y(),hits->at(j).position().z());

		  for(int k = 0; k < NBranch; k++)
		  {	
			  branch a_bran = ArborBranch[k];
			  if(find(a_bran.begin(), a_bran.end(), j) != a_bran.end() )
			  {
				  BranchSize = a_bran.size();
				  for(int k0 = 0; k0 < BranchSize; k0++)
				  {	
					  currhitindex = a_bran[k0];
					  if( currhitindex != j)
					  {
						  reco::PFRecHitRef hitRef(hits,currhitindex);
						  reco::PFRecHitFraction fraction(hitRef,1.0);
						  c.addRecHitFraction(fraction);
						  TouchedHits.push_back( currhitindex );

						  cout<<j<<"th hit with link to "<<currhitindex<<endl; 

					  }
				  }
			  }
		  }

		  /*
		     reco::PFRecHitRef hitRef(hits,j);
		     reco::PFRecHitFraction fraction(hitRef,1.0);
		     c.addRecHitFraction(fraction);
		   */

		  out->push_back(c);
		  TouchedHits.push_back(j);
	  }
  }

  //  ArborBranch = Arbor(inputArborHits, cellsize, layerthickness);


  iEvent.put( out,"allLinks");

}

PFArborLinker::~PFArborLinker() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFArborLinker::beginRun(const edm::Run& run,
		const EventSetup& es) {


}


