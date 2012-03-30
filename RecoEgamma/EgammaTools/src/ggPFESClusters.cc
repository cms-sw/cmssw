#include "RecoEgamma/EgammaTools/interface/ggPFESClusters.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "TMath.h"
using namespace edm;
using namespace std;
using namespace reco;

ggPFESClusters::ggPFESClusters(
			       edm::Handle<EcalRecHitCollection>& ESRecHits
			       ):
  ESRecHits_(ESRecHits)
{
  
}

ggPFESClusters::~ggPFESClusters(){

}

vector<reco::PreshowerCluster>ggPFESClusters::getPFESClusters(
							      reco::SuperCluster sc 
							      ){
  // cout<<"SC Eta "<<sc.eta()<<endl;
  std::vector<PreshowerCluster>PFPreShowerClust;
  for(reco::CaloCluster_iterator ps=sc.preshowerClustersBegin(); ps!=sc.preshowerClustersEnd(); ++ps){
    std::vector< std::pair<DetId, float> > psCells=(*ps)->hitsAndFractions();
    float PS1E=0;
    float PS2E=0;	
    int plane=0;
    for(unsigned int s=0; s<psCells.size(); ++s){
      for(EcalRecHitCollection::const_iterator es=ESRecHits_->begin(); es!= ESRecHits_->end(); ++es){
	if(es->id().rawId()==psCells[s].first.rawId()){ 
	  ESDetId id=ESDetId(psCells[s].first.rawId());
	  plane=id.plane();
	  if(id.plane()==1)PS1E=PS1E + es->energy() * psCells[s].second;
	  if(id.plane()==2)PS2E=PS2E + es->energy() * psCells[s].second;
	  break;
	}		  
      }
    }
    //make PreShower object storing plane PSEnergy and Position
    if(plane==1){
      PreshowerCluster PS1=PreshowerCluster(PS1E,(*ps)->position(),(*ps)->hitsAndFractions(),plane);
      PFPreShowerClust.push_back(PS1);	
    }
    if(plane==2){
      PreshowerCluster PS2=PreshowerCluster(PS2E,(*ps)->position(),(*ps)->hitsAndFractions(),plane);
      PFPreShowerClust.push_back(PS2);      
    }
  }  
  return PFPreShowerClust;  
}
