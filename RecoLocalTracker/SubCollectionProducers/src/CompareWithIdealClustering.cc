#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelArrayBuffer.h"
#include <stack>
using namespace std;

class CompareWithIdealClustering : public edm::EDProducer 
{

	public:
		CompareWithIdealClustering(const edm::ParameterSet& iConfig) ;
		~CompareWithIdealClustering() ;
		void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

		int AllEvents_countMatching002;
		int AllEvents_countMatching005;
		int AllEvents_countMatching01;
		int AllEvents_countMatching015;
		int AllEvents_countMatching02;
		int AllEvents_countNoMatching;
		int AllEvents_totalSplitted;
		int AllEvents_totalIdeal;
		int AllEvents_totalMatched;
		
		float AllEvents_totalSquareddistanceValue;

	private:
		std::string pixelCPE_; 
		edm::InputTag vertices_;
		edm::InputTag pixelClusters_;

};

CompareWithIdealClustering::CompareWithIdealClustering(const edm::ParameterSet& iConfig):
	pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
	vertices_(iConfig.getParameter<edm::InputTag>("vertices")),
	pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters"))
{

	AllEvents_countMatching002=0;
	AllEvents_countMatching005=0;
	AllEvents_countMatching01=0;
	AllEvents_countMatching015=0;
	AllEvents_countMatching02=0;
	AllEvents_countNoMatching=0;
	AllEvents_totalSquareddistanceValue=0;
	AllEvents_totalSplitted=0;
	AllEvents_totalIdeal=0;
	AllEvents_totalMatched=0;
		
	produces< edmNew::DetSetVector<SiPixelCluster> >();

}



CompareWithIdealClustering::~CompareWithIdealClustering()
{	
}

void CompareWithIdealClustering::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	edm::ESHandle<GlobalTrackingGeometry> geometry;
	iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);


	Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
	iEvent.getByLabel(pixelClusters_, inputPixelClusters);
	Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClustersIDEAL;
	iEvent.getByLabel("IdealsiPixelClusters", inputPixelClustersIDEAL);


	Handle<std::vector<reco::Vertex> > vertices; 
	iEvent.getByLabel(vertices_, vertices);
	const reco::Vertex & pv = (*vertices)[0];
	Handle<std::vector<reco::CaloJet> > jets;
	iEvent.getByLabel("ak5CaloJets", jets);
	

	edm::ESHandle<PixelClusterParameterEstimator> pe; 
	const PixelClusterParameterEstimator * pp ;
	iSetup.get<TkPixelCPERecord>().get(pixelCPE_ , pe );  
	pp = pe.product();

	std::auto_ptr<edmNew::DetSetVector<SiPixelCluster> > output(new edmNew::DetSetVector<SiPixelCluster>());



	edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt=inputPixelClustersIDEAL->begin();
	for(;detIt!=inputPixelClustersIDEAL->end();detIt++)
	{

		int countMatching002=0;
		int countMatching005=0;
		int countMatching01=0;
		int countMatching015=0;
		int countMatching02=0;
		int countNoMatching=0;

//		int countMatchingIdeal002=0;
//		int countMatchingIdeal005=0;
//		int countMatchingIdeal01=0;
//		int countMatchingIdeal015=0;
//		int countMatchingIdeal02=0;
//		int countNoMatchingIdeal=0;
	
		int countClusterIdeal=0;
		int countClusterSplitted=0;

		float totalSquareddistanceValue=0;
	
		const edmNew::DetSet<SiPixelCluster> & detset= *detIt;
		const GeomDet *det = geometry->idToDet( detset.id() );

		std::multimap<float,std::pair<int,int> > distance; 
		std::map<int,bool> usedSplit;
		std::map<int,bool> usedIdeal;

		for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=detset.begin(); cluster!=detset.end(); cluster++)
		{
			const SiPixelCluster & aIdealCluster =  *cluster;
			GlobalPoint posIdealCluster = det->surface().toGlobal(pp->localParametersV( aIdealCluster,( *geometry->idToDetUnit(detIt->id())))[0].first) ;
			GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
			GlobalVector clusterDir = posIdealCluster -ppv;			
			bool found=false;
			GlobalVector jetDir;
			for(std::vector<reco::CaloJet>::const_iterator jit = jets->begin() ; jit != jets->end() && !found ; jit++)
			{
				if(jit->pt() > 100)
				{				
//					float jetZOverRho = jit->momentum().Z()/jit->momentum().Rho();
					jetDir = GlobalVector(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());
					if(Geom::deltaR(jetDir,clusterDir) < 0.05   ) 
					{
						found=true;
					}
				}
			}
			
			if(found && aIdealCluster.sizeX() < 4)
			{
				SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClusters->find(detIt->id());
				const GeomDet *detBis = geometry->idToDet( myDet->id() );
				for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++ )
				{
					const SiPixelCluster & aCluster =  *clusterIt;
					GlobalPoint posCluster = detBis->surface().toGlobal(pp->localParametersV( aCluster,( *geometry->idToDetUnit(myDet->id())))[0].first) ;
				
					float distanceValue = (posCluster - posIdealCluster).mag();
					std::pair<int,int> myPair = std::pair<int,int>(cluster-detset.begin(),clusterIt-myDet->begin()); 
					//cluster = ideal, clusterIt = splitted
					
					GlobalVector clusterDir = posCluster -ppv;			
//					bool found=false;
					if(Geom::deltaR(jetDir,clusterDir) < 0.05  && aCluster.sizeX() < 4 ) 
					distance.insert(pair<float,std::pair<int,int> >(distanceValue,myPair));
				}
				}
		}
		
		for (std::multimap<float,std::pair<int,int> >::iterator it=distance.begin(); it!=distance.end(); ++it)
    		{
    			if(!usedSplit[it->second.second] && !usedIdeal[it->second.first])
    			{
	    			usedSplit[it->second.second]=true;
	    			usedIdeal[it->second.first]=true;
	    			float mindistanceValue=it->first;
				if(mindistanceValue<0.002) {countMatching002++;}
				else if(mindistanceValue<0.005)  {countMatching005++;}
				else if(mindistanceValue<0.01)  {countMatching01++;}
				else if(mindistanceValue<0.015)  {countMatching015++;}
				else if(mindistanceValue<0.02) {countMatching02++;} // 0.015 cm = 10 x 150 µm 
				else {countNoMatching++; totalSquareddistanceValue = totalSquareddistanceValue + 0.015*0.015;}

				AllEvents_totalSquareddistanceValue+=mindistanceValue*mindistanceValue;
        		}
        		
        		countClusterIdeal=usedIdeal.size();
        		countClusterSplitted=usedSplit.size();
        		
    		}    				

		int totalMatched=countClusterIdeal;
		if(countClusterIdeal!=countClusterSplitted) cout<<"\n**************BUG************\n";
		
		for (std::multimap<float,std::pair<int,int> >::iterator it=distance.begin(); it!=distance.end(); ++it)
    		{
    			if(!usedIdeal[it->second.first])
    			{
	    			usedIdeal[it->second.first]=true;
        		}
    			if(!usedSplit[it->second.second])
    			{
	    			usedSplit[it->second.second]=true;
			}        		        		
    		}    				
    		countClusterIdeal=usedIdeal.size();
        	countClusterSplitted=usedSplit.size();


		cout<<"\n";
		int totalSplitted=countClusterSplitted;
		int totalIdeal=countClusterIdeal;

		AllEvents_countMatching002+=countMatching002;
		AllEvents_countMatching005+=countMatching005;
		AllEvents_countMatching01+=countMatching01;
		AllEvents_countMatching015+=countMatching015;
		AllEvents_countMatching02+=countMatching02;
		AllEvents_countNoMatching+=countNoMatching;
		AllEvents_totalSquareddistanceValue+=totalSquareddistanceValue;
		AllEvents_totalSplitted+=totalSplitted;
		AllEvents_totalIdeal+=totalIdeal;
		AllEvents_totalMatched+=totalMatched;

		if(totalIdeal==0) continue;
		cout<<"\n Final comparer";
		cout<<"\n Final comparer: current detId= "<<detset.id();
		cout<<"\n Final comparer: Clusters Ideal= " <<AllEvents_totalIdeal;
		cout<<"\n Final comparer: Clusters Splitted= " <<AllEvents_totalSplitted;
		cout<<"\n Final comparer: Clusters Matched= " <<AllEvents_totalMatched;
		cout<<"\n Final comparer: matching (<0.002cm)\t= % "<<(float)AllEvents_countMatching002*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.005cm)\t= % "<<(float)AllEvents_countMatching005*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.01cm)\t= % "<<(float)AllEvents_countMatching01*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.015cm)\t= % "<<(float)AllEvents_countMatching015*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.02cm)\t= % "<<(float)AllEvents_countMatching02*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: bad matching (>0.02cm)\t= % "<<(float)AllEvents_countNoMatching*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: ideal not matched (lost)\t= % "<<(float)(AllEvents_totalIdeal - AllEvents_totalMatched )*100 /AllEvents_totalIdeal;
		cout<<"\n Final comparer: splitted not matched (fake)\t= % "<<(float)(AllEvents_totalSplitted - AllEvents_totalMatched )*100 /AllEvents_totalSplitted;
		cout<<"\n Final comparer: squared mean error = (um) "<<(float)10000.*sqrt(AllEvents_totalSquareddistanceValue) /AllEvents_totalMatched;
		
		cout<<"\n";

	}
	iEvent.put(output);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CompareWithIdealClustering);
