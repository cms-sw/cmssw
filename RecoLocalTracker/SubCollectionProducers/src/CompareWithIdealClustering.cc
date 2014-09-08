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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2.h"

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
		//int AllEvents_maxstepZero;

		float AllEvents_totalSquareddistanceValue;

		bool debug;

	private:
		std::string pixelCPE_; 
		edm::InputTag vertices_;
		edm::InputTag pixelClusters_;
		TH1D * AllDistances;
		TH1D * AllDistancesOnePixel;
		TH1D * AllDistancesNoOnePixel;
		TH1D * AllDistancesTwoPixel;
		TH1D * AllDistancesNoTwoPixel;
		TH1D * distances;
		TH1D * distances2;
		TH1D * Landau;
		TH2D * andamentoSizeYIdeale;
		TH2D * andamentoSizeYMisurato;
};

CompareWithIdealClustering::CompareWithIdealClustering(const edm::ParameterSet& iConfig):
	pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
	vertices_(iConfig.getParameter<edm::InputTag>("vertices")),
	pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters"))
{
	edm::Service<TFileService> fs;
	AllDistances = fs->make<TH1D>("AllDistances" , "tutte le distanze tra i cluster ideali e quelli dopo lo split" , 1000 , 0 , 400);
	AllDistancesOnePixel = fs->make<TH1D>("AllDistancesOnePixel" , "tutte le distanze dei cluster con un pixel" , 1000 , 0 , 400);
	AllDistancesNoOnePixel = fs->make<TH1D>("AllDistancesNoOnePixel" , "tutte le distanze dei cluster non con un pixel" , 1000 , 0 , 400);
	AllDistancesTwoPixel = fs->make<TH1D>("AllDistancesTwoPixel" , "tutte le distanze dei cluster con due pixel" , 1000 , 0 , 400);
	AllDistancesNoTwoPixel = fs->make<TH1D>("AllDistancesNoTwoPixel" , "tutte le distanze dei cluster con più di due pixel" , 1000 , 0 , 400);
	distances = fs->make<TH1D>("distances" , "distanze" , 1000 , 0 , 200);
	distances2 = fs->make<TH1D>("shorterDistances" , "distanze vicine a zero" , 1000 , -0.01 , 0.1);
	Landau = fs->make<TH1D>("Landau" , "Landau" , 500 , 0 , 200000 );
	andamentoSizeYIdeale= fs->make<TH2D>("IdealSizeY" , "SizeY(y) Vs jetZOverRho(x) ideal" , 500 , 0 , 20 , 500 , 0 , 20 );
	andamentoSizeYMisurato= fs->make<TH2D>("AfterSplitSizeY" , "SizeY(y) Vs jetZOverRho(x) dopo lo Split" , 500 , 0 , 20 , 500 , 0 , 20 );

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
	//AllEvents_maxstepZero=0;
		
	produces< edmNew::DetSetVector<SiPixelCluster> >();

	debug = false;

}



CompareWithIdealClustering::~CompareWithIdealClustering()
{	
}

void CompareWithIdealClustering::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	cout<< setprecision(6); 
	using namespace edm;
	edm::ESHandle<GlobalTrackingGeometry> geometry;
	iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);


	Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClusters;
	iEvent.getByLabel(pixelClusters_, inputPixelClusters);
	Handle<edmNew::DetSetVector<SiPixelCluster> > inputPixelClustersIDEAL;
	iEvent.getByLabel("IdealsiPixelClusters", inputPixelClustersIDEAL);

	/*Handle<TrackCollection> tracks;
	iEvent.getByLabel("IdealTrackTags",tracks);*/

	/*Handle<int> inputmaxstep;
	iEvent.getByLabel( pixelClusters_, inputmaxstep);
	AllEvents_maxstepZero = *inputmaxstep;

	Handle<int> inputmaxstep;
	iEvent.getByLabel( pixelClusters_, inputmaxstep);
	AllEvents_maxstepZero = *inputmaxstep;

	Handle<int> inputmaxstep;
	iEvent.getByLabel( pixelClusters_, inputmaxstep);
	AllEvents_maxstepZero = *inputmaxstep;

	Handle<int> inputmaxstep;
	iEvent.getByLabel( pixelClusters_Zero_, inputmaxstep);
	AllEvents_maxstepZero = *inputmaxstep;*/



	Handle<std::vector<reco::Vertex> > vertices; 
	iEvent.getByLabel(vertices_, vertices);
	const reco::Vertex & pv = (*vertices)[0];               //LO 0 È IL VERTICE PRIMARIO?
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
	

		int idealNotCounted = 0;
		int splittedNotCounted = 0;

		int idealCounted = 0;
		int splittedCounted = 0;

		float totalSquareddistanceValue=0;
		int daAggiungere=0;
	
		const edmNew::DetSet<SiPixelCluster> & detset= *detIt;
		const GeomDet *det = geometry->idToDet( detset.id() );

		std::multimap<float,std::pair<int,int> > distance; 
		std::map<int,bool> usedSplit;
		std::map<int,bool> usedIdeal;
		//unsigned int SizeY =0;

		float jetZOverRho =0;

		for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=detset.begin(); cluster!=detset.end(); cluster++)
		{
			const SiPixelCluster & aIdealCluster =  *cluster;
			GlobalPoint posIdealCluster = det->surface().toGlobal(pp->localParametersV( aIdealCluster,( *geometry->idToDetUnit(detIt->id())))[0].first) ;
			GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
			GlobalVector idealClusterDir = posIdealCluster -ppv;			
			bool found=false;
			GlobalVector jetDir;
			for(std::vector<reco::CaloJet>::const_iterator jit = jets->begin() ; jit != jets->end() && !found ; jit++)
			{
				if(jit->pt() > 100)
				{		
					jetZOverRho = jit->momentum().Z()/jit->momentum().Rho();
					if(fabs(posIdealCluster.z())>30) jetZOverRho=jit->momentum().Rho()/jit->momentum().Z();
							
//					float jetZOverRho = jit->momentum().Z()/jit->momentum().Rho();
					jetDir = GlobalVector(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());
					if(Geom::deltaR(jetDir,idealClusterDir) < 0.05   ) 
					{
						found=true;
					andamentoSizeYIdeale->Fill(1.9*jetZOverRho , aIdealCluster.sizeY() );
					}
//SizeY = sqrt(1.3*1.3+1.9*1.9*jit->momentum().Rho()/jit->momentum().Z()*jit->momentum().Rho()/jit->momentum().Z());
				}
			}
			
			if(found  && aIdealCluster.sizeX() < 4 )
			{
				SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClusters->find(detIt->id());
				const GeomDet *detBis = geometry->idToDet( myDet->id() );
				bool atleasteone = false;
				for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++ )
				{
					const SiPixelCluster & aCluster =  *clusterIt;
					
					GlobalPoint posCluster = detBis->surface().toGlobal(pp->localParametersV( aCluster,( *geometry->idToDetUnit(myDet->id())))[0].first) ;
				
					float distanceValue = (posCluster - posIdealCluster).mag();
					if(isnan(distanceValue))
					{ 
						cout <<  "(posCluster - posIdealCluster).mag()=" << (posCluster - posIdealCluster).mag() << " posCluster.mag()" << posCluster.mag() << " posIdealCluster.mag()" << posIdealCluster.mag() << endl ; 
					} 
					std::pair<int,int> myPair = std::pair<int,int>(cluster-detset.begin(),clusterIt-myDet->begin()); 
					//cluster = ideal, clusterIt = splitted
					
					GlobalVector splittedClusterDir = posCluster -ppv;			
//					bool found=false;


					if(Geom::deltaR(jetDir,splittedClusterDir) < 0.05  && aCluster.sizeX() < 4   )
					{
						distance.insert(pair<float,std::pair<int,int> >(distanceValue,myPair));
						atleasteone=true;
						if(10000*distanceValue>0.05) {
						AllDistances->Fill (10000*distanceValue);
						if(aCluster.sizeX() < 2 && aCluster.sizeY() < 2 ) AllDistancesOnePixel->Fill (10000*distanceValue);
						else {
						AllDistancesNoOnePixel->Fill (10000*distanceValue);
					if((aCluster.sizeX()<3 && aCluster.sizeY()<2) || (aCluster.sizeX()<2 && aCluster.sizeY()<3))
						AllDistancesTwoPixel->Fill (10000*distanceValue);
					else AllDistancesNoTwoPixel->Fill (10000*distanceValue);
						}
						andamentoSizeYMisurato->Fill(1.9*jetZOverRho , aCluster.sizeY() );
						}
					}
				}
				if(!atleasteone)  daAggiungere++;
				
			
			}

		if(!found && aIdealCluster.sizeX() < 3 && aIdealCluster.sizeY() < 2) Landau->Fill(cluster->charge());

		}


		int NumberOfIterations = 0;

		for (std::multimap<float,std::pair<int,int> >::iterator it=distance.begin(); it!=distance.end(); ++it)
    		{
    			float mindistanceValue=it->first;
    			if(!usedSplit[it->second.second] && !usedIdeal[it->second.first])
    			{
				
				NumberOfIterations++;
	    			usedSplit[it->second.second]=true;
	    			usedIdeal[it->second.first]=true;

				if(10000*mindistanceValue>0.05) distances->Fill (10000*mindistanceValue);
				if(10000.*mindistanceValue<0.1 && 10000.*mindistanceValue!=0. ) distances2->Fill (10000*mindistanceValue);
				
				if(mindistanceValue<0.002) {countMatching002++;}
				else if(mindistanceValue<0.005)  {countMatching005++;}
				else if(mindistanceValue<0.01)  {countMatching01++;}
				else if(mindistanceValue<0.015)  {countMatching015++;}
				else if(mindistanceValue<0.02) {countMatching02++;} // 0.015 cm = 10 x 150 µm 
				else {countNoMatching++; totalSquareddistanceValue = totalSquareddistanceValue + 0.015*0.015;}

        		}

    		}    				



		
		for (std::multimap<float,std::pair<int,int> >::iterator it=distance.begin(); it!=distance.end(); ++it)
    		{
//		cout << "("<< it->second.first << ", " << it->second.second << ") distanza = " << it->first << endl;
    			if(!usedIdeal[it->second.first])
    			{
	    			usedIdeal[it->second.first]=true;
				idealNotCounted++;
        		}
    			if(!usedSplit[it->second.second])
    			{
	    			usedSplit[it->second.second]=true;
				splittedNotCounted++;
			}        

		}


		idealCounted = NumberOfIterations;
		splittedCounted = NumberOfIterations;

		int totalSplitted = splittedCounted + splittedNotCounted;
		int totalIdeal = idealCounted + idealNotCounted+ daAggiungere;
		int totalMatched = NumberOfIterations;

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
		cout<< setprecision(6) <<"\n Final comparer";
		cout<<"\n Final comparer: current detId= "<<detset.id();
		cout<<"\n Final comparer: Clusters Ideal= " <<AllEvents_totalIdeal;
		cout<<"\n Final comparer: Clusters Splitted= " <<AllEvents_totalSplitted;
		cout<<"\n Final comparer: Clusters Matched= " <<AllEvents_totalMatched;
		cout<<"\n Final comparer: matching (<0.002cm)\t= % "<<(float)AllEvents_countMatching002*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.005cm)\t= % "<<(float)AllEvents_countMatching005*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.01cm)\t= % "<<(float)AllEvents_countMatching01*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.015cm)\t= % "<<(float)AllEvents_countMatching015*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: matching (<0.02cm)\t= % "<<(float)AllEvents_countMatching02*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: bad matching (>0.02cm)\t= % "<<(float)AllEvents_countNoMatching*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: ideal not matched (lost)\t= % "<<(float)(AllEvents_totalIdeal - AllEvents_totalMatched )*100. /AllEvents_totalIdeal;
		cout<<"\n Final comparer: splitted not matched (fake)\t= % "<<(float)(AllEvents_totalSplitted - AllEvents_totalMatched )*100 /AllEvents_totalIdeal;
		//cout<<"\n Final comparer: maxstep zero \t= % "<< (float)(AllEvents_maxstepZero )*100 /AllEvents_totalSplitted;
		cout<<"\n Final comparer: squared mean error = (um) "<<(float)10000.*sqrt(AllEvents_totalSquareddistanceValue) /AllEvents_totalMatched;
		
		cout<<"\n";
		if (AllEvents_totalSplitted - AllEvents_totalMatched != 0 && !debug ) {cout << "Questo Evento \n"; debug = true;}
	}
	iEvent.put(output);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CompareWithIdealClustering);
