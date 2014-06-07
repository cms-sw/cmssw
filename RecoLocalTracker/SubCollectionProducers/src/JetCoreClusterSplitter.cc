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
#include <time.h>
#include <algorithm> 

using namespace std;
	template < typename T >
std::ostream& bin(T& value, std::ostream &o)
{
	char c='A';
	for ( T bit = 16; bit; bit >>= 1,c++)
	{
		o << ( ( value & bit ) ? c : '.' );
	}
	return o;
}


const int BinsJetOverRho=21;
const int BinsXposition=5;
const int BinsDirections=4;
const int BinsX=20;
const int BinsY=20;
const float jetZOverRhoWidth=0.5;


class JetCoreClusterSplitter : public edm::EDProducer 
{

	public:
		JetCoreClusterSplitter(const edm::ParameterSet& iConfig) ;
		~JetCoreClusterSplitter() ;
		void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

	private:
		bool split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY,float jetZOverRho, const edmNew::DetSet<SiPixelCluster> & );
		float distanceCluster(const SiPixelCluster & cluster,const edmNew::DetSet<SiPixelCluster> & idealClusters);
		void print(const SiPixelArrayBuffer & b, const SiPixelCluster & aCluster, int div=1000 );
		std::vector<SiPixelCluster> fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY,float jetZOverRho);
		bool nextCombination(std::vector<int> & comb,int npos);
		float combinations(float npos, float expectedClusters);
		float pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction,int bintheta);
		float pixelWeight2(int clx, int cly, int x, int y,int sizeY, int direction);
//		void initCharge();
	        bool verbose;
		std::string pixelCPE_; 
		edm::InputTag pixelClusters_;
		edm::InputTag vertices_;
		int mapcharge[21][5][3][20][20];
		int count[21][5][3];
		int totalcharge[21][5][3];
		int nDirections;

};

JetCoreClusterSplitter::JetCoreClusterSplitter(const edm::ParameterSet& iConfig):
	verbose(iConfig.getParameter<bool>("verbose")),
	pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
	pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters")),
	vertices_(iConfig.getParameter<edm::InputTag>("vertices"))

{
	nDirections=4;
	
	for(int a=0;a<21;a++)
	for(int b=0;b<5;b++)
	for(int e=0;e<3;e++){
	count[a][b][e]=0;
	totalcharge[a][b][e]=0;
	for(int c=0;c<20;c++)
	for(int d=0;d<20;d++)
	  mapcharge[a][b][e][c][d]=0;
	}
//	initCharge();
	produces< edmNew::DetSetVector<SiPixelCluster> >();

}



JetCoreClusterSplitter::~JetCoreClusterSplitter()
{
}

bool SortPixels (const SiPixelCluster::Pixel& i,const SiPixelCluster::Pixel& j) { return (i.adc>j.adc); }

void JetCoreClusterSplitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace edm;
	edm::ESHandle<GlobalTrackingGeometry> geometry;
	iSetup.get<GlobalTrackingGeometryRecord>().get(geometry);
	/*edm::ESHandle<TrackerGeometry> tracker;
	  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
	  const TrackerGeometry * trackerGeometry = tracker.product();*/

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

	edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt=inputPixelClusters->begin();
	for(;detIt!=inputPixelClusters->end();detIt++)
	{
		edmNew::DetSetVector<SiPixelCluster>::FastFiller filler(*output,detIt->id());
		const edmNew::DetSet<SiPixelCluster> & detset= *detIt;
		const GeomDet *det = geometry->idToDet( detset.id() );
		for(edmNew::DetSet<SiPixelCluster>::const_iterator cluster=detset.begin(); cluster!=detset.end(); cluster++)
		{
			const SiPixelCluster & aCluster =  *cluster;
			bool hasBeenSplit = false;
			GlobalPoint cPos = det->surface().toGlobal(pp->localParametersV( aCluster,( *geometry->idToDetUnit(detIt->id())))[0].first) ;
			GlobalPoint ppv(pv.position().x(),pv.position().y(),pv.position().z());
			GlobalVector clusterDir = cPos -ppv;
			for(std::vector<reco::CaloJet>::const_iterator jit = jets->begin() ; jit != jets->end() ; jit++)
			{
				if(jit->pt() > 100)
				{
					float jetZOverRho = jit->momentum().Z()/jit->momentum().Rho();
					if(fabs(cPos.z())>30) jetZOverRho=jit->momentum().Rho()/jit->momentum().Z();
					GlobalVector jetDir(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());
					unsigned int maxSizeY=fabs(sqrt(1.3*1.3+1.9*1.9*jetZOverRho*jetZOverRho));
//					unsigned int maxSizeY=fabs(jetZOverRho*1.9);
					
//					unsigned int maxSizeY=fabs(jetZOverRho*1.75)+0.5;

					if(maxSizeY < 1) maxSizeY=1;
					if(Geom::deltaR(jetDir,clusterDir) < 0.05 && aCluster.charge() > 30000 && (aCluster.sizeX() > 2 || ((unsigned int)aCluster.sizeY()) > maxSizeY+1) )
					{
						if(verbose) std::cout << "CHECK FOR NEW SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						if(verbose)std::cout << "jetZOverRho="<< jetZOverRho << std::endl;	
						SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClustersIDEAL->find(detIt->id());
						clock_t init=clock(), final;
						const edmNew::DetSet<SiPixelCluster> & idealClusters  = (*myDet);
						if(split(aCluster,filler,sqrt(1.08+jetZOverRho*jetZOverRho)*26000,maxSizeY,jetZOverRho,idealClusters)) {hasBeenSplit=true;
						 final=clock()-init; 
						 if(verbose)cout<<"Time used: (s) " << (double)final / ((double)CLOCKS_PER_SEC)<<endl;
						}
						if(verbose)std::cout << "IDEAL was : "  << std::endl; 
						int xmin=aCluster.minPixelRow();                        
						int ymin=aCluster.minPixelCol();                                
						int xmax=aCluster.maxPixelRow();                                
						int ymax=aCluster.maxPixelCol(); 
						int last=1;
						std::map<int,int> sh;  
						for(int x=xmin; x<= xmax;x++){                                          
							for(int y=ymin; y<= ymax;y++)                                   
							{                                                                       
								int h=0;
								int flag=0;                                           
								for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++,h++)
								{

									std::vector<SiPixelCluster::Pixel> pixels = clusterIt->pixels();
									for(unsigned int j = 0; j < pixels.size(); j++)
									{                               
										if(pixels[j].x==x && pixels[j].y==y){
										 if(!sh[h]) {sh[h]=last; last++; }
										 flag|=(1<<(sh[h]-1));
										}
									}                               
								}                                       

								if(verbose) std::cout << " " ;  
								if(verbose) bin( flag,std::cout) ; 
								// std::setiosflags(std::ios::fixed)
								//                                << std::setprecision(0)
								//                              << std::setw(7)
								//                            << std::left ; bin( flag,std::cout);
								//                                << std::left << hex << flag;


							}
							if(verbose) std::cout << std::endl;         
						}
						int h=0;
						for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++,h++)
                                                                {
									if(sh[h] && verbose) std::cout << "IDEAL POS: " << h << " x: "  << std::setprecision(2) << clusterIt->x() << " y: " << clusterIt->y() << " c: " << clusterIt->charge() << std::endl;
                                                                }
                       






					}

				}
			}
			if(!hasBeenSplit)
			{
				//blowup the error
				SiPixelCluster c=aCluster;
//				c.setSplitClusterErrorX(c.sizeX()*100./3.);
//				c.setSplitClusterErrorY(c.sizeY()*150./3.);
				 filler.push_back(c);	
			}

		}
	}
	iEvent.put(output);
}



float JetCoreClusterSplitter::distanceCluster(const SiPixelCluster & cluster,const edmNew::DetSet<SiPixelCluster> & idealClusters)
{
float minDistance=1e99;
for(edmNew::DetSet<SiPixelCluster>::const_iterator ideal=idealClusters.begin(); ideal <  idealClusters.end() ; ideal++)
{
	float distance = sqrt( (cluster.x()-ideal->x())*(cluster.x()-ideal->x())  +   (cluster.y()-ideal->y())*(cluster.y()-ideal->y())*1.5*1.5 );
	if(distance<minDistance) minDistance=distance;
}
return minDistance;
}

bool JetCoreClusterSplitter::split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,int sizeY,float jetZOverRho,const edmNew::DetSet<SiPixelCluster> & idealClusters)
{
	std::vector<SiPixelCluster> sp=fittingSplit(aCluster,expectedADC,sizeY,jetZOverRho);
	

	for(unsigned int i = 0; i < sp.size();i++ )
	{
		float distance = JetCoreClusterSplitter::distanceCluster(sp[i],idealClusters);
if(verbose)		std::cout << "NEW POS: " << i << " x: "  << std::setprecision(2) << sp[i].x() << " y: " << sp[i].y() << " c: " << sp[i].charge() << " distance=" <<  std::setprecision(6)<< 100*distance << " um"  << std::endl;
		filler.push_back(sp[i]);
	}




	int xmin=aCluster.minPixelRow();
	int ymin=aCluster.minPixelCol();
	int xmax=aCluster.maxPixelRow();
	int ymax=aCluster.maxPixelCol();
	if(verbose)std::cout << "Splitted clusters map:" << std::endl;
	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++)
		{
			int flag=0;
			for(unsigned int i = 0; i < sp.size();i++ ){

				std::vector<SiPixelCluster::Pixel> pixels = sp[i].pixels();
				for(unsigned int j = 0; j < pixels.size(); j++)
				{
					if(pixels[j].x==x && pixels[j].y==y) flag|=(1<<i);
				}	
			}	

			std::cout << " " ;  bin( flag,std::cout) ; 
			// std::setiosflags(std::ios::fixed)
			//                                << std::setprecision(0)
			//                              << std::setw(7)
			//                            << std::left ; bin( flag,std::cout);
			//                                << std::left << hex << flag;


		}
		std::cout << std::endl;
	}
	return (sp.size() > 0);
}

float JetCoreClusterSplitter::pixelWeight2(int clx, int cly, int x, int y,int sizeY,int direction)
{
 if (direction>1 || direction<0) return 0;
 float fact=0; 
                               if(x==clx &&  (y>=cly && y < cly+(sizeY+1)/2) ) fact=2;
                                if(x==clx+1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx &&  (y>= cly+(sizeY+1)/2)  && y < cly+sizeY ) fact=1;
                                if(x==clx+1 && direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=1;
                                if(x==clx-1 && ! direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=1;
                                if(x==clx+1 && direction && y==cly+sizeY ) fact=1;
                                if(x==clx-1 &&  ! direction && y==cly+sizeY ) fact=1;
                                if(x==clx && y==cly-1 ) fact=1;
return fact/(0.5+sizeY)/4.;
//return fact/(0.5+sizeY)/4.;
}

float JetCoreClusterSplitter::pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction, int)
{
 float fact=0;
                               if(x==clx &&  (y>=cly && y < y+sizeY) ) fact=8;
                                if(x==clx+1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=2;
                                if(x==clx+1 && !direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=1;
                                if(x==clx-1 && direction  &&  (y>=cly+(sizeY+1)/2) && y < cly+sizeY ) fact=1;
//                              if(x==clx &&  (y>= cly+(sizeY+1)/2)  && y < cly+sizeY ) fact=1;
                                if(x==clx+1 && direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=2;
                                if(x==clx-1 && ! direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=2;
                                if(x==clx+1 && !direction  &&  (y>=cly && y<cly+(sizeY+1)/2)  ) fact=1;
                                if(x==clx-1 &&  direction  &&  (y>=cly && y<cly+(sizeY+1)/2) ) fact=1;
//                                if(x==clx+1 && direction && y==cly+sizeY ) fact=1;
                                if(x==clx-1 &&  ! direction && y==cly+sizeY ) fact=1;
                                if(x==clx && y==cly-1 ) fact=1;
return fact/(11.*sizeY+2);
}
/*float JetCoreClusterSplitter::pixelWeight(int clx, int cly, int x, int y,int sizeY,int direction,int bintheta)
{
 
 
 if(x-clx+10<-20) return 0;
 if(y-cly+(sizeY+1)/2<0) return 0;
 if(x-clx+10>=20) return 0;
 if(y-cly+(sizeY+1)/2>=20) return 0;

//if(direction>2) {cout<<"*** BUG direction>2 *****"; return 0;}
//if(direction<0) {cout<<"*** BUG direction<0 *****"; return 0;}

if(bintheta<0) {cout<<"Forced bintheta=0. It was "<<bintheta; bintheta=0;}
if(bintheta>20) {cout<<"Forced bintheta=20. It was "<<bintheta; bintheta=20;}

int caseX=direction/2;
direction=direction%2;

direction=direction+1;

// if(x-clx<=-10) x=clx-10;
// if(y-cly<=0) y=cly;
// if(x-clx>10) x=clx+9;
// if(y-cly>20) y=cly+19;

// if(x-clx<-10) return 0;
// if(y-cly<0) return 0;
// if(x-clx>=10) return 0;
// if(y-cly>20) return 0;
 unsigned int binX = clx*5./160;
// int mapcharge[21][5][3][20][20];
sizeY=sizeY+(direction-1);
 float fact=1.*mapcharge[bintheta][binX][direction][x-clx+10+caseX][y-cly+(sizeY-1)/2]/totalcharge[bintheta][binX][direction]*count[bintheta][binX][direction];
// float fact=1.*mapcharge[bintheta][binX][direction][x-clx+10][y-cly]/totalcharge[bintheta][binX][direction]*count[bintheta][binX][direction];
 //std::cout << "bin " << bintheta <<  ", " << binX  <<  ", " << x-clx+10  <<  ", " << y-cly+10 << " map " << mapcharge[bintheta][binX][x-clx+10][y-cly+10] << " tot " << totalcharge[bintheta][binX] << " fact " << fact << std::endl;  	 	
return fact;
}
*/

float  JetCoreClusterSplitter::combinations(float npos, float expectedClusters){
		// combination with repetition ( n+k-1  over k )
                float up=npos+expectedClusters-1;
                float down=expectedClusters;
		float fdown=1,prod=1;
		for(unsigned int i=npos; i <= up; i++)  prod*=i;
              //float fup=1,fdown=1,fup_down=1;
          //    for(unsigned int i=1; i <= up; i++)  fup*=i;
            //  for(unsigned int i=1; i <= up-down; i++)  fup_down*=i;

                for(unsigned int i=1; i <= down; i++)  fdown*=i;
                return prod/fdown;

}




std::vector<SiPixelCluster> JetCoreClusterSplitter::fittingSplit(const SiPixelCluster & aCluster, float expectedADC,int sizeY,float jetZOverRho)
{
//	bool verbose=false;
	int xmin=aCluster.minPixelRow();
	int ymin=aCluster.minPixelCol();
	int xmax=aCluster.maxPixelRow();
	int ymax=aCluster.maxPixelCol();
	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++)
		{
			int flag=0;
			std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
			for(unsigned int j = 0; j < pixels.size(); j++)
			{
				if(pixels[j].x==x && pixels[j].y==y) flag=j;
			}

			if(verbose)     std::cout << " "<<
				std::setiosflags(std::ios::fixed)
					<< std::setprecision(0)
					<< std::setw(7)
					<< std::left << flag ;


		}
		if(verbose)             std::cout << std::endl;
	}
	std::cout <<  std::setprecision(7) << "sizeY " << sizeY << std::endl; 	
	unsigned int meanExp = floor(aCluster.charge() / expectedADC +0.5) ;
	std::vector<SiPixelCluster> output;
	if(meanExp==0) {std::cout << "ZERO????" << std::endl;} 
	if(meanExp<=1) {
		output.push_back(aCluster);	
		return output;
	}	

	std::vector<float> clx(meanExp);
	std::vector<float> cly(meanExp);
	std::vector<float> cls(meanExp);
	std::vector<float> oldclx(meanExp);
	std::vector<float> oldcly(meanExp);
	std::vector<SiPixelCluster::Pixel> originalpixels = aCluster.pixels();
	std::vector<SiPixelCluster::Pixel> pixels;
	for(unsigned int j = 0; j < originalpixels.size(); j++)
	{
		int sub=originalpixels[j].adc/2000;
		if(sub < 1) sub=1;
		int perDiv=originalpixels[j].adc/sub;
if(verbose) 		std::cout << "Splitting  " << j << "  in [ " << pixels.size() << " , " << pixels.size()+sub << " ] "<< std::endl;
		for(int k=0;k<sub;k++)
		{
			pixels.push_back(SiPixelCluster::Pixel(originalpixels[j].x,originalpixels[j].y,perDiv));
		}
	}
	std::vector<int> clusterForPixel(pixels.size());
	//initial values
	for(unsigned int j = 0; j < meanExp; j++)
	{
		oldclx[j]=-999;
		oldcly[j]=-999;
		clx[j]=originalpixels[0].x+j;
		cly[j]=originalpixels[0].y+j;
		cls[j]=0;
	}
	bool stop=false;
	int maxsteps=100;
	while(!stop && maxsteps > 0){
		maxsteps--;
		//Compute all distances
		std::vector<std::vector<float> > distanceMapX( pixels.size(), vector<float>(meanExp));
		std::vector<std::vector<float> > distanceMapY( pixels.size(), vector<float>(meanExp));
		std::vector<std::vector<float> > distanceMap( pixels.size(), vector<float>(meanExp));
		for(unsigned int j = 0; j < pixels.size(); j++)
		{
if(verbose) 			std::cout << "pixel pos " << j << " " << pixels[j].x << " " << pixels[j].y << std::endl;
			for(unsigned int i = 0; i < meanExp; i++)
			{
				distanceMapX[j][i]=1.*pixels[j].x-clx[i];
				distanceMapY[j][i]=1.*pixels[j].y-cly[i];
			        float dist=0;
//				float sizeX=2;
/*		if(std::abs(distanceMapX[j][i])>sizeX/2.)
                                {
                                        dist+=(std::abs(distanceMapX[j][i])-sizeX/2.+1)*(std::abs(distanceMapX[j][i])-sizeX/2.+1);
                                } else {
                                        dist+=distanceMapX[j][i]/sizeX*2*distanceMapX[j][i]/sizeX*2;
                                }*/
				dist+=1.*distanceMapX[j][i]*distanceMapX[j][i];

                                if(std::abs(distanceMapY[j][i])>sizeY/2.)
                                {
                                        dist+=1.*(std::abs(distanceMapY[j][i])-sizeY/2.+1.)*(std::abs(distanceMapY[j][i])-sizeY/2.+1.);
                                } else {
                                        dist+=1.*distanceMapY[j][i]/sizeY*2.*distanceMapY[j][i]/sizeY*2.;
                                }
                                distanceMap[j][i]=sqrt(dist);
if(verbose) 				std::cout << "Cluster " << i << " Pixel " << j << " distances: " << distanceMapX[j][i] << " " << distanceMapY[j][i]  << " " << distanceMap[j][i] << std::endl;

			}

		}
		//Compute scores for sequential addition
		std::multimap<float,int> scores;
		
		for(unsigned int j = 0; j < pixels.size(); j++)
                {                       
			float minDist=9e99;
			float secondMinDist=9e99;
//			int pkey=-1;
//			int skey=-1;
                        for(unsigned int i = 0; i < meanExp; i++)
                        {
				float dist=distanceMap[j][i];
				if(dist < minDist) {
					secondMinDist=minDist;
//					skey=pkey;
					minDist=dist;
//					pkey=i;
				} else if(dist < secondMinDist) {
					secondMinDist=dist;
//					skey=i;
				}
			}
			scores.insert(std::pair<float,int>(-secondMinDist,j));
		}

		//Iterate starting from the ones with furthest second best clusters, i.e. easy choices
		std::vector<float> weightOfPixel(pixels.size());
		for(std::multimap<float,int>::iterator it=scores.begin(); it!=scores.end(); it++)
		{
			int j=it->second;
if(verbose)			std::cout << "Pixel " << j << " with score " << it->first << std::endl;
			//find cluster that is both close and has some charge still to assign
			float maxEst=0;
			int cl=-1;
                        for(unsigned int i = 0; i < meanExp; i++)
			{
			  float chi2=(cls[i]*cls[i]-expectedADC*expectedADC)/2./(expectedADC*0.2); //20% uncertainty? realistic from Landau?
			  float clQest=1./(1.+exp(chi2))+1e-6; //1./(1.+exp(x*x-3*3))	
			  float clDest=1./(distanceMap[j][i]+0.05);

if(verbose) 			  std::cout <<" Q: " <<  clQest << " D: " << clDest << " " << distanceMap[j][i] <<  std::endl;
			  float est=clQest*clDest;
			  if(est> maxEst) {
				cl=i;
				maxEst=est;
			  }	
			}
			cls[cl]+=pixels[j].adc;
			clusterForPixel[j]=cl;
			weightOfPixel[j]=maxEst;
if(verbose) 			std::cout << "Pixel j weight " << weightOfPixel[j] << " " << j << std::endl;
		}
		//Recompute cluster centers
		stop=true;
	        for(unsigned int i = 0; i < meanExp; i++){
			if(std::abs(clx[i]-oldclx[i]) > 0.01) stop = false; //still moving
			if(std::abs(cly[i]-oldcly[i]) > 0.01) stop = false;
			oldclx[i]=clx[i]; oldcly[i]=cly[i]; clx[i]=0; cly[i]=0; cls[i]=1e-99; 
		}
                for(unsigned int j = 0; j < pixels.size(); j++)
		{
			if(clusterForPixel[j]<0) continue;
if(verbose) 			std::cout << "x " << pixels[j].x <<" * " << pixels[j].adc << " * " << weightOfPixel[j]<<std::endl;
			clx[clusterForPixel[j]]+=pixels[j].x*pixels[j].adc;
			cly[clusterForPixel[j]]+=pixels[j].y*pixels[j].adc;
			cls[clusterForPixel[j]]+=pixels[j].adc;
//			std::cout << "update cluster " << clusterForPixel[j] << " x,y " << clx[clusterForPixel[j]] << " " << cly[clusterForPixel[j]] <<  "weight x,y " << cls[clusterForPixel[j]] <<  " " << clx[clusterForPixel[j]]/cls[clusterForPixel[j]] << " " << cly[clusterForPixel[j]]/cls[clusterForPixel[j]] <<std::endl;
			
		}
		for(unsigned int i = 0; i < meanExp; i++){
			if(cls[i]!=0){
			clx[i]/=cls[i];
			cly[i]/=cls[i];
			}
if(verbose) 			std::cout << "Center for cluster " << i << " x,y " << clx[i] << " " << cly[i] << std::endl;
			cls[i]=0;
		}
		

	}
	std::cout << "maxstep " << maxsteps << std::endl;
	//accumulate pixel with same cl
	std::vector<std::vector<SiPixelCluster::Pixel> > pixelsForCl(meanExp);
	for(int cl=0;cl<(int)meanExp;cl++){
		for(unsigned int j = 0; j < pixels.size(); j++) {
			if(clusterForPixel[j]==cl and pixels[j].adc!=0) {  //for each pixel of cluster cl find the other pixels with same x,y and accumulate+reset their adc
				for(unsigned int k = j+1; k < pixels.size(); k++)
				{
					if(pixels[k].adc!=0 and pixels[k].x==pixels[j].x and pixels[k].y==pixels[j].y)
					{
						pixels[j].adc+=pixels[k].adc;
						pixels[k].adc=0;
					}
				}
				pixelsForCl[cl].push_back(pixels[j]);
			}
		}
	}

	//	std::vector<std::vector<std::vector<SiPixelCluster::PixelPos *> > > pixelMap(meanExp,std::vector<std::vector<SiPixelCluster::PixelPos *> >(512,std::vector<SiPixelCluster::Pixel *>(512,0))); 	



	for(int cl=0;cl<(int)meanExp;cl++){
		SiPixelCluster * cluster=0;
		std::cout << "Pixels of cl " << cl << " " ;
		for(unsigned int j = 0; j < pixelsForCl[cl].size(); j++) {
			std::cout << pixelsForCl[cl][j].x<<","<<pixelsForCl[cl][j].y<<"|";
			if(cluster){

				SiPixelCluster::PixelPos newpix(pixelsForCl[cl][j].x,pixelsForCl[cl][j].y);
				cluster->add( newpix, pixelsForCl[cl][j].adc);
			}
			else
			{
				SiPixelCluster::PixelPos newpix(pixelsForCl[cl][j].x,pixelsForCl[cl][j].y);
				cluster = new  SiPixelCluster( newpix,pixelsForCl[cl][j].adc ); // create protocluster
			}
		}
		std::cout << std::endl;
		if(cluster){ 
			output.push_back(*cluster);
			delete cluster;
		}
	}
	//	if(verbose)	std::cout << "Weights" << std::endl;	
	//	if(verbose)	print(theWeights,aCluster,1);
	//	if(verbose)	std::cout << "Unused charge" << std::endl;	
	//	if(verbose)	print(theBufferResidual,aCluster);

	return output;
}

bool JetCoreClusterSplitter::nextCombination(std::vector<int> & comb,int npos)
{
	comb[comb.size()-1]+=1;//increment
	for(int i=comb.size()-1; i>=0 ;i--)
	{
		if(i > 0 && comb[i]>comb[i-1]) {comb[i]=0;comb[i-1]+=1;}
		if(i==0 && comb[i]>=npos) { return false; }		
	}
	return true;

}
void JetCoreClusterSplitter::print(const SiPixelArrayBuffer & b, const SiPixelCluster & c, int div )
{
	int xmin=c.minPixelRow();
	int ymin=c.minPixelCol();
	int xmax=c.maxPixelRow();
	int ymax=c.maxPixelCol();
	for(int x=xmin-5; x<= xmax+5;x++){
		for(int y=ymin-5; y<= ymax+5;y++){
			if(x<0||y<0) continue;
			if(b(x,y)!=0 && x<xmin) xmin=x;
			if(b(x,y)!=0 && y<ymin) ymin=y;
			if(b(x,y)!=0 && x>xmax) xmax=x;
			if(b(x,y)!=0 && y>ymax) ymax=y;
		}}


	for(int x=xmin; x<= xmax;x++){
		for(int y=ymin; y<= ymax;y++){
			std::cout << std::setiosflags(std::ios::fixed)
				<< std::setprecision(0)
				<< std::setw(6)
				<< std::left << b(x,y)/div;
		}
		std::cout << std::endl;
	}


}








#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter);


