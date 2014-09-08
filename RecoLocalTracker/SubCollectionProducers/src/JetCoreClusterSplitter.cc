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


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2.h"

float tanLorentz=0.404;

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
		bool split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,float xDeviation, float yDeviation,float jetZOverRho, const edmNew::DetSet<SiPixelCluster> & );
		float distanceCluster(const SiPixelCluster & cluster,const edmNew::DetSet<SiPixelCluster> & idealClusters);
		void print(const SiPixelArrayBuffer & b, const SiPixelCluster & aCluster, int div=1000 );
		std::vector<SiPixelCluster> fittingSplit(const SiPixelCluster & aCluster, float expectedADC,float xDeviation, float yDeviation,float jetZOverRho, int more);
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
		int steps00;
		int steps40;
		int steps60;
		int steps80;
		int steps90;
		int steps100;
		int clVuoti;
		int clVuotiMaxstep0;
		int clVuotiMaxstepNo0;
		int tuttiCl;
		int tuttiClMaxstep0;
		int fuori;
		int moreThanTen;
		int moreThanTen0;
		int moreThanEight;
		int moreThanEight0;
		TH1D * histoSteps;
		TH1D * histoNumberClusters;
		TH1D * numberIDEALClusters;
		TH1D * histoNumberClustersZero;
		TH1D * histoNotCentered;
		TH1D * histoNotCentered2;
		TH2D * numberIDEALClustersVsCharge;
		TH2D * numberIDEALClustersVsChargels10;

		TH2D * histoNotCentered2D;

		TH2D * CorrelazioneLunghezzeY;
		//TH2D * CorrelazioneLunghezzeY2;

		TH2D * andamentoSizeY;
		TH2D * andamentoSizeY_Carica;
		TH2D * andamentoSizeY_CaricaDaSplittare;
		TH2D * andamentoSizeY_CaricaDaNonSplittare;
		TH2D * andamentoSizeY_CaricaNo1SizeY;
		TH2D * andamentoSizeY_Carica1SizeY;
		TH2D * andamentoSizeYdaSplittare;
		TH2D * andamentoSizeYdaNonSplittare;
		TH1F * ScalarProductEB;
		TH1F * ScalarProductBperp;

		TH2F * XYProiection;
		TH2F * XYDrift;
		TH2F * LenghtsInDetectorBarrel;
		TH2F * LenghtsInDetector;
		TH2F * DifferenceLenghtsBarrel;
		TH2F * DifferenceLenghtsEndCap;
		TH2F * mVsZ;
		TH2F * jetZsuRhoCalcolata;
		TH1F * BdotY;
		TH1F * BdotX;
		TH2F * direzioneModuliEndCapX;

};

JetCoreClusterSplitter::JetCoreClusterSplitter(const edm::ParameterSet& iConfig):
	verbose(iConfig.getParameter<bool>("verbose")),
	pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),
	pixelClusters_(iConfig.getParameter<edm::InputTag>("pixelClusters")),
	vertices_(iConfig.getParameter<edm::InputTag>("vertices"))

{
   edm::Service<TFileService> fs;

   ScalarProductEB = fs->make<TH1F>("EdotB","Scalar product between E and B",200, -1.,1.);
   BdotY = fs->make<TH1F>("BdotY","Scalar product between local Y and B",200, -1.1,1.1);
   BdotX = fs->make<TH1F>("BdotX","Scalar product between local X and B",200, -1.1,1.1);

   direzioneModuliEndCapX = fs->make<TH2F>("direzioneModuliEndCap","Proiezione di X globale in X e Y locali nell'Endcap",200, -1.2,1.2, 200, -1.2,1.2);
   XYDrift = fs->make<TH2F>("XYDrift","Proiezione della velocità in X e Y locali",200, -1.2,1.2, 200, -1.2,1.2);


   XYProiection = fs->make<TH2F>("XYProiection","proiezione della direzione del jet su X e Y locali",200, -1.2,1.2, 200, -1.2,1.2);

   LenghtsInDetector = fs->make<TH2F>("LenghtsInDetector","Local Lenghts X and Y in the Detector",200, -2.,2., 200, -15.,15.);
   DifferenceLenghtsBarrel = fs->make<TH2F>("DifferenceLenghtsBarrel","Local difference in X and Y between begin and end point (Barrel)",200, -2.,2., 200, -15.,15.);
   DifferenceLenghtsEndCap = fs->make<TH2F>("DifferenceLenghtsEndCap","Local difference in X and Y between begin and end point (EndCap)",200, -2.,2., 200, -5,5);
   mVsZ = fs->make<TH2F>("mVsZ","m (angolar coefficient) Vs Zposition",200, -60,60, 300,-30,30);


   jetZsuRhoCalcolata = fs->make<TH2F>("jetZsuRhoCalcolata","calcolo dalla direzione Vs jetZOverRho",1000, 0.,15, 1000, 0.,15);

   histoSteps = fs->make<TH1D>("steps" , "steps" , 100 , 0 , 100 );
   histoNumberClusters = fs->make<TH1D>("numberClusters" , "Cluster che contengono x cluster" , 80 , 0 , 80 );
   numberIDEALClusters = fs->make<TH1D>("numberIDEALClusters" , "Cluster che contengono x IDEAL cluster" , 80 , 0 , 80 );
   histoNumberClustersZero = fs->make<TH1D>("StepsZeroVsNumCluster" , "StepsZero(y) Vs Number of Cluster(x)" , 80 , 0, 80 );
   histoNotCentered = fs->make<TH1D>("differenzaDistanzaY" , "y-(ymax+ymin)/2 se sizeY>1" , 1000 , -3. , 3. );
   histoNotCentered2 = fs->make<TH1D>("quasiMedia" , "distanza peso-media da (ymax+ymin)/2" , 1000 , -1. , 1. );
   numberIDEALClustersVsCharge = fs->make<TH2D>("numberIDEALClustersVsCharge" , "IDEAL cluster in un Cluster Vs la sua carica" , 100 , 0 , 70 , 60 , 0 , 20);
   numberIDEALClustersVsChargels10 = fs->make<TH2D>("numberIDEALClustersVsChargels10" , "IDEAL cluster in un Cluster Vs la sua carica" , 100 , 0 , 15 , 60 , 0 , 10);
   //ClustersVsChargels10 = fs->make<THD>("ClustersVsChargels10" , "cluster in un Cluster Vs la sua carica" , 1000 , 0 , 20 );

   histoNotCentered2D = fs->make<TH2D>("dist2D" , "dist2D" , 1000 , -3. , 3. , 1000 , -1. , 1. );

   CorrelazioneLunghezzeY = fs->make<TH2D>("correlazioneY" , "correlazione SizeY e lenght" , 1000 , 0 , 15 , 1000 , 0 , 15 );
   //CorrelazioneLunghezzeY2 = fs->make<TH2D>("correlazioneY(int)" , "correlazione sizeY(int) e lenght" , 1000 , 0 , 15 , 1000 , 0 , 15 );

   andamentoSizeY= fs->make<TH2D>("andamentoSizeYMisurato" , "SizeY(y) Vs jetZOverRho(x) misurato" , 500 , 0 , 20 , 500 , 0 , 20 );
   andamentoSizeY_Carica= fs->make<TH2D>("andamentoSizeY_NCluster" , "SizeY_misurata/attesa(y) Vs Number of Cluster(x) " , 500 , 0 , 20 , 200 , 0 , 10 );
   andamentoSizeY_CaricaDaSplittare= fs->make<TH2D>("andamentoSizeY_NClusterDaSplittare" , "SizeY_misurata/attesa(y) Vs Number of Cluster(x) da splittare " , 500 , 0 , 20 , 200 , 0 , 10 );
   andamentoSizeY_CaricaDaNonSplittare= fs->make<TH2D>("andamentoSizeY_NClusterDaNonSplittare" , "SizeY_misurata/attesa(y) Vs Number of Cluster(x) da splittare " , 500 , 0 , 20 , 200 , 0 , 10 );
   andamentoSizeY_CaricaNo1SizeY= fs->make<TH2D>("andamentoSizeY_NClusterNo1Y" , "SizeY_misurata/attesa(y) Vs Number of Cluster(x) SizeY!=1" , 500 , 0 , 20 , 200 , 0 , 10 );
   andamentoSizeY_Carica1SizeY= fs->make<TH2D>("andamentoSizeY_NCluster1Y" , "SizeY_misurata/attesa(y) Vs Number of Cluster(x) SizeY=1" , 500 , 0 , 20 , 200 , 0 , 10 );
   andamentoSizeYdaSplittare= fs->make<TH2D>("andamentoSizeYDaSpittare" , "SizeY(y) Vs jetZOverRho(x) da splittare" , 500 , 0 , 20 , 500 , 0 , 20 );
   andamentoSizeYdaNonSplittare= fs->make<TH2D>("andamentoSizeYDaNonSpittare" , "SizeY(y) Vs jetZOverRho(x) da non splittare" , 500 , 0 , 20 , 500 , 0 , 20 );

	nDirections=4;

	moreThanEight=0;
	moreThanEight0=0;
	moreThanTen0=0;
	moreThanTen=0;
	fuori=0;
	clVuoti =0;
	clVuotiMaxstep0 =0;
	clVuotiMaxstepNo0 =0;
	tuttiCl =0;
	tuttiClMaxstep0 =0;
	steps00 = 0;
	steps40 = 0;
	steps60 = 0;
	steps80 = 0;
	steps90 = 0;
	steps100 = 0;
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

	edm::ESHandle<TrackerGeometry> trkgeo;
	iSetup.get<TrackerDigiGeometryRecord>().get("",trkgeo);



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





	const GlobalPoint origin(0.,0.,0.);
	const GlobalPoint zAxis(0.,0.,1.);
	GlobalVector MagneticFieldDirection = zAxis -origin;

	const Local3DPoint center(0.,0.,0.);
	const Local3DPoint locz(0.,0.,1.);
	const Local3DPoint locx(1.,0.,0.);
	const Local3DPoint locy(0.,1.,0.);

	GlobalPoint position = trkgeo->idToDet(detIt->id())->toGlobal(center);
	GlobalPoint zpos = trkgeo->idToDet(detIt->id())->toGlobal(locz);
	GlobalPoint xpos = trkgeo->idToDet(detIt->id())->toGlobal(locx);
	GlobalPoint ypos = trkgeo->idToDet(detIt->id())->toGlobal(locy);

	GlobalVector dz = zpos - position;
	GlobalVector dx = xpos - position;
	GlobalVector dy = ypos - position;


	//GlobalVector firstDirection = ElectricFieldDirection;
	GlobalVector ElectricFieldDirection = dz;
	GlobalVector thirdDirection = ElectricFieldDirection.cross(MagneticFieldDirection);
	thirdDirection *= (1./thirdDirection.mag());
	GlobalVector secondDirection = thirdDirection.cross(ElectricFieldDirection);
	secondDirection *= (1./secondDirection.mag());


	//magnetudeThirdDir->Fill(secondDirection.mag());

	float scalarProductEB = ElectricFieldDirection.dot(MagneticFieldDirection);
	float scalarProductBperp = sqrt(MagneticFieldDirection.dot(dx)*MagneticFieldDirection.dot(dx) + MagneticFieldDirection.dot(dy) * MagneticFieldDirection.dot(dy));


	float v_1= 1./tanLorentz + scalarProductEB*tanLorentz;
	float v_2= tanLorentz*scalarProductBperp*scalarProductEB;
	float v_3= scalarProductBperp;
	secondDirection*=v_2/v_1;
	thirdDirection*=v_3/v_1;
	GlobalVector velocityDirection = secondDirection + thirdDirection;
	float xDrift = velocityDirection.dot(dx);
	float yDrift = velocityDirection.dot(dy);

	ScalarProductEB->Fill(scalarProductEB);
	XYDrift->Fill( xDrift, yDrift);
	BdotY->Fill( MagneticFieldDirection.dot(dy) );
	BdotX->Fill( MagneticFieldDirection.dot(dx) );
	if(fabs(cPos.z())>30) {
	const GlobalPoint xAxis(1.,0.,0.);
	GlobalVector xDir = xAxis - origin;
	direzioneModuliEndCapX->Fill( xDir.dot(dx), xDir.dot(dy) );
	}


			for(std::vector<reco::CaloJet>::const_iterator jit = jets->begin() ; jit != jets->end() ; jit++)
			{
				if(jit->pt() > 100)
				{
					float jetZOverRho = clusterDir.z()/clusterDir.transverse();
					if(fabs(cPos.z())>30) jetZOverRho=clusterDir.transverse()/clusterDir.z();
					GlobalVector jetDir(jit->momentum().x(),jit->momentum().y(),jit->momentum().z());

					unsigned int maxSizeY=fabs(sqrt(1.3*1.3+1.9*1.9*jetZOverRho*jetZOverRho));
//					unsigned int maxSizeY=fabs(jetZOverRho*1.9);
//					unsigned int maxSizeY=fabs(jetZOverRho*1.75)+0.5;
					if(maxSizeY < 1) maxSizeY=1;

					if(Geom::deltaR(jetDir,clusterDir) < 0.05 && aCluster.charge() > 30000 && (aCluster.sizeX() > 2 || ((unsigned int)aCluster.sizeY()) > maxSizeY+1) )
					{

					clusterDir*= (1./clusterDir.mag()) ;
					float xProiection = clusterDir.dot(dx);
					float yProiection = clusterDir.dot(dy);
					float zProiection = clusterDir.dot(dz);

					float lenghtDoneOnX=xProiection/zProiection;
					float lenghtDoneOnY=yProiection/zProiection;
					float lenghtDoneInXY = sqrt(lenghtDoneOnY*lenghtDoneOnY + lenghtDoneOnX*lenghtDoneOnX);
					
					XYProiection->Fill( xProiection, yProiection);


					//if(fabs(cPos.z())>30)  lenghtDone = lenghtDoneOnX;

					jetZsuRhoCalcolata->Fill (1.9*lenghtDoneInXY, 1.9*abs(jetZOverRho));
					LenghtsInDetector -> Fill( 1.9*lenghtDoneOnX , 1.9*lenghtDoneOnY );

					float xDeviation = 1.9*(lenghtDoneOnX-xDrift) ;
					float yDeviation = 1.9*(lenghtDoneOnY-yDrift) ;

					if(fabs(cPos.z())<30) DifferenceLenghtsBarrel -> Fill( xDeviation, yDeviation );
					else DifferenceLenghtsEndCap -> Fill( xDeviation, yDeviation );
					

					//float lenghtISee = sqrt(yDeviation*yDeviation + xDeviation*xDeviation);
					float m =  10000;
					if(abs(xDeviation)>0.001) m=yDeviation/xDeviation;
					mVsZ->Fill( cPos.z(), m );
					
					
		
	int ParticelleNelCluster=0;
	SiPixelClusterCollectionNew::const_iterator myDet =  inputPixelClustersIDEAL->find(detIt->id());
	for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIDEAL = myDet->begin(); clusterIDEAL != myDet->end() ;clusterIDEAL++ ) {
		bool appartiene=false;
		std::vector<SiPixelCluster::Pixel> pixels = aCluster.pixels();
		std::vector<SiPixelCluster::Pixel> pixelsIDEAL = clusterIDEAL->pixels();
		for(unsigned int j = 0; !appartiene && j < pixels.size(); j++)
		for(unsigned int k = 0; !appartiene && k < pixelsIDEAL.size(); k++)
		if(pixels[j].x==pixelsIDEAL[k].x && pixels[j].y==pixelsIDEAL[k].y) appartiene=true;

		if(appartiene) ParticelleNelCluster++;
	}

	float expectedADC = sqrt(1.+lenghtDoneInXY*lenghtDoneInXY)*26000;

	numberIDEALClusters->Fill(ParticelleNelCluster);
	numberIDEALClustersVsCharge->Fill( aCluster.charge()/expectedADC , ParticelleNelCluster);
	numberIDEALClustersVsChargels10->Fill( aCluster.charge()/expectedADC , ParticelleNelCluster);


						if(verbose) std::cout << "CHECK FOR NEW SPLITTING: charge and deltaR " <<aCluster.charge() << " " << Geom::deltaR(jetDir,clusterDir) << " size x y"<< aCluster.sizeX()  << " " << aCluster.sizeY()<< " detid " << detIt->id() << std::endl;	
						if(verbose)std::cout << "jetZOverRho="<< jetZOverRho << std::endl;	
						myDet =  inputPixelClustersIDEAL->find(detIt->id());
						clock_t init=clock(), final;
						const edmNew::DetSet<SiPixelCluster> & idealClusters  = (*myDet);
						if(split(aCluster,filler, expectedADC, xDeviation, yDeviation, lenghtDoneInXY, idealClusters)) 							{hasBeenSplit=true;
						 final=clock()-init; 
						 if(verbose)cout<<"Time used: (s) " << (double)final / ((double)CLOCKS_PER_SEC)<<endl;
						}
						//if(verbose)
						  std::cout << "IDEAL was : "  << std::endl; 
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
								 //std::setiosflags(std::ios::fixed)
								 //                               << std::setprecision(0)
								 //                             << std::setw(7)
								 //                          << std::left ; bin( flag,std::cout);
								 //                               << std::left << hex << flag;


							}
							if(verbose) std::cout << std::endl;         
						}
						int h=0;
						for(edmNew::DetSet<SiPixelCluster>::const_iterator clusterIt = myDet->begin(); clusterIt != myDet->end() ; clusterIt++,h++)
                                                                {
									//if(sh[h] && verbose) 
									if(sh[h]) 
									std::cout << "IDEAL POS: " << h << "sh[h]=" << sh[h] << " x: "  << std::setprecision(2) << clusterIt->x() << " y: " << clusterIt->y() << " c: " << clusterIt->charge() << std::endl;
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


cout<<"\n Final number of iterations: maxsteps Zero\t= % "<< std::setprecision(6) << (float) steps00*100. /steps100;

cout<<"\n Final number of iterations: maxsteps40   \t= % "<<(float) steps40*100. /steps100;

cout<<"\n Final number of iterations: maxsteps60   \t= % "<<(float) steps60*100. /steps100;

cout<<"\n Final number of iterations: maxsteps80   \t= % "<<(float) steps80*100. /steps100;

cout<<"\n Final number of iterations: maxsteps90   \t= % "<<(float) steps90*100. /steps100;
	
cout<<"\n Final number of iterations               \t=   "<< steps100;

cout<<"\n Clusters with no pixels and maxsteps Zero\t= % "<< (float) 100*clVuotiMaxstep0/tuttiClMaxstep0;

cout<<"\n Clusters with no pixels                  \t= % "<< (float) 100*clVuoti/tuttiCl;

cout<<"\n Total clusters                           \t=   "<< tuttiCl;

cout<<"\n Total clusters with maxsteps Zero        \t=   "<< tuttiClMaxstep0;

cout<<"\n (clVuoti,clVuotiMaxstep0,clVuotiMaxstepNo0)\t=   ( "<< clVuoti <<", " << clVuotiMaxstep0 << " ," << clVuotiMaxstepNo0 << ")";

cout<<"\n Pixels out of 1 in Y:                    \t=   "<< fuori;

cout<<"\n Clusters whit more than 10 particles inside\t=   "<< moreThanEight;
cout<<"\n Clusters whit more than 8 particles inside \t=   "<< moreThanTen;
cout<<"\n Clusters whit more than 10 particles Zero  \t=   "<< moreThanTen0;
cout<<"\n Clusters whit more than 8 particles Zero   \t=   "<< moreThanEight0;
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

bool JetCoreClusterSplitter::split(const SiPixelCluster & aCluster, edmNew::DetSetVector<SiPixelCluster>::FastFiller & filler, float expectedADC,float xDeviation, float yDeviation,float jetZOverRho,const edmNew::DetSet<SiPixelCluster> & idealClusters)
{
	std::vector<SiPixelCluster> sp=fittingSplit(aCluster,expectedADC,xDeviation,yDeviation,jetZOverRho, 0);
	

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


std::vector<SiPixelCluster> JetCoreClusterSplitter::fittingSplit(const SiPixelCluster & aCluster, float expectedADC,float xDeviation, float yDeviation,float jetZOverRho, int more)
{

andamentoSizeY->Fill(1.9*abs(jetZOverRho) , aCluster.sizeY() );
//check -> Fill( xDeviation , yDeviation );

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

	std::vector<SiPixelCluster> output;
	SiPixelCluster * trueCluster=0;
	SiPixelCluster * falseCluster=0;
	std::vector<bool> daTogliere(xmax-xmin+1);
	std::vector<bool> gtThanSizey(xmax-xmin+1);
	std::vector<bool> gtThanSizey2(xmax-xmin);
	std::vector<SiPixelCluster::Pixel> pixelsToSplit = aCluster.pixels();

	int NumberPixel1=0;
	int NumberPixel2=0;

	for(int x=xmin; x<= xmax;x++){
	NumberPixel1=NumberPixel2;
	NumberPixel2=0;

			for(unsigned int j = 0; j < pixelsToSplit.size(); j++)
			{
				if(pixelsToSplit[j].x==x ) 
				NumberPixel2++;
			}

	daTogliere[x-xmin] = true;	
	if(x!=xmin) {
	if(NumberPixel1+NumberPixel2 < 1.9*abs(jetZOverRho)-1 ) gtThanSizey2[x-xmin] = false;
	else gtThanSizey2[x-xmin-1] = true;
	}

	if(NumberPixel2 < 1.9*abs(jetZOverRho)-1 ) gtThanSizey[x-xmin] = false;
	else gtThanSizey[x-xmin] = true;
	/*if(NumberPixel1 > 1.9*abs(jetZOverRho)-2 ) 
	gtThanSizey2[x-xmin] = true;
	else gtThanSizey2[x-xmin] = false;*/

	}

for(unsigned int j=0; j<gtThanSizey2.size(); j++) {

//if(gtThanSizey[j]) daTogliere[j]=false;

//if(gtThanSizey[j+1]) daTogliere[j+1]=false;

if(gtThanSizey2[j] /* && !gtThanSizey[j+1] && !gtThanSizey[j]*/ ) {
daTogliere[j]=false;
daTogliere[j+1]=false;
}
}



	for(unsigned int j = 0; j < pixelsToSplit.size(); j++) {
		if(daTogliere[pixelsToSplit[j].x-xmin]) {

			if(falseCluster){

				SiPixelCluster::PixelPos newpix(pixelsToSplit[j].x,pixelsToSplit[j].y);
				falseCluster->add( newpix, pixelsToSplit[j].adc);
			}
			else
			{
				SiPixelCluster::PixelPos newpix(pixelsToSplit[j].x,pixelsToSplit[j].y);
				falseCluster = new  SiPixelCluster( newpix,pixelsToSplit[j].adc ); 
			}
		}


		else {
			if(trueCluster){

				SiPixelCluster::PixelPos newpix(pixelsToSplit[j].x,pixelsToSplit[j].y);
				trueCluster->add( newpix, pixelsToSplit[j].adc);
			}
			else
			{
				SiPixelCluster::PixelPos newpix(pixelsToSplit[j].x,pixelsToSplit[j].y);
				trueCluster = new  SiPixelCluster( newpix,pixelsToSplit[j].adc ); 
			}
		}
	}

int oneClusterMore=0;
	if(falseCluster){ 
		oneClusterMore=1;
		output.push_back(*falseCluster);
		andamentoSizeYdaNonSplittare->Fill(abs(1.9*jetZOverRho) , falseCluster->sizeY() );
	
		delete falseCluster;
	}


if(trueCluster) {}
else return output;

	unsigned int meanExp = 0 ;
	float lenght = 1.9*abs(jetZOverRho);


	std::cout <<  std::setprecision(7) << "sizeY " << floor(lenght+1) << std::endl; 	
	float NumberOfCluster = trueCluster->charge() / expectedADC +0.6 + more ;


if(1.*trueCluster->sizeY()/floor(lenght+1) <  0.5 ) NumberOfCluster = 1.1;
else {
if( lenght>1 && 1.*trueCluster->sizeY()/(floor(lenght+1)) >  NumberOfCluster*1.1+1)
if(1.*trueCluster->sizeY()/floor(lenght+1) > NumberOfCluster ) NumberOfCluster = 1.*trueCluster->sizeY()/floor(lenght+1);

if( lenght<1 && 1.*trueCluster->sizeY()/2. >  1.2*NumberOfCluster ) 
if(1.*trueCluster->sizeY()/2. > NumberOfCluster ) NumberOfCluster = 1.*trueCluster->sizeY()+0.1;

if(  1.*trueCluster->sizeY()/(floor(lenght)+1) <  0.3*NumberOfCluster-0.5 ) NumberOfCluster = 1.1;
}

andamentoSizeY_Carica->Fill(NumberOfCluster  , 1.*trueCluster->sizeY()/floor(lenght+1) );
if(trueCluster->sizeY()!=1)
andamentoSizeY_CaricaNo1SizeY->Fill( NumberOfCluster , 1.*trueCluster->sizeY()/floor(lenght+1) );
else
andamentoSizeY_Carica1SizeY->Fill( NumberOfCluster , 1.*trueCluster->sizeY()/floor(lenght+1) );

	if (NumberOfCluster >= 0 ) meanExp = floor(NumberOfCluster);

	if(meanExp==0) {std::cout << "ZERO????" << std::endl;} 
	if(meanExp<=1 || trueCluster->sizeY() < lenght-1 ||  trueCluster->sizeY() < lenght/2. ) {
		output.push_back(*trueCluster);	
		andamentoSizeYdaNonSplittare->Fill(1.9*abs(jetZOverRho) , trueCluster->sizeY() );
		andamentoSizeY_CaricaDaNonSplittare->Fill( NumberOfCluster , 1.*trueCluster->sizeY()/floor(lenght+1));	
		delete trueCluster;
		return output;
	}



andamentoSizeY_CaricaDaSplittare->Fill( NumberOfCluster , 1.*trueCluster->sizeY()/floor(lenght+1));
andamentoSizeYdaSplittare->Fill(1.9*abs(jetZOverRho) , trueCluster->sizeY() );	


	std::vector<float> clx(meanExp);
	std::vector<float> cly(meanExp);
	std::vector<float> cls(meanExp);
	std::vector<SiPixelCluster::Pixel> originalpixels = trueCluster->pixels();
	std::vector<SiPixelCluster::Pixel> pixels;



if(trueCluster)
delete trueCluster;



	std::vector<float> notCentered(meanExp);

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

	//initial values
	for(unsigned int j = 0; j < meanExp; j++)
	{
		/*clx[j]=originalpixels[0].x+j;
		cly[j]=originalpixels[0].y+j;*/


		  //CLUSTER INIZIALIZZATI IN PIXEL A CASO
      		if(j < originalpixels.size())
		{clx[j]=originalpixels[j].x;
		cly[j]=originalpixels[j].y;}
		else
		{clx[j]=originalpixels[0].x;
		cly[j]=originalpixels[0].y;} 


		cls[j]=0;
	}



	float m =  10000;
	//if(abs(xDeviation)>0.001) m=yDeviation/xDeviation;
	float lenghtISee=sqrt(xDeviation*xDeviation+yDeviation*yDeviation);

//HO PROVATO A STIMARE MEGLIO SIZEY

	/*float SizeY=0;
	lenght = sqrt(xDeviation*xDeviation+yDeviation*yDeviation);

	float minLenghtInTheDetector = 0.5;
	float delta = minLenghtInTheDetector*sqrt(jetZOverRho*jetZOverRho/(1.+jetZOverRho*jetZOverRho));
	int parteIntera = floor(lenght);
	float Pn = 0.;
	if(parteIntera + 2*delta < lenght) Pn =  lenght - parteIntera -2*delta;
	else Pn = lenght - parteIntera - 2*delta;

	if(delta>0.5)
	{ SizeY = lenght; lenght = 14;}
	else SizeY = ((float) parteIntera) + 1 + Pn;*/

	float SizeY=lenght;
	//float SizeY= sqrt(xDeviation*xDeviation+yDeviation*yDeviation);
	//if(SizeY<1) SizeY =1;

	CorrelazioneLunghezzeY->Fill( lenght , SizeY );
	//CorrelazioneLunghezzeY2->Fill( lenght , sizeY );


//BEGIN OF THE LOOP THAT ASSIGNE PIXELS TO CLUSTERS. REMEMBER THIS LINE:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

	//std::multimap<float,int> scoresDopo;

	std::vector<std::vector<int> > clusterForPix( 100, vector<int>(pixels.size()));
	std::vector<int> clusterForPixel(pixels.size());
	for(unsigned int i=0; i< clusterForPixel.size(); i++)
	clusterForPixel[i] = -1;

	bool pixelsAreMoving=true;
	int maxsteps=100;
	while( pixelsAreMoving && maxsteps > 0){
		maxsteps--;	
		//Compute all distances
		std::vector<std::vector<float> > distanceMapX( pixels.size(), vector<float>(meanExp));
		std::vector<std::vector<float> > distanceMapY( pixels.size(), vector<float>(meanExp));
		std::vector<std::vector<float> > distanceMap( pixels.size(), vector<float>(meanExp));
		for(unsigned int j = 0; j < pixels.size(); j++)
		{
if(verbose) 			std::cout << "pixel pos " << j << " " << pixels[j].x << " " << pixels[j].y << std::endl;



			float normalization = sqrt(1+(m*m));
			for(unsigned int i = 0; i < meanExp; i++)
			{
				float distanceX=(1.*pixels[j].x-clx[i])/1.5;
				float distanceY=1.*pixels[j].y-cly[i];
				distanceMapY[j][i]=(m*distanceY+distanceX)/normalization;
				distanceMapX[j][i]=(m*distanceX-distanceY)/normalization;

				//bool nelCluster = abs(m*distanceX-distanceY)/normalization > lenghtISee/2. + 0.5;

				//distanceMapX[j][i]=1.*pixels[j].x-clx[i];
				//distanceMapY[j][i]=1.*pixels[j].y-cly[i];
			        float dist=0;

				/*if(m>0) {
				if(distanceMapX[j][i]>0) {
				float distanzaClusterVertice=(m*(1.*pixels[j].x+0.5-clx[i])/1.5-(1.*pixels[j].y-0.5-cly[i]))/normalization;
				if(distanzaClusterVertice<0) distanceMapX[j][i]=0;
				else distanceMapX[j][i]=distanzaClusterVertice;
				} 
				else  {
				float distanzaClusterVertice=(m*(1.*pixels[j].x-0.5-clx[i])/1.5-(1.*pixels[j].y+0.5-cly[i]))/normalization;
				if(distanzaClusterVertice>0) distanceMapX[j][i]=0;
				else distanceMapX[j][i]=distanzaClusterVertice;
				}
				}

				else {
				if(distanceMapX[j][i]>0) {
				float distanzaClusterVertice=(m*(1.*pixels[j].x-0.5-clx[i])/1.5-(1.*pixels[j].y-0.5-cly[i]))/normalization;
				if(distanzaClusterVertice<0) distanceMapX[j][i]=0;
				else distanceMapX[j][i]=distanzaClusterVertice;
				} 
				
				else  {
				float distanzaClusterVertice=(m*(1.*pixels[j].x+0.5-clx[i])/1.5-(1.*pixels[j].y+0.5-cly[i]))/normalization;
				if(distanzaClusterVertice>0) distanceMapX[j][i]=0;
				else distanceMapX[j][i]=distanzaClusterVertice;
				}
				}


				dist+=(distanceMapX[j][i]+1)*(distanceMapX[j][i]+1);


                                if(std::abs(distanceMapY[j][i])>lenghtISee/2.+0.5)
                                {
                                        dist+=1.*(std::abs(distanceMapY[j][i])-lenghtISee/2.+0.5)*(std::abs(distanceMapY[j][i])-lenghtISee/2.+0.5);
                                } else {
                                        //dist+=1.*distanceMapY[j][i]/sizeY*2.*distanceMapY[j][i]/sizeY*2.;
					dist+=1;
                                }*/

				dist+=1.5*(distanceMapX[j][i])*(distanceMapX[j][i]);
                                if(std::abs(distanceMapY[j][i])>SizeY/2.)
                                {
                                        dist+=1.*(std::abs(distanceMapY[j][i])-SizeY/2.+1.)*(std::abs(distanceMapY[j][i])-SizeY/2.+1.);
                                } else {
                                        //dist+=1.*distanceMapY[j][i]/sizeY*2.*distanceMapY[j][i]/sizeY*2.;
					dist+=1;
                                }
                                distanceMap[j][i]=sqrt(dist);
if(verbose) 				std::cout << "Cluster " << i << " Pixel " << j << " distances: " << distanceMapX[j][i] << " " << distanceMapY[j][i]  << " " << distanceMap[j][i] << std::endl;


			}

		}



		//Compute scores for sequential addition
		std::multimap<float,int> scores;

		std::vector<float> weightOfPixel(pixels.size());


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
			float distanceToPutIn = minDist-secondMinDist;
			if(maxsteps <= 60 && maxsteps>40) distanceToPutIn = minDist;
			//if(maxsteps <= 40 && maxsteps>25) distanceToPutIn = minDist;
			if(maxsteps <=20) distanceToPutIn = minDist;
			scores.insert(std::pair<float,int>(distanceToPutIn,j));
			/*if(maxsteps == 10 ) 
			scoresDopo.insert(std::pair<float,int>(distanceToPutIn,j));
			if(maxsteps<10 ) 
				for(std::multimap<float,int>::iterator it=scoresDopo.begin(); it!=scoresDopo.end(); it++)
				 scores.insert(std::pair<float,int>(it->first,it->second));*/
				
			
		}

		//Iterate starting from the ones with furthest second best clusters, i.e. easy choices

		pixelsAreMoving = false;
		for(std::multimap<float,int>::iterator it=scores.begin(); it!=scores.end(); it++)
		{
			std::multimap<float,int>::iterator  pointInTheMap = it;
			pointInTheMap++;
			int j=it->second;

if(verbose)			std::cout << "Pixel " << j << " with score " << it->first << std::endl;
			//find cluster that is both close and has some charge still to assign
			float minEst=1e10;
			int cl=-1;
                        for(unsigned int i = 0; i < meanExp; i++)
			{
				if( pointInTheMap==scores.end() &&  cls[i] == 0 )
				{cl = i;
				pixelsAreMoving = true;}
				else {
			float chi1 = cls[i]/(expectedADC); 
			//float chi1=cls[i]*cls[i]/(expectedADC*10000); 
			//float chi2=expectedADC/10000; //20% uncertainty?realisticfrom Landau?
			  //float clQechi1st=1./sqrt(1.+exp(chi2))+1e-6; //1./(1.+exp(x*x-3*3))	
			//float chi = log(exp(chi2)+exp(chi1));

			//float clDest= 1./(exp(distanceMap[j][i]-1));


			float est = 0;
			if(maxsteps > 80)
			est = chi1+distanceMap[j][i];
			else {
			if(maxsteps > 40)
			est = (chi1*chi1+2)*distanceMap[j][i];
			else {
			/*if(maxsteps > 40 )
			est = (chi1+1)*distanceMap[j][i];
			else {
			if(maxsteps > 5)
			est = distanceMap[j][i];
			else */
			est = (chi1+3)*distanceMap[j][i];
			//}
			}
			}

//if(verbose)			  std::cout <<" Q: " <<  clQest << " D: " << clDest << " " << distanceMap[j][i] <<  std::endl; 
			  //float est=clQest*clDest;



			  if(est< minEst) {
				cl=i;
				minEst=est;
			//std::cout <<" Q: " <<  chi1 << " D: " << distanceMap[j][i] <<  std::endl;
			  }
				}
			}
			cls[cl]+=pixels[j].adc;
			if(clusterForPixel[j]!=cl) {
			pixelsAreMoving = true;
			clusterForPixel[j]=cl;
			}
			
			weightOfPixel[j]=minEst;
if(verbose) 			std::cout << "Pixel j weight " << weightOfPixel[j] << " " << j << std::endl;
		}

for(unsigned int j=0; j< clusterForPixel.size() ;j++)
clusterForPix[maxsteps][j]=clusterForPixel[j];


		//Recompute cluster centers

		std::vector<int> adcmin(meanExp);
		std::vector<int> adcmax(meanExp);
		std::vector<int> ymin(meanExp);
		std::vector<int> ymax(meanExp);

	        for(unsigned int i = 0; i < meanExp; i++) {
		cls[i]=0;
		adcmin[i]=0;
		adcmax[i]=0;
		ymin[i]=-1;
		ymax[i]=-1;
		}

                for(unsigned int j = 0; j < pixels.size(); j++)
		{
			if(clusterForPixel[j]<0) continue;
			cls[clusterForPixel[j]]+=pixels[j].adc;	
			ymin[clusterForPixel[j]]=pixels[j].y;
			ymax[clusterForPixel[j]]=pixels[j].y;
		}

		for(unsigned int j=0; j<pixels.size(); j++) {
		if(pixels[j].y < ymin[clusterForPixel[j]]) ymin[clusterForPixel[j]]=pixels[j].y;
		if(pixels[j].y > ymax[clusterForPixel[j]]) ymax[clusterForPixel[j]]=pixels[j].y;
		}

		for(unsigned int i=0; i<meanExp; i++) {
			for(unsigned int j=0; j<pixels.size(); j++) {
			int cl = i;
			if(clusterForPixel[j]==cl) {
			if(pixels[j].y == ymin[i]) adcmin[i]+=pixels[j].adc;
			if(pixels[j].y == ymax[i]) adcmax[i]+=pixels[j].adc;
			}
			}
		adcmin[i] = sqrt(adcmin[i]);
		adcmax[i] = sqrt(adcmax[i]);
		}


	        for(unsigned int i = 0; i < meanExp; i++)
		if(cls[i]!=0)
		{clx[i]=0; cly[i]=0;}

                for(unsigned int j = 0; j < pixels.size(); j++)
		{
			if(clusterForPixel[j]<0) continue;
if(verbose) 			std::cout << "x " << pixels[j].x <<" * " << pixels[j].adc << " * " << weightOfPixel[j]<<std::endl;
			clx[clusterForPixel[j]]+=pixels[j].x*pixels[j].adc;
			cly[clusterForPixel[j]]+=pixels[j].y*pixels[j].adc;	
	

			//if(ymax[clusterForPixel[j]]-ymin[clusterForPixel[j]]+1-1.9*jetZOverRho>0) cly[clusterForPixel[j]] = 1.*(ymin[clusterForPixel[j]]+ymax[clusterForPixel[j]])/2.;

		}

	        /*for(unsigned int i = 0; i < meanExp; i++) {
		if(cls[i]!=0) {
		cly[i]+=adcmax[i]*ymax[i];	
		cly[i]+=adcmin[i]*ymin[i];
		}
		}*/


		for(unsigned int i = 0; i < meanExp; i++){
			if(cls[i]!=0){
			clx[i]/=cls[i];
			/*if(ymax[i]-ymin[i]+1-1.9*jetZOverRho>0) cly[i]+=(adcmax[i]-adcmin[i])/(adcmax[i]+adcmin[i])*(ymax[i]-ymin[i]+1-1.9*jetZOverRho>0)/2;
			else */cly[i]/=(cls[i]/*+adcmin[i]+adcmax[i]*/);
			//cly[i]=1.*(ymax[i]+ymin[i])/2. + 1.*(adcmax[i]-adcmin[i])/2./(adcmax[i]+adcmin[i]);
			}
if(verbose) 			std::cout << "Center for cluster " << i << " x,y " << clx[i] << " " << cly[i] << std::endl;
		}
		


//If you aready have had this assignment, do not iterate again
for(int k=99; k>maxsteps && pixelsAreMoving; k--) {
bool loop=true;
for(unsigned int j=0; j< clusterForPixel.size() ;j++)
if(clusterForPix[k][j]!=clusterForPixel[j]) loop = false;
if(loop)  pixelsAreMoving = false;
}



//if there is a cluster without pixels, do another iteration
		bool thereAreEverybody = true;
		for(unsigned int i = 0; thereAreEverybody && (i< meanExp) ; i++) {
			thereAreEverybody = false;
			int aus = i;
			for(unsigned int j=0; j<pixels.size(); j++) {
				if(clusterForPixel[j]== aus)
				{thereAreEverybody = true;}
			}

		}
			if(!thereAreEverybody) pixelsAreMoving=true;





//Limit on the y position
	        for(unsigned int i = 0; i < meanExp; i++) {
		if(cls[i]!=0) {
		if(!pixelsAreMoving && ymax[i]>ymin[i]+1) {
		histoNotCentered->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2.);
		histoNotCentered2->Fill( 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) );
		/*if(ymax[i]>ymin[i]+2) {
		centridiversi->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2.);
		centridiversi2->Fill( 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) ); }*/
		adcmax[i]=adcmax[i]*adcmax[i];
		adcmin[i]=adcmin[i]*adcmin[i];
		histoNotCentered2D->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2. , 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) );
		/*if(ymax[i]>ymin[i]+2)
		histoNotCentered2D2->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2. , 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) );
		if(ymax[i]>ymin[i]+3)
		histoNotCentered2D3->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2. , 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) );
		if(ymax[i]>ymin[i]+4)
		histoNotCentered2D4->Fill( cly[i] - 1*(ymax[i]+ymin[i])/2. , 1.*(adcmax[i]-adcmin[i])/(1.*(adcmax[i]+adcmin[i])) );*/
		}
		if(cly[i]> 1*(ymax[i]+ymin[i])/2.+1)
		{cly[i]=1.*(ymax[i]+ymin[i])/2+1;
		fuori++;}
		if(cly[i]< 1*(ymax[i]+ymin[i])/2.-1)
		{cly[i]=1.*(ymax[i]+ymin[i])/2-1;
		fuori++;}
		}
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
					if(clusterForPixel[k]==cl and pixels[k].adc!=0 and pixels[k].x==pixels[j].x and pixels[k].y==pixels[j].y)
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
		bool clusterVuoto = true;
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
				clusterVuoto = false;
			}
		}
		std::cout << "\t" << expectedADC;
		if(!clusterVuoto) std::cout << "\t" << ((float) cluster->charge())/expectedADC;
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



/*
if(more == 0) {
std::vector<SiPixelCluster> oneMoreCluster = JetCoreClusterSplitter::fittingSplit( aCluster, expectedADC, sizeY, jetZOverRho, 1);
std::vector<SiPixelCluster> oneLessCluster = JetCoreClusterSplitter::fittingSplit( aCluster, expectedADC, sizeY, jetZOverRho, -1);

float chi0 = 100000;
float chiMore = 100000;
float chiLess = 100000;



for (std::vector<SiPixelCluster>::const_iterator it= output.begin(); it != output.end(); ++it) {
//float chi=((it->charge())*(it->charge())-expectedADC*expectedADC)/2./(expectedADC*2000); //20% uncertainty? realistic from Landau?
float chi = it->charge()/expectedADC;
float chi1 = 100*(0.5+5*chi)*(0.5+5*chi);
chi0=chi0-chi1;      //1./(1.+exp(x*x-3*3))
float dist =0;

	for(unsigned int i=0; i < it->pixels().size(); i++) {

		float distanceX=1.*it->pixels()[i].x-it->x();
		float distanceY=1.*it->pixels()[i].y-it->y();
		dist=0;
		dist+=1.*distanceX*distanceX;
			if(std::abs(distanceY)>SizeY/2.)
                        dist+=1.*(std::abs(distanceY)-SizeY/2.+1.)*(std::abs(distanceY)-SizeY/2.+1.);
			else 
			//dist+=1.*distanceY/SizeY*2.*distanceY/SizeY*2.;
			dist+=1.;
		}
cout << "chi0 è: \t" << chi0 << endl; 
chi0=chi0 - log(dist);
cout << "chi0 è: \t" << chi0 << endl; 
}


for (std::vector<SiPixelCluster>::const_iterator it= oneMoreCluster.begin(); it != oneMoreCluster.end(); ++it) {
//float chi=((it->charge())*(it->charge())-expectedADC*expectedADC)/2./(expectedADC*2000); //20% uncertainty? realistic from Landau?
float chi = it->charge()/expectedADC;
float chi1 = 100*(0.5+5*chi)*(0.5+5*chi);
chiMore=chiMore-chi1;
float dist=0;

	for(unsigned int i=0; i < it->pixels().size(); i++) {

		float distanceX=1.*it->pixels()[i].x-it->x();
		float distanceY=1.*it->pixels()[i].y-it->y();
	        dist=0;
		dist+=1.*distanceX*distanceX;
			if(std::abs(distanceY)>SizeY/2.)
                        dist+=1.*(std::abs(distanceY)-SizeY/2.+1.)*(std::abs(distanceY)-SizeY/2.+1.);
			else 
			//dist+=1.*distanceY/SizeY*2.*distanceY/SizeY*2.;
			dist+=1.;

	}
cout << "chiMore è: \t" << chiMore << endl; 
chiMore=chiMore-log(dist);
cout << "chiMore è: \t" << chiMore << endl; 
}


for (std::vector<SiPixelCluster>::const_iterator it= oneLessCluster.begin(); it != oneLessCluster.end(); ++it) {
//float chi=((it->charge())*(it->charge())-expectedADC*expectedADC)/2./(expectedADC*2000); //20% uncertainty? realistic from Landau?
float chi = it->charge()/expectedADC;
float chi1 = 100*(0.5+5*chi)*(0.5+5*chi);
chiLess=chiLess-chi1;
float dist=0;

	for(unsigned int i=0; i < it->pixels().size(); i++) {

		float distanceX=1.*it->pixels()[i].x-it->x();
		float distanceY=1.*it->pixels()[i].y-it->y();
	        dist=0;
		dist+=1.*distanceX*distanceX;
			if(std::abs(distanceY)>SizeY/2.)
                        dist+=1.*(std::abs(distanceY)-SizeY/2.+1.)*(std::abs(distanceY)-SizeY/2.+1.);
			else 
			//dist+=1.*distanceY/SizeY*2.*distanceY/SizeY*2.;
			dist+=1.;
	}
chiLess=chiLess-log(dist);
cout << "chiLess è: \t" << chiLess << endl; 
}

if(chiMore>chi0 && chiMore>chiLess) {
int More=2;
std::vector<SiPixelCluster> & outputMore = oneMoreCluster;
	while (chiMore>chi0) {
	std::vector<SiPixelCluster> clusterMore = JetCoreClusterSplitter::fittingSplit( aCluster, expectedADC, SizeY, jetZOverRho, More);
	chi0=chiMore;
	chiMore=100000;


	for (std::vector<SiPixelCluster>::const_iterator it= clusterMore.begin(); it != clusterMore.end(); ++it) {

	float chi = it->charge()/expectedADC;
	float chi1 = 100*(0.5+5*chi)*(0.5+5*chi);
	chiMore=chiMore-chi1;      //1./(1.+exp(x*x-3*3))	
	float dist=0;
		for(unsigned int i=0; i < it->pixels().size(); i++) {

		float distanceX=1.*it->pixels()[i].x-it->x();
		float distanceY=1.*it->pixels()[i].y-it->y();
	        dist=0;
		dist+=1.*distanceX*distanceX;
			if(std::abs(distanceY)>SizeY/2.)
                        dist+=1.*(std::abs(distanceY)-SizeY/2.+1.)*(std::abs(distanceY)-SizeY/2.+1.);
			else 
			//dist+=1.*distanceY/SizeY*2.*distanceY/SizeY*2.;
			dist+=1.;
		}
	chiMore=chiMore-log(dist);

	if(chiMore>chi0) outputMore = clusterMore;
	}
	More++;
	}
return outputMore;
}



if(chiLess>chi0 && chiLess>chiMore) {
int Less=2;
std::vector<SiPixelCluster> & outputLess = oneLessCluster;
	while (chiLess>chi0) {
	std::vector<SiPixelCluster> clusterLess = JetCoreClusterSplitter::fittingSplit( aCluster, expectedADC, sizeY, jetZOverRho, Less);
	chi0=chiLess;
	chiLess=100000;
	float dist=0;

	for (std::vector<SiPixelCluster>::const_iterator it= clusterLess.begin(); it != clusterLess.end(); ++it) {

	float chi = it->charge()/expectedADC;
	float chi1 = 100*(0.5+5*chi)*(0.5+5*chi);
	chiLess=chiLess-chi1;      //1./(1.+exp(x*x-3*3))	

		for(unsigned int i=0; i < it->pixels().size(); i++) {

		float distanceX=1.*it->pixels()[i].x-it->x();
		float distanceY=1.*it->pixels()[i].y-it->y();
	        dist=0;
		dist+=1.*distanceX*distanceX;
			if(std::abs(distanceY)>SizeY/2.)
                        dist+=1.*(std::abs(distanceY)-SizeY/2.+1.)*(std::abs(distanceY)-SizeY/2.+1.);
			else 
			//dist+=1.*distanceY/SizeY*2.*distanceY/SizeY*2.;
			dist+=1.;
		}
	chiLess=chiLess-log(dist);

	if(chiLess>chi0) outputLess = clusterLess;
	}
	Less++;
	}
return outputLess;
}

}
*/


if(meanExp>10) moreThanEight++;
if(meanExp>8) moreThanTen++;

tuttiCl += meanExp + oneClusterMore;
clVuoti += meanExp - output.size() + oneClusterMore; 

histoSteps -> Fill (maxsteps);
if(maxsteps < 80) 
histoSteps -> Fill (maxsteps, 9);



histoNumberClusters -> Fill (output.size());

	if(maxsteps == 0) {
	histoNumberClustersZero -> Fill (meanExp);
	steps00++;
	tuttiClMaxstep0 += meanExp;
	clVuotiMaxstep0 += meanExp - output.size() + oneClusterMore; 
	if( meanExp - output.size() !=0) cout << "è qui: \t" << clVuotiMaxstep0 << endl;
	if(meanExp>10) moreThanTen0++;
	if(meanExp>8) moreThanEight0++;
	}
	else {
	clVuotiMaxstepNo0 += meanExp - output.size() + oneClusterMore; 
	if(maxsteps < 40) steps40++;
	else {	
	if(maxsteps < 60) steps60++;
	else {
	if(maxsteps < 80) steps80++;
	else if(maxsteps < 90) steps90++;
	}
	}
	}
	steps100++;
	return output;

//cout << "maxstep " << steps << endl;

}




		//CLUSTER INIZIALIZZATI SU PIXEL A CASO CON DIVERSA X
		/*std::vector<int> cl_intx(meanExp);
		for(unsigned int i = 0; i<meanExp; i++) clx[i]=-1;

		bool diverse=true;
		for(unsigned int J = 0; J < originalpixels.size(); J++) {
		diverse = true;
		for(unsigned int k=0; k<j; k++)
		{if(originalpixels[J].x == (int) clx[k])
		diverse = false;}
		if (diverse) {
		clx[j]=originalpixels[J].x;
		cly[j]=originalpixels[J].y;
		break;}
		}
		if (!diverse)
		{if(j < originalpixels.size())
		{clx[j]=originalpixels[meanExp-j-1].x;
		cly[j]=originalpixels[meanExp-j-1].y;}
		else
		{clx[j]=originalpixels[0].x+j;
		cly[j]=originalpixels[0].y+j;}
		}*/ 


	
// Ridistribuisco i pixel un ultima volta per metterli meglio

/*	std::vector<float> weightOfPixel(pixels.size());
		std::vector<int> cl(pixels.size());

		for(unsigned int S = pixels.size(); S>0;  S--) {
		int clu=-1;
			for(unsigned int j = 0; j < S; j++)
			{

			//if(verbose)			std::cout << "Pixel " << j << " with score " << it->first << std::endl;
			//find cluster that is both close and has some charge still to assign
			float maxEst=0;

                        	for(unsigned int i = 0; i < meanExp; i++)
				{
			  	float chi2=(cls[i]*cls[i]-expectedADC*expectedADC)/2./(expectedADC*0.2); //20% uncertainty? realistic from Landau?
			  	float clQest=1./(1.+exp(chi2))+1e-6; //1./(1.+exp(x*x-3*3))	
			  	float clDest=1./(distanceMap[j][i]+0.05);

//if(verbose) 			  std::cout <<" Q: " <<  clQest << " D: " << clDest << " " << distanceMap[j][i] <<  std::endl;
			  	float est=clQest*clDest;
			  	if(est> maxEst) 
				{cl[j]=i;
				maxEst=est;}
				}
			
			weightOfPixel[j]=maxEst;
			//if(verbose) 			std::cout << "Pixel j weight " << weightOfPixel[j] << " " << j << std::endl;
			}

			float maxMaxEst = 0;
			unsigned int pix=0;
				for(unsigned int j = 0; j < S; j++)
				{
				if(weightOfPixel[j] > maxMaxEst) 
				{pix = j;
				clu = cl[j];}
				}

				
				for(unsigned int i = 0; i < meanExp; i++)
				{distanceMap[pix][i] = distanceMap[S-1][i];}
						

			cls[clu]+=pixels[pix].adc;

			int X = pixels[pix].x;
			int Y = pixels[pix].y;
			int ADC = pixels[pix].adc;
			pixels[pix].x = pixels[S-1].x;
			pixels[pix].y = pixels[S-1].y;
			pixels[pix].adc = pixels[S-1].adc;
			pixels[S-1].x = X;
			pixels[S-1].y = Y;
			pixels[S-1].adc = ADC;
	
			clusterForPixel[S-1]=clu;
			
	}
*/ 






	/*	
  float frac = 0.5;
 std::vector<std::vector<float> > Redistribution( originalpixels.size(), vector<float>(meanExp));
std::vector<int> Charge(originalpixels.size());
std::vector<float> NormalizationCharge(originalpixels.size());

for(unsigned int i = 0; i < meanExp; i++) {
for(unsigned int j = 0; j <  originalpixels.size(); j++) {
  if(j<pixelsForCl[i].size())
Redistribution[j][i]=exp(-distanceMap[j][i])/(1.+exp(chi2));
else Redistribution[j][i]=0;
}
}


for(unsigned int J = 0; J < originalpixels.size(); J++) {
NormalizationCharge[J]=0;
Charge[J]=0;
for(unsigned int i = 0; i < meanExp; i++) {
for(unsigned int j = 0; j < pixelsForCl[i].size(); j++) {
if(pixelsForCl[i][j].x == originalpixels[J].x && pixelsForCl[i][j].y == originalpixels[J].y ) 
{NormalizationCharge[J] +=  frac*Charge[J] * Redistribution[j][i]/NormalizationCharge[J];
Charge[J] += 1.*pixelsForCl[i][j].adc;}
}
}
}


for(unsigned int J = 0; J < originalpixels.size(); J++) {
for(unsigned int i = 0; i < meanExp; i++) {
for(unsigned int j = 0; j < pixelsForCl[i].size(); j++) {
if(pixelsForCl[i][j].x == originalpixels[J].x && pixelsForCl[i][j].y == originalpixels[J].y ) 
  pixelsForCl[i][j].adc = (int)( (1.-frac)*pixelsForCl[i][j].adc + frac*Charge[J] * Redistribution[j][i]/NormalizationCharge[J]); 
if(pixelsForCl[i][j].adc < 3 ) pixelsForCl[i][j].adc = 0;
}
}
}


	*/ 

	


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

	}


}








#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCoreClusterSplitter);


