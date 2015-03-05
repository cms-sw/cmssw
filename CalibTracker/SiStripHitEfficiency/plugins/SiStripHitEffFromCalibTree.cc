//Original Author:  Christopher Edelmaier
//        Created:  Feb. 11, 2010
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CalibTracker/SiStripHitEfficiency/interface/HitEff.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"

#include "TFile.h"
#include "TCanvas.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"
#include "TTree.h"
#include "TChain.h"
#include "TStyle.h"
#include "TLeaf.h"
#include "TGaxis.h"
#include "TGraphAsymmErrors.h"
#include "TLatex.h"
#include "TLegend.h"

using namespace edm;
using namespace reco;
using namespace std;

struct hit{
  double x;
  double y;
  double z;
  unsigned int id;
};

class SiStripHitEffFromCalibTree : public ConditionDBWriter<SiStripBadStrip> {
  public:
    explicit SiStripHitEffFromCalibTree(const edm::ParameterSet&);
    ~SiStripHitEffFromCalibTree();

  private:
    virtual void algoBeginJob();
    virtual void algoEndJob() override;
    virtual void algoAnalyze(const edm::Event& e, const edm::EventSetup& c) override;
    void SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC, std::stringstream ssV[4][19], int NBadComponent[4][19][4]);
    void makeTKMap();
    void makeHotColdMaps();
    void makeSQLite();
    void totalStatistics();
    void makeSummary();
    float calcPhi(float x, float y);

    edm::Service<TFileService> fs;
    SiStripDetInfoFileReader* reader;
    edm::FileInPath FileInPath_;
    SiStripQuality* quality_;
    SiStripBadStrip* getNewObject() override;
    
    TFile* CalibTreeFile;
    TTree* CalibTree;
    TString CalibTreeFilename; 
    float threshold;
    unsigned int nModsMin;
    unsigned int doSummary;
    float _ResXSig;
    unsigned int _bunchx;
    vector<hit> hits[23];
    vector<TH2F*> HotColdMaps;
    map< unsigned int, pair< unsigned int, unsigned int> > modCounter[23];
    TrackerMap *tkmap;
    TrackerMap *tkmapbad;
    int layerfound[23];
    int layertotal[23];
    int goodlayertotal[35];
    int goodlayerfound[35];
    int alllayertotal[35];
    int alllayerfound[35];
    map< unsigned int, double > BadModules;
};

SiStripHitEffFromCalibTree::SiStripHitEffFromCalibTree(const edm::ParameterSet& conf) :
  ConditionDBWriter<SiStripBadStrip>(conf),
  FileInPath_("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat")
{
  CalibTreeFilename = conf.getParameter<std::string>("CalibTreeFilename"); 
  threshold = conf.getParameter<double>("Threshold");
  nModsMin = conf.getParameter<int>("nModsMin");
  doSummary = conf.getParameter<int>("doSummary");
  _ResXSig = conf.getUntrackedParameter<double>("ResXSig",-1);
  _bunchx = conf.getUntrackedParameter<int>("BunchCrossing",0);
  reader = new SiStripDetInfoFileReader(FileInPath_.fullPath());
  
  quality_ = new SiStripQuality;
}

SiStripHitEffFromCalibTree::~SiStripHitEffFromCalibTree() { }

void SiStripHitEffFromCalibTree::algoBeginJob() {
  //I have no idea what goes here
  //fs->make<TTree>("HitEffHistos","Tree of the inefficient hit histograms");
}

void SiStripHitEffFromCalibTree::algoEndJob() {
  //Still have no idea what goes here

}

void SiStripHitEffFromCalibTree::algoAnalyze(const edm::Event& e, const edm::EventSetup& c) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  c.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //Open the ROOT Calib Tree
  CalibTreeFile = TFile::Open(CalibTreeFilename,"READ");
  CalibTreeFile->cd("anEff"); 
  CalibTree = (TTree*)(gDirectory->Get("traj")) ;
  TLeaf* BadLf = CalibTree->GetLeaf("ModIsBad");
  TLeaf* sistripLf = CalibTree->GetLeaf("SiStripQualBad");
  TLeaf* idLf = CalibTree->GetLeaf("Id");
  TLeaf* acceptLf = CalibTree->GetLeaf("withinAcceptance");
  TLeaf* layerLf = CalibTree->GetLeaf("layer");
  TLeaf* nHitsLf = CalibTree->GetLeaf("nHits");
  TLeaf* xLf = CalibTree->GetLeaf("TrajGlbX");
  TLeaf* yLf = CalibTree->GetLeaf("TrajGlbY");
  TLeaf* zLf = CalibTree->GetLeaf("TrajGlbZ");
  TLeaf* ResXSigLf = CalibTree->GetLeaf("ResXSig");
  TLeaf* BunchLf(0);
  for(int l=0; l < 35; l++) {
    goodlayertotal[l] = 0;
    goodlayerfound[l] = 0;
    alllayertotal[l] = 0;
    alllayerfound[l] = 0;
  }
  if(_bunchx != 0) {
    BunchLf = CalibTree->GetLeaf("bunchx");
  }
  int nevents = CalibTree->GetEntries();
  cout << "Successfully loaded analyze function with " << nevents << " events!\n";
  cout << "A module is bad if efficiency < " << threshold << " and has at least " << nModsMin << " nModsMin." << endl;

  //Loop through all of the events
  for(int j =0; j < nevents; j++) {
    CalibTree->GetEvent(j);
    unsigned int isBad = (unsigned int)BadLf->GetValue();
    unsigned int quality = (unsigned int)sistripLf->GetValue();
    unsigned int id = (unsigned int)idLf->GetValue();
    unsigned int accept = (unsigned int)acceptLf->GetValue();
    unsigned int layer = (unsigned int)layerLf->GetValue();
    unsigned int nHits = (unsigned int)nHitsLf->GetValue();
    double x = xLf->GetValue();
    double y = yLf->GetValue();
    double z = zLf->GetValue();
    double resxsig = ResXSigLf->GetValue();
    bool badquality = false;
    if(_bunchx != 0) {
      if(_bunchx != BunchLf->GetValue()) continue;
    }
    //We have two things we want to do, both an XY color plot, and the efficiency measurement
    //First, ignore anything that isn't in acceptance and isn't good quality
    
    //if(quality == 1 || accept != 1 || nHits < 8) continue;
    if(accept != 1 || nHits < 8) continue;
    if(quality == 1) badquality = true;
    
    //Now that we have a good event, we need to look at if we expected it or not, and the location
    //if we didn't
    //Fill the missing hit information first
    bool badflag = false;
    if(_ResXSig < 0) {
      if(isBad == 1) badflag = true;
    }
    else {
      if(isBad == 1 || resxsig > _ResXSig) badflag = true;
    }
    if(badflag && !badquality) {   
      hit temphit;         
      temphit.x = x;
      temphit.y = y;
      temphit.z = z;
      temphit.id = id;
      hits[layer].push_back(temphit);
    } 
    pair<unsigned int, unsigned int> newgoodpair (1,1);
    pair<unsigned int, unsigned int> newbadpair (1,0);
    //First, figure out if the module already exists in the map of maps
    map< unsigned int, pair< unsigned int, unsigned int> >::iterator it = modCounter[layer].find(id);
    if(!badquality) {
      if(it == modCounter[layer].end()) {
        if(badflag) modCounter[layer][id] = newbadpair;
        else modCounter[layer][id] = newgoodpair;
      }
      else {
        ((*it).second.first)++;
        if(!badflag) ((*it).second.second)++;
      }
      //Have to do the decoding for which side to go on (ugh)
      if(layer <= 10) {
        if(!badflag) goodlayerfound[layer]++;
        goodlayertotal[layer]++;
      }
      else if(layer > 10 && layer < 14) {
        if( ((id>>13)&0x3) == 1) {
	  if(!badflag) goodlayerfound[layer]++;
          goodlayertotal[layer]++;
	}
	else if( ((id>>13)&0x3) == 2) {
	  if(!badflag) goodlayerfound[layer+3]++;
          goodlayertotal[layer+3]++;
	}
      }
      else if(layer > 13 && layer <= 22) {
        if( ((id>>18)&0x3) == 1) {
	  if(!badflag) goodlayerfound[layer+3]++;
          goodlayertotal[layer+3]++;
	}
	else if( ((id>>18)&0x3) == 2) {
	  if(!badflag) goodlayerfound[layer+12]++;
          goodlayertotal[layer+12]++;
	}
      } 
    }
    //Do the one where we don't exclude bad modules!
    if(layer <= 10) {
      if(!badflag) alllayerfound[layer]++;
      alllayertotal[layer]++;
    }
    else if(layer > 10 && layer < 14) {
      if( ((id>>13)&0x3) == 1) {
	if(!badflag) alllayerfound[layer]++;
        alllayertotal[layer]++;
      }
      else if( ((id>>13)&0x3) == 2) {
        if(!badflag) alllayerfound[layer+3]++;
        alllayertotal[layer+3]++;
      }
    }
    else if(layer > 13 && layer <= 22) {
      if( ((id>>18)&0x3) == 1) {
        if(!badflag) alllayerfound[layer+3]++;
        alllayertotal[layer+3]++;
      }
      else if( ((id>>18)&0x3) == 2) {
        if(!badflag) alllayerfound[layer+12]++;
        alllayertotal[layer+12]++;
      }
    }  
    //At this point, both of our maps are loaded with the correct information
  }
  //CalibTreeFile->Close();
  makeHotColdMaps();
  makeTKMap();
  makeSQLite();
  totalStatistics();
  makeSummary();
  
  ////////////////////////////////////////////////////////////////////////
  //try to write out what's in the quality record
  /////////////////////////////////////////////////////////////////////////////
  int NTkBadComponent[4]; //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];  
  //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  std::stringstream ssV[4][19];
  
  for(int i=0;i<4;++i){
    NTkBadComponent[i]=0;
    for(int j=0;j<19;++j){
      ssV[i][j].str("");
      for(int k=0;k<4;++k)
        NBadComponent[i][j][k]=0;
    }
  }
 
 
  std::vector<SiStripQuality::BadComponent> BC = quality_->getBadComponentList();
   
  for (size_t i=0;i<BC.size();++i){
     
    //&&&&&&&&&&&&&
    //Full Tk
    //&&&&&&&&&&&&&
 
    if (BC[i].BadModule) 
      NTkBadComponent[0]++;
    if (BC[i].BadFibers) 
      NTkBadComponent[1]+= ( (BC[i].BadFibers>>2)&0x1 )+ ( (BC[i].BadFibers>>1)&0x1 ) + ( (BC[i].BadFibers)&0x1 );
    if (BC[i].BadApvs)
      NTkBadComponent[2]+= ( (BC[i].BadApvs>>5)&0x1 )+ ( (BC[i].BadApvs>>4)&0x1 ) + ( (BC[i].BadApvs>>3)&0x1 ) + 
	( (BC[i].BadApvs>>2)&0x1 )+ ( (BC[i].BadApvs>>1)&0x1 ) + ( (BC[i].BadApvs)&0x1 );
 
    //&&&&&&&&&&&&&&&&&
    //Single SubSystem
    //&&&&&&&&&&&&&&&&&
 
    int component;
    SiStripDetId a(BC[i].detid);
    if ( a.subdetId() == SiStripDetId::TIB ){
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&
       
      component=tTopo->tibLayer(BC[i].detid);
      SetBadComponents(0, component, BC[i], ssV, NBadComponent);	      
 
    } else if ( a.subdetId() == SiStripDetId::TID ) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&
 
      component=tTopo->tidSide(BC[i].detid)==2?tTopo->tidWheel(BC[i].detid):tTopo->tidWheel(BC[i].detid)+3;
      SetBadComponents(1, component, BC[i], ssV, NBadComponent);	      
 
    } else if ( a.subdetId() == SiStripDetId::TOB ) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&
 
      component=tTopo->tobLayer(BC[i].detid);
      SetBadComponents(2, component, BC[i], ssV, NBadComponent);	      
 
    } else if ( a.subdetId() == SiStripDetId::TEC ) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&
 
      component=tTopo->tecSide(BC[i].detid)==2?tTopo->tecWheel(BC[i].detid):tTopo->tecWheel(BC[i].detid)+9;
      SetBadComponents(3, component, BC[i], ssV, NBadComponent);	      
 
    }    
  }
 
  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&
  float percentage=0;
 
  SiStripQuality::RegistryIterator rbegin = quality_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend   = quality_->getRegistryVectorEnd();
   
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    unsigned int detid=rp->detid;
 
    int subdet=-999; int component=-999;
    SiStripDetId a(detid);
    if ( a.subdetId() == 3 ){
      subdet=0;
      component=tTopo->tibLayer(detid);
    } else if ( a.subdetId() == 4 ) {
      subdet=1;
      component=tTopo->tidSide(detid)==2?tTopo->tidWheel(detid):tTopo->tidWheel(detid)+3;
    } else if ( a.subdetId() == 5 ) {
      subdet=2;
      component=tTopo->tobLayer(detid);
    } else if ( a.subdetId() == 6 ) {
      subdet=3;
      component=tTopo->tecSide(detid)==2?tTopo->tecWheel(detid):tTopo->tecWheel(detid)+9;
    } 
 
    SiStripQuality::Range sqrange = SiStripQuality::Range( quality_->getDataVectorBegin()+rp->ibegin , quality_->getDataVectorBegin()+rp->iend );
	 
    percentage=0;
    for(int it=0;it<sqrange.second-sqrange.first;it++){
      unsigned int range=quality_->decode( *(sqrange.first+it) ).range;
      NTkBadComponent[3]+=range;
      NBadComponent[subdet][0][3]+=range;
      NBadComponent[subdet][component][3]+=range;
      percentage+=range;
    }
    if(percentage!=0)
      percentage/=128.*reader->getNumberOfApvsAndStripLength(detid).first;
    if(percentage>1)
      edm::LogError("SiStripQualityStatistics") <<  "PROBLEM detid " << detid << " value " << percentage<< std::endl; 
  }
  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&
 
  cout << "\n-----------------\nNew IOV starting from run " <<   e.id().run() << " event " << e.id().event() << " lumiBlock " << e.luminosityBlock() << " time " << e.time().value()  << "\n-----------------\n";
  cout << "\n-----------------\nGlobal Info\n-----------------";
  cout << "\nBadComponent \t	Modules \tFibers \tApvs\tStrips\n----------------------------------------------------------------";
  cout << "\nTracker:\t\t"<<NTkBadComponent[0]<<"\t"<<NTkBadComponent[1]<<"\t"<<NTkBadComponent[2]<<"\t"<<NTkBadComponent[3];
  cout << endl;
  cout << "\nTIB:\t\t\t"<<NBadComponent[0][0][0]<<"\t"<<NBadComponent[0][0][1]<<"\t"<<NBadComponent[0][0][2]<<"\t"<<NBadComponent[0][0][3];
  cout << "\nTID:\t\t\t"<<NBadComponent[1][0][0]<<"\t"<<NBadComponent[1][0][1]<<"\t"<<NBadComponent[1][0][2]<<"\t"<<NBadComponent[1][0][3];
  cout << "\nTOB:\t\t\t"<<NBadComponent[2][0][0]<<"\t"<<NBadComponent[2][0][1]<<"\t"<<NBadComponent[2][0][2]<<"\t"<<NBadComponent[2][0][3];
  cout << "\nTEC:\t\t\t"<<NBadComponent[3][0][0]<<"\t"<<NBadComponent[3][0][1]<<"\t"<<NBadComponent[3][0][2]<<"\t"<<NBadComponent[3][0][3];
  cout << "\n";
 
  for (int i=1;i<5;++i)
    cout << "\nTIB Layer " << i   << " :\t\t"<<NBadComponent[0][i][0]<<"\t"<<NBadComponent[0][i][1]<<"\t"<<NBadComponent[0][i][2]<<"\t"<<NBadComponent[0][i][3];
  cout << "\n";
  for (int i=1;i<4;++i)
    cout << "\nTID+ Disk " << i   << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  for (int i=4;i<7;++i)
    cout << "\nTID- Disk " << i-3 << " :\t\t"<<NBadComponent[1][i][0]<<"\t"<<NBadComponent[1][i][1]<<"\t"<<NBadComponent[1][i][2]<<"\t"<<NBadComponent[1][i][3];
  cout << "\n";
  for (int i=1;i<7;++i)
    cout << "\nTOB Layer " << i   << " :\t\t"<<NBadComponent[2][i][0]<<"\t"<<NBadComponent[2][i][1]<<"\t"<<NBadComponent[2][i][2]<<"\t"<<NBadComponent[2][i][3];
  cout << "\n";
  for (int i=1;i<10;++i)
    cout << "\nTEC+ Disk " << i   << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
  for (int i=10;i<19;++i)
    cout << "\nTEC- Disk " << i-9 << " :\t\t"<<NBadComponent[3][i][0]<<"\t"<<NBadComponent[3][i][1]<<"\t"<<NBadComponent[3][i][2]<<"\t"<<NBadComponent[3][i][3];
  cout << "\n";
 
  cout << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers Apvs\n----------------------------------------------------------------";
  for (int i=1;i<5;++i)
    cout << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  cout << "\n";
  for (int i=1;i<4;++i)
    cout << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i=4;i<7;++i)
    cout << "\nTID- Disk " << i-3 << " :" << ssV[1][i].str();
  cout << "\n";
  for (int i=1;i<7;++i)
    cout << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  cout << "\n";
  for (int i=1;i<10;++i)
    cout << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i=10;i<19;++i)
    cout << "\nTEC- Disk " << i-9 << " :" << ssV[3][i].str();

}

void SiStripHitEffFromCalibTree::makeHotColdMaps() {
  cout << "Entering hot cold map generation!\n";
  TStyle* gStyle = new TStyle("gStyle","myStyle");
  gStyle->cd();
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetOptStat(0);
  //Here we make the hot/cold color maps that we love so very much
  //Already have access to the data as a private variable
  //Create all of the histograms in the TFileService 
  TH2F *temph2;
  for(Long_t maplayer = 1; maplayer <=22; maplayer++) { 										  
    //Initialize all of the histograms													  
    if(maplayer > 0 && maplayer <= 4) { 												  
  	//We are in the TIB														  
  	temph2 = fs->make<TH2F>(Form("%s%i","TIB",(int)(maplayer)),"TIB",100,-1,361,100,-100,100);					  
  	temph2->GetXaxis()->SetTitle("Phi");
	temph2->GetXaxis()->SetBinLabel(1,TString("360"));
	temph2->GetXaxis()->SetBinLabel(50,TString("180"));
	temph2->GetXaxis()->SetBinLabel(100,TString("0"));												  
  	temph2->GetYaxis()->SetTitle("Global Z");
	temph2->SetOption("colz");											  
  	HotColdMaps.push_back(temph2);													  
      } 																  
      else if(maplayer > 4 && maplayer <= 10) { 											  
  	//We are in the TOB														  
  	temph2 = fs->make<TH2F>(Form("%s%i","TOB",(int)(maplayer-4)),"TOB",100,-1,361,100,-120,120);				  
  	temph2->GetXaxis()->SetTitle("Phi");
	temph2->GetXaxis()->SetBinLabel(1,TString("360"));
	temph2->GetXaxis()->SetBinLabel(50,TString("180"));
	temph2->GetXaxis()->SetBinLabel(100,TString("0"));												  
  	temph2->GetYaxis()->SetTitle("Global Z");
	temph2->SetOption("colz");												  
  	HotColdMaps.push_back(temph2);													  
      } 																  
      else if(maplayer > 10 && maplayer <= 13) {											  
  	//We are in the TID														  
  	//Split by +/-															  
  	temph2 = fs->make<TH2F>(Form("%s%i","TID-",(int)(maplayer-10)),"TID-",100,-100,100,100,-100,100);			  
  	temph2->GetXaxis()->SetTitle("Global Y");
	temph2->GetXaxis()->SetBinLabel(1,TString("+Y"));
	temph2->GetXaxis()->SetBinLabel(50,TString("0"));
	temph2->GetXaxis()->SetBinLabel(100,TString("-Y"));												  
  	temph2->GetYaxis()->SetTitle("Global X");
	temph2->GetYaxis()->SetBinLabel(1,TString("-X"));
	temph2->GetYaxis()->SetBinLabel(50,TString("0"));
	temph2->GetYaxis()->SetBinLabel(100,TString("+X"));
	temph2->SetOption("colz");												  
  	HotColdMaps.push_back(temph2);													  
  	temph2 = fs->make<TH2F>(Form("%s%i","TID+",(int)(maplayer-10)),"TID+",100,-100,100,100,-100,100);			  
  	temph2->GetXaxis()->SetTitle("Global Y");
	temph2->GetXaxis()->SetBinLabel(1,TString("+Y"));
	temph2->GetXaxis()->SetBinLabel(50,TString("0"));
	temph2->GetXaxis()->SetBinLabel(100,TString("-Y"));												  
  	temph2->GetYaxis()->SetTitle("Global X");
	temph2->GetYaxis()->SetBinLabel(1,TString("-X"));
	temph2->GetYaxis()->SetBinLabel(50,TString("0"));
	temph2->GetYaxis()->SetBinLabel(100,TString("+X"));												  
  	temph2->SetOption("colz");
	HotColdMaps.push_back(temph2);													  
      } 																  
      else if(maplayer > 13) {
        //We are in the TEC
        //Split by +/-
        temph2 = fs->make<TH2F>(Form("%s%i","TEC-",(int)(maplayer-13)),"TEC-",100,-120,120,100,-120,120);
        temph2->GetXaxis()->SetTitle("Global Y");
	temph2->GetXaxis()->SetBinLabel(1,TString("+Y"));
	temph2->GetXaxis()->SetBinLabel(50,TString("0"));
	temph2->GetXaxis()->SetBinLabel(100,TString("-Y"));												  
  	temph2->GetYaxis()->SetTitle("Global X");
	temph2->GetYaxis()->SetBinLabel(1,TString("-X"));
	temph2->GetYaxis()->SetBinLabel(50,TString("0"));
	temph2->GetYaxis()->SetBinLabel(100,TString("+X"));
	temph2->SetOption("colz");
        HotColdMaps.push_back(temph2);
        temph2 = fs->make<TH2F>(Form("%s%i","TEC+",(int)(maplayer-13)),"TEC+",100,-120,120,100,-120,120);
        temph2->GetXaxis()->SetTitle("Global Y");
	temph2->GetXaxis()->SetBinLabel(1,TString("+Y"));
	temph2->GetXaxis()->SetBinLabel(50,TString("0"));
	temph2->GetXaxis()->SetBinLabel(100,TString("-Y"));												  
  	temph2->GetYaxis()->SetTitle("Global X");
	temph2->GetYaxis()->SetBinLabel(1,TString("-X"));
	temph2->GetYaxis()->SetBinLabel(50,TString("0"));
	temph2->GetYaxis()->SetBinLabel(100,TString("+X"));
	temph2->SetOption("colz");
        HotColdMaps.push_back(temph2);
    }
  }
  for(Long_t mylayer = 1; mylayer <= 22; mylayer++) {
    //Determine what kind of plot we want to write out
    //Loop through the entirety of each layer
    //Create an array of the histograms
    vector<hit>::const_iterator iter;
    for(iter = hits[mylayer].begin(); iter != hits[mylayer].end(); iter++) {
      //Looping over the particular layer
      //Fill by 360-x to get the proper location to compare with TKMaps of phi
      //Also global xy is messed up
      if(mylayer > 0 && mylayer <= 4) {
        //We are in the TIB
        float phi = calcPhi(iter->x, iter->y);
        HotColdMaps[mylayer - 1]->Fill(360.-phi,iter->z,1.); 
      }
      else if(mylayer > 4 && mylayer <= 10) {
        //We are in the TOB
        float phi = calcPhi(iter->x,iter->y);
        HotColdMaps[mylayer - 1]->Fill(360.-phi,iter->z,1.);
      }
      else if(mylayer > 10 && mylayer <= 13) {
        //We are in the TID
        //There are 2 different maps here
        int side = (((iter->id)>>13) & 0x3);
	if(side == 1) HotColdMaps[(mylayer - 1) + (mylayer - 11)]->Fill(-iter->y,iter->x,1.); 
        else if(side == 2) HotColdMaps[(mylayer - 1) + (mylayer - 10)]->Fill(-iter->y,iter->x,1.);
        //if(side == 1) HotColdMaps[(mylayer - 1) + (mylayer - 11)]->Fill(iter->x,iter->y,1.); 
        //else if(side == 2) HotColdMaps[(mylayer - 1) + (mylayer - 10)]->Fill(iter->x,iter->y,1.);
      }
      else if(mylayer > 13) {
        //We are in the TEC
        //There are 2 different maps here
        int side = (((iter->id)>>18) & 0x3);
	if(side == 1) HotColdMaps[(mylayer + 2) + (mylayer - 14)]->Fill(-iter->y,iter->x,1.);
        else if(side == 2) HotColdMaps[(mylayer + 2) + (mylayer - 13)]->Fill(-iter->y,iter->x,1.);
        //if(side == 1) HotColdMaps[(mylayer + 2) + (mylayer - 14)]->Fill(iter->x,iter->y,1.);
        //else if(side == 2) HotColdMaps[(mylayer + 2) + (mylayer - 13)]->Fill(iter->x,iter->y,1.);
      }
    }
  }
  cout << "Finished HotCold Map Generation\n";
}

void SiStripHitEffFromCalibTree::makeTKMap() {
  cout << "Entering TKMap generation!\n";
  tkmap = new TrackerMap("  Detector Inefficiency  ");
  tkmapbad = new TrackerMap("  Inefficient Modules  ");
  for(Long_t i = 1; i <= 22; i++) {
    layertotal[i] = 0;
    layerfound[i] = 0;
    //Loop over every layer, extracting the information from
    //the map of the efficiencies
    map<unsigned int, pair<unsigned int, unsigned int> >::const_iterator ih;
    for( ih = modCounter[i].begin(); ih != modCounter[i].end(); ih++) {
      //We should be in the layer in question, and looping over all of the modules in said layer
      //Generate the list for the TKmap, and the bad module list
      double myeff = (double)(((*ih).second).second)/(((*ih).second).first);
      if ( ((((*ih).second).first) >= nModsMin) && (myeff < threshold) ) {
        //We have a bad module, put it in the list!
	BadModules[(*ih).first] = myeff;
	tkmapbad->fillc((*ih).first,255,0,0);
	cout << "Layer " << i << " module " << (*ih).first << " efficiency " << myeff << " " << (((*ih).second).second) << "/" << (((*ih).second).first) << endl;
      }
      else {
        //Fill the bad list with empty results for every module
        tkmapbad->fillc((*ih).first,255,255,255);
      }
      if((((*ih).second).first) < 100 ) {
        cout << "Module " << (*ih).first << " layer " << i << " is under occupancy at " << (((*ih).second).first) << endl;
      }
      //Put any module into the TKMap
      //Should call module ID, and then 1- efficiency for that module
      //if((*ih).first == 369137820) {
      //  cout << "Module 369137820 has 1-eff of " << 1.-myeff << endl;
	//cout << "Which is " << ((*ih).second).second << "/" << ((*ih).second).first << endl;
      //}
      tkmap->fill((*ih).first,1.-myeff);
      //Find the total number of hits in the module
      layertotal[i] += int(((*ih).second).first);
      layerfound[i] += int(((*ih).second).second);
    }
  }
  tkmap->save(true, 0, 0, "SiStripHitEffTKMap.png");
  tkmapbad->save(true, 0, 0, "SiStripHitEffTKMapBad.png");
  cout << "Finished TKMap Generation\n";
}

void SiStripHitEffFromCalibTree::makeSQLite() {
  //Generate the SQLite file for use in the Database of the bad modules!
  cout << "Entering SQLite file generation!\n";
  std::vector<unsigned int> BadStripList;
  unsigned short NStrips;
  unsigned int id1;
  SiStripQuality* pQuality = new SiStripQuality;
  //This is the list of the bad strips, use to mask out entire APVs
  //Now simply go through the bad hit list and mask out things that
  //are bad!
  map< unsigned int, double >::const_iterator it;
  for(it = BadModules.begin(); it != BadModules.end(); it++) {
    //We need to figure out how many strips are in this particular module
    //To Mask correctly!
    NStrips=reader->getNumberOfApvsAndStripLength((*it).first).first*128;
    cout << "Number of strips module " << (*it).first << " is " << NStrips << endl;
    BadStripList.push_back(pQuality->encode(0,NStrips,0));
    //Now compact into a single bad module
    id1=(unsigned int)(*it).first;
    cout << "ID1 shoudl match list of modules above " << id1 << endl;
    quality_->compact(id1,BadStripList);
    SiStripQuality::Range range(BadStripList.begin(),BadStripList.end());
    quality_->put(id1,range);
    BadStripList.clear();
  }
  //Fill all the bad components now
  quality_->fillBadComponents();
}

void SiStripHitEffFromCalibTree::totalStatistics() {
  //Calculate the statistics by layer
  int totalfound = 0;
  int totaltotal = 0;
  double layereff;
  for(Long_t i=1; i<=22; i++) {
    layereff = double(layerfound[i])/double(layertotal[i]);
    cout << "Layer " << i << " has total efficiency " << layereff << " " << layerfound[i] << "/" << layertotal[i] << endl;
    totalfound += layerfound[i];
    totaltotal += layertotal[i];
  }
  cout << "The total efficiency is " << double(totalfound)/double(totaltotal) << endl;
}

void SiStripHitEffFromCalibTree::makeSummary() {
  //setTDRStyle();
  
  int nLayers = 34;
  
  TH1F *found = fs->make<TH1F>("found","found",nLayers+1,0,nLayers+1);
  TH1F *all = fs->make<TH1F>("all","all",nLayers+1,0,nLayers+1);
  TH1F *found2 = fs->make<TH1F>("found2","found2",nLayers+1,0,nLayers+1);
  TH1F *all2 = fs->make<TH1F>("all2","all2",nLayers+1,0,nLayers+1);
  // first bin only to keep real data off the y axis so set to -1
  found->SetBinContent(0,-1);
  all->SetBinContent(0,1);
  
  TCanvas *c7 =new TCanvas("c7"," test ",10,10,800,600);
  c7->SetFillColor(0);
  c7->SetGrid();

  for (Long_t i=1; i< nLayers+1; ++i) {
    if (i==10) i++;
    if (i==25) i++;
    if (i==34) break;

    cout << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i] << "    B = " << goodlayertotal[i] << endl;
    if (goodlayertotal[i] > 5) {
      found->SetBinContent(i,goodlayerfound[i]);
      all->SetBinContent(i,goodlayertotal[i]);
    } else {
      found->SetBinContent(i,0);
      all->SetBinContent(i,10);
    }
    
    cout << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] << "    B = " << alllayertotal[i] << endl;
    if (alllayertotal[i] > 5) {
      found2->SetBinContent(i,alllayerfound[i]);
      all2->SetBinContent(i,alllayertotal[i]);
    } else {
      found2->SetBinContent(i,0);
      all2->SetBinContent(i,10);
    }

  }

  found->Sumw2();
  all->Sumw2();

  found2->Sumw2();
  all2->Sumw2();

  TGraphAsymmErrors *gr = new TGraphAsymmErrors(nLayers+1);
  gr->BayesDivide(found,all); 

  TGraphAsymmErrors *gr2 = new TGraphAsymmErrors(nLayers+1);
  gr2->BayesDivide(found2,all2);

  for(int j = 0; j<nLayers+1; j++){
    gr->SetPointError(j, 0., 0., gr->GetErrorYlow(j),gr->GetErrorYhigh(j) );
    gr2->SetPointError(j, 0., 0., gr2->GetErrorYlow(j),gr2->GetErrorYhigh(j) );
  }

  gr->GetXaxis()->SetLimits(0,nLayers);
  gr->SetMarkerColor(2);
  gr->SetMarkerSize(1.2);
  gr->SetLineColor(2);
  gr->SetLineWidth(4);
  gr->SetMarkerStyle(20);
  gr->SetMinimum(0.90);
  gr->SetMaximum(1.001);
  gr->GetYaxis()->SetTitle("Efficiency");

  gr2->GetXaxis()->SetLimits(0,nLayers);
  gr2->SetMarkerColor(1);
  gr2->SetMarkerSize(1.2);
  gr2->SetLineColor(1);
  gr2->SetLineWidth(4);
  gr2->SetMarkerStyle(21);
  gr2->SetMinimum(0.90);
  gr2->SetMaximum(1.001);
  gr2->GetYaxis()->SetTitle("Efficiency");
  //cout << "starting labels" << endl;
  //for ( int k=1; k<nLayers+1; k++) {
  for ( Long_t k=1; k<nLayers+1; k++) {
    if (k==10) k++;
    if (k==25) k++;
    if (k==34) break;
    TString label;
    if (k<5) {
      label = TString("TIB  ") + k;
    } else if (k>4&&k<11) {
      label = TString("TOB  ")+(k-4);
    } else if (k>10&&k<14) {
      label = TString("TID- ")+(k-10);
    } else if (k>13&&k<17) {
      label = TString("TID+ ")+(k-13);
    } else if (k>16&&k<26) {
      label = TString("TEC- ")+(k-16);
    } else if (k>25) {
      label = TString("TEC+ ")+(k-25);
    }
    gr->GetXaxis()->SetBinLabel(((k+1)*100)/(nLayers)-2,label);
    gr2->GetXaxis()->SetBinLabel(((k+1)*100)/(nLayers)-2,label);
  }
  
  gr->Draw("AP");
  gr->GetXaxis()->SetNdivisions(36);

  c7->cd();
  TPad *overlay = new TPad("overlay","",0,0,1,1);
  overlay->SetFillStyle(4000);
  overlay->SetFillColor(0);
  overlay->SetFrameFillStyle(4000);
  overlay->Draw("same");
  overlay->cd();
  gr2->Draw("AP");

  TLegend *leg = new TLegend(0.70,0.20,0.92,0.39);
  leg->AddEntry(gr,"Good Modules","p");
  leg->AddEntry(gr2,"All Modules","p");
  leg->SetTextSize(0.020);
  leg->SetFillColor(0);
  leg->Draw("same");
  
  c7->SaveAs("Summary.png");
}

SiStripBadStrip* SiStripHitEffFromCalibTree::getNewObject() {
  //Need this for a Condition DB Writer
  //Initialize a return variable
  SiStripBadStrip* obj=new SiStripBadStrip();
  
  SiStripBadStrip::RegistryIterator rIter=quality_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rIterEnd=quality_->getRegistryVectorEnd();
  
  for(;rIter!=rIterEnd;++rIter){
    SiStripBadStrip::Range range(quality_->getDataVectorBegin()+rIter->ibegin,quality_->getDataVectorBegin()+rIter->iend);
    if ( ! obj->put(rIter->detid,range) )
    edm::LogError("SiStripHitEffFromCalibTree")<<"[SiStripHitEffFromCalibTree::getNewObject] detid already exists"<<std::endl;
  }
  
  return obj;
}

float SiStripHitEffFromCalibTree::calcPhi(float x, float y) {
  float phi = 0;
  float Pi = 3.14159;
  if((x>=0)&&(y>=0)) phi = atan(y/x);
  else if((x>=0)&&(y<=0)) phi = atan(y/x) + 2*Pi;
  else if((x<=0)&&(y>=0)) phi = atan(y/x) + Pi;
  else phi = atan(y/x) + Pi;
  phi = phi*180.0/Pi;

  return phi;
} 

void SiStripHitEffFromCalibTree::SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC, std::stringstream ssV[4][19], int NBadComponent[4][19][4]){
 
  int napv=reader->getNumberOfApvsAndStripLength(BC.detid).first;
 
  ssV[i][component] << "\n\t\t " 
		    << BC.detid 
		    << " \t " << BC.BadModule << " \t " 
		    << ( (BC.BadFibers)&0x1 ) << " ";
  if (napv==4)
    ssV[i][component] << "x " <<( (BC.BadFibers>>1)&0x1 );
   
  if (napv==6)
    ssV[i][component] << ( (BC.BadFibers>>1)&0x1 ) << " "
		      << ( (BC.BadFibers>>2)&0x1 );
    ssV[i][component] << " \t " 
		      << ( (BC.BadApvs)&0x1 ) << " " 
		      << ( (BC.BadApvs>>1)&0x1 ) << " ";
  if (napv==4) 
    ssV[i][component] << "x x " << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 );
  if (napv==6) 
    ssV[i][component] << ( (BC.BadApvs>>2)&0x1 ) << " " 
		      << ( (BC.BadApvs>>3)&0x1 ) << " " 
		      << ( (BC.BadApvs>>4)&0x1 ) << " " 
		      << ( (BC.BadApvs>>5)&0x1 ) << " "; 
 
  if (BC.BadApvs){
    NBadComponent[i][0][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
    NBadComponent[i][component][2]+= ( (BC.BadApvs>>5)&0x1 )+ ( (BC.BadApvs>>4)&0x1 ) + ( (BC.BadApvs>>3)&0x1 ) + 
      ( (BC.BadApvs>>2)&0x1 )+ ( (BC.BadApvs>>1)&0x1 ) + ( (BC.BadApvs)&0x1 );
  }
  if (BC.BadFibers){ 
    NBadComponent[i][0][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
    NBadComponent[i][component][1]+= ( (BC.BadFibers>>2)&0x1 )+ ( (BC.BadFibers>>1)&0x1 ) + ( (BC.BadFibers)&0x1 );
  }   
  if (BC.BadModule){
    NBadComponent[i][0][0]++;
    NBadComponent[i][component][0]++;
  }
}

DEFINE_FWK_MODULE(SiStripHitEffFromCalibTree);
