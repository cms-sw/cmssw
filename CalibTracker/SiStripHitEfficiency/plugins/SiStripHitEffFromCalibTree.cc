//Original Author:  Christopher Edelmaier
//        Created:  Feb. 11, 2010
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

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
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
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
#include "TStyle.h"
#include "TLeaf.h"
#include "TGaxis.h"
#include "TGraphAsymmErrors.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TEfficiency.h"

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
    ~SiStripHitEffFromCalibTree() override;

  private:
    virtual void algoBeginJob();
    void algoEndJob() override;
    void algoAnalyze(const edm::Event& e, const edm::EventSetup& c) override;
    void SetBadComponents(int i, int component,SiStripQuality::BadComponent& BC, std::stringstream ssV[4][19], int NBadComponent[4][19][4]);
    void makeTKMap();
    void makeHotColdMaps();
    void makeSQLite();
    void totalStatistics();
    void makeSummary();
    void makeSummaryVsBx();
	void ComputeEff(vector< TH1F* > &vhfound, vector< TH1F* > &vhtotal, string name);
    void makeSummaryVsLumi();
    void makeSummaryVsCM();
    TString GetLayerName(Long_t k);
    TString GetLayerSideName(Long_t k);
    float calcPhi(float x, float y);

    edm::Service<TFileService> fs;
    SiStripDetInfoFileReader* reader;
    edm::FileInPath FileInPath_;
    SiStripQuality* quality_;
    SiStripBadStrip* getNewObject() override;
    
    TTree* CalibTree;
    vector<string> CalibTreeFilenames; 
    float threshold;
    unsigned int nModsMin;
    unsigned int doSummary;
    string _badModulesFile;
    unsigned int _clusterMatchingMethod;
    float _ResXSig;
    float _clusterTrajDist;
    float _stripsApvEdge;
    unsigned int _bunchx;
    unsigned int _spaceBetweenTrains;
	bool _useCM;
    bool _showEndcapSides;
    bool  _showRings;
    bool  _showTOB6TEC9;
	bool _showOnlyGoodModules;
    float _tkMapMin;
    float _effPlotMin;
    TString _title;
	
    unsigned int nTEClayers;
	
    vector<hit> hits[23];
    vector<TH2F*> HotColdMaps;
    map< unsigned int, pair< unsigned int, unsigned int> > modCounter[23];
    TrackerMap *tkmap;
    TrackerMap *tkmapbad;
    TrackerMap *tkmapeff;
    TrackerMap *tkmapnum;
    TrackerMap *tkmapden;
    int layerfound[23];
    int layertotal[23];
    map< unsigned int, vector<int> > layerfound_perBx;
    map< unsigned int, vector<int> > layertotal_perBx;
	vector< TH1F* > layerfound_vsLumi;
	vector< TH1F* > layertotal_vsLumi;
	vector< TH1F* > layerfound_vsPU;
	vector< TH1F* > layertotal_vsPU;
	vector< TH1F* > layerfound_vsCM;
	vector< TH1F* > layertotal_vsCM;
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
  CalibTreeFilenames = conf.getUntrackedParameter<vector<std::string> >("CalibTreeFilenames");
  threshold = conf.getParameter<double>("Threshold");
  nModsMin = conf.getParameter<int>("nModsMin");
  doSummary = conf.getParameter<int>("doSummary");
  _badModulesFile = conf.getUntrackedParameter<std::string>("BadModulesFile", ""); 
  _clusterMatchingMethod = conf.getUntrackedParameter<int>("ClusterMatchingMethod",0);
  _ResXSig = conf.getUntrackedParameter<double>("ResXSig",-1);
  _clusterTrajDist = conf.getUntrackedParameter<double>("ClusterTrajDist",64.0);
  _stripsApvEdge = conf.getUntrackedParameter<double>("StripsApvEdge",10.0);
  _bunchx = conf.getUntrackedParameter<int>("BunchCrossing",0);
  _spaceBetweenTrains = conf.getUntrackedParameter<int>("SpaceBetweenTrains",25);
  _useCM = conf.getUntrackedParameter<bool>("UseCommonMode",false);
  _showEndcapSides = conf.getUntrackedParameter<bool>("ShowEndcapSides",true);
  _showRings = conf.getUntrackedParameter<bool>("ShowRings",false);
  _showTOB6TEC9 = conf.getUntrackedParameter<bool>("ShowTOB6TEC9",false);
  _showOnlyGoodModules = conf.getUntrackedParameter<bool>("ShowOnlyGoodModules",false);
  _tkMapMin = conf.getUntrackedParameter<double>("TkMapMin",0.9);
  _effPlotMin = conf.getUntrackedParameter<double>("EffPlotMin",0.9);
  _title = conf.getParameter<std::string>("Title"); 
  reader = new SiStripDetInfoFileReader(FileInPath_.fullPath());
  
  nTEClayers = 9; // number of wheels
  if(_showRings) nTEClayers = 7; // number of rings
  
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

  edm::ESHandle<TrackerGeometry> tracker;
  c.get<TrackerDigiGeometryRecord>().get( tracker );
  const TrackerGeometry * tkgeom=&(* tracker);
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  c.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // read bad modules to mask
  ifstream badModules_file;
  set<uint32_t> badModules_list;
  if(_badModulesFile!="") {
	badModules_file.open(_badModulesFile.c_str());
	uint32_t badmodule_detid;
	int mods, fiber1, fiber2, fiber3;
	if(badModules_file.is_open()) {
      string line;
	  while ( getline (badModules_file,line) ) {
		if(badModules_file.eof()) continue;
		stringstream ss(line);
		ss >> badmodule_detid >> mods >> fiber1 >> fiber2 >> fiber3;
		if(badmodule_detid!=0 && mods==1 && (fiber1==1 || fiber2==1 || fiber3==1) )
	      badModules_list.insert(badmodule_detid);
	  }
      badModules_file.close();
	}
  }
  if(!badModules_list.empty()) cout<<"Remove additionnal bad modules from the analysis: "<<endl;
  set<uint32_t>::iterator itBadMod;
  for (itBadMod=badModules_list.begin(); itBadMod!=badModules_list.end(); ++itBadMod) 
    cout<<" "<<*itBadMod<<endl;


  // initialze counters and histos
  for(int l=0; l < 35; l++) {
    goodlayertotal[l] = 0;
    goodlayerfound[l] = 0;
    alllayertotal[l] = 0;
    alllayerfound[l] = 0;
  }

  TH1F* resolutionPlots[23];
  for(Long_t ilayer = 0; ilayer <23; ilayer++) {
    resolutionPlots[ilayer] = fs->make<TH1F>(Form("resol_layer_%i",(int)(ilayer)),GetLayerName(ilayer),125,-125,125);
    resolutionPlots[ilayer]->GetXaxis()->SetTitle("trajX-clusX [strip unit]");

	layerfound_vsLumi.push_back( fs->make<TH1F>(Form("layerfound_vsLumi_layer_%i",(int)(ilayer)),GetLayerName(ilayer),60,0,15000));
	layertotal_vsLumi.push_back( fs->make<TH1F>(Form("layertotal_vsLumi_layer_%i",(int)(ilayer)),GetLayerName(ilayer),60,0,15000));
	layerfound_vsPU.push_back( fs->make<TH1F>(Form("layerfound_vsPU_layer_%i",(int)(ilayer)),GetLayerName(ilayer),30,0,60));
	layertotal_vsPU.push_back( fs->make<TH1F>(Form("layertotal_vsPU_layer_%i",(int)(ilayer)),GetLayerName(ilayer),30,0,60));

	if(_useCM) {
	  layerfound_vsCM.push_back( fs->make<TH1F>(Form("layerfound_vsCM_layer_%i",(int)(ilayer)),GetLayerName(ilayer),20,0,400));
	  layertotal_vsCM.push_back( fs->make<TH1F>(Form("layertotal_vsCM_layer_%i",(int)(ilayer)),GetLayerName(ilayer),20,0,400));
	}
  }

  cout << "A module is bad if efficiency < " << threshold << " and has at least " << nModsMin << " nModsMin." << endl;


  //Open the ROOT Calib Tree
  for( unsigned int ifile=0; ifile < CalibTreeFilenames.size(); ifile++) {

    cout<<"Loading file: "<<CalibTreeFilenames[ifile]<<endl;
	TFile* CalibTreeFile = TFile::Open(CalibTreeFilenames[ifile].c_str(),"READ");
	CalibTreeFile->cd("anEff");
	CalibTree = (TTree*)(gDirectory->Get("traj"));
	
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
	TLeaf* TrajLocXLf = CalibTree->GetLeaf("TrajLocX");
	TLeaf* TrajLocYLf = CalibTree->GetLeaf("TrajLocY");
	TLeaf* ClusterLocXLf = CalibTree->GetLeaf("ClusterLocX");
	TLeaf* BunchLf = CalibTree->GetLeaf("bunchx");
	TLeaf* InstLumiLf = CalibTree->GetLeaf("instLumi");
	TLeaf* PULf = CalibTree->GetLeaf("PU");
	TLeaf* CMLf = nullptr;
	if(_useCM) CMLf = CalibTree->GetLeaf("commonMode");

	int nevents = CalibTree->GetEntries();
	cout << "Successfully loaded analyze function with " << nevents << " events!\n";


	//Loop through all of the events
	for(int j =0; j < nevents; j++) {
      CalibTree->GetEntry(j);
      unsigned int isBad = (unsigned int)BadLf->GetValue();
      unsigned int quality = (unsigned int)sistripLf->GetValue();
      unsigned int id = (unsigned int)idLf->GetValue();
      unsigned int accept = (unsigned int)acceptLf->GetValue();
      unsigned int layer_wheel = (unsigned int)layerLf->GetValue();
      unsigned int layer = layer_wheel;
      if(_showRings && layer >10) { // use rings instead of wheels
    	if(layer<14) layer = 10 + ((id>>9)&0x3); //TID   3 disks and also 3 rings -> use the same container
    	else layer = 13 + ((id>>5)&0x7); //TEC
      }
      unsigned int nHits = (unsigned int)nHitsLf->GetValue();
      double x = xLf->GetValue();
      double y = yLf->GetValue();
      double z = zLf->GetValue();
      double resxsig = ResXSigLf->GetValue();
      double TrajLocX = TrajLocXLf->GetValue();
      double TrajLocY = TrajLocYLf->GetValue();
      double ClusterLocX = ClusterLocXLf->GetValue();
      double TrajLocXMid;
      double stripTrajMid;
      double stripCluster;
      bool badquality = false;
      unsigned int bx = (unsigned int)BunchLf->GetValue();
      if(_bunchx > 0 && _bunchx != bx) continue;
	  double instLumi = 0;
	  if(InstLumiLf!=nullptr) instLumi = InstLumiLf->GetValue();
	  double PU = 0;
	  if(PULf!=nullptr) PU = PULf->GetValue();
	  int CM = -100;
	  if(_useCM) CM = CMLf->GetValue();

      //We have two things we want to do, both an XY color plot, and the efficiency measurement
      //First, ignore anything that isn't in acceptance and isn't good quality

      //if(quality == 1 || accept != 1 || nHits < 8) continue;
      if(accept != 1 || nHits < 8) continue;
      if(quality == 1) badquality = true;

      // don't compute efficiencies in modules from TOB6 and TEC9
      if(!_showTOB6TEC9 && (layer_wheel==10 || layer_wheel==22)) continue; 

	  // don't use bad modules given in the bad module list
	  itBadMod = badModules_list.find(id);
	  if(itBadMod!=badModules_list.end()) continue;


      //Now that we have a good event, we need to look at if we expected it or not, and the location
      //if we didn't
      //Fill the missing hit information first
      bool badflag = false;

	  // By default uses the old matching method
      if(_ResXSig < 0) {
    	if(isBad == 1) badflag = true; // isBad set to false in the tree when resxsig<999.0
      }
      else {
    	if(isBad == 1 || resxsig > _ResXSig) badflag = true;
      }

	  // Conversion of positions in strip unit
      int   nstrips = -9; 
      float Pitch   = -9.0; 

      if (resxsig==1000.0) { // special treatment, no GeomDetUnit associated in some cases when no cluster found
    	Pitch = 0.0205;  // maximum
    	nstrips = 768;  // maximum
    	stripTrajMid   =    TrajLocX/Pitch + nstrips/2.0 ;      
    	stripCluster   = ClusterLocX/Pitch + nstrips/2.0 ;
      }
      else {
		  DetId ClusterDetId(id);
		  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tkgeom->idToDetUnit(ClusterDetId);
		  const StripTopology& Topo  = stripdet->specificTopology();
		  nstrips = Topo.nstrips();
		  Pitch = stripdet->surface().bounds().width() / Topo.nstrips();
		  stripTrajMid   =    TrajLocX/Pitch + nstrips/2.0 ; //layer01->10
		  stripCluster   = ClusterLocX/Pitch + nstrips/2.0 ;

		  // For trapezoidal modules: extrapolation of x trajectory position to the y middle of the module
		  //  for correct comparison with cluster position
    	  float hbedge   = 0;
    	  float htedge   = 0;
    	  float hapoth   = 0;
		  if(layer>=11) {
			const BoundPlane& plane = stripdet->surface();
			const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
			std::array<const float, 4> const & parameters = (*trapezoidalBounds).parameters(); 
			hbedge         = parameters[0];
			htedge         = parameters[1];
			hapoth         = parameters[3];
			TrajLocXMid = TrajLocX / (1 + (htedge-hbedge)*TrajLocY/(htedge+hbedge)/hapoth) ; // radialy extrapolated x loc position at middle  
			stripTrajMid   =    TrajLocXMid/Pitch + nstrips/2.0 ;
		  }
	  }


	  if(!badquality && layer<23) {
		if(resxsig!=1000.0) resolutionPlots[layer]->Fill(stripTrajMid-stripCluster);
		else resolutionPlots[layer]->Fill(1000);
	  }


	  // New matching methods
      int   tapv   = -9;
      int   capv    = -9;
	  float stripInAPV = 64.;

      if ( _clusterMatchingMethod >=1 ) { 
    	badflag = false;  // reset 
    	if(resxsig == 1000.0) { // default value when no cluster found in the module
          badflag = true; // consider the module inefficient in this case
    	}
    	else{
		  if (_clusterMatchingMethod==2 || _clusterMatchingMethod==4) { // check the distance between cluster and trajectory position
			if ( abs(stripCluster - stripTrajMid) > _clusterTrajDist ) badflag = true;
		  }
		  if (_clusterMatchingMethod==3 || _clusterMatchingMethod==4) { // cluster and traj have to be in the same APV (don't take edges into accounts)
			tapv = (int) stripTrajMid/128;
			capv = (int) stripCluster/128;
			stripInAPV = stripTrajMid-tapv*128;

			if(stripInAPV<_stripsApvEdge || stripInAPV>128-_stripsApvEdge) continue;
			if(tapv != capv) badflag = true;
		  } 	
    	}
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

		if(layerfound_perBx.find(bx)==layerfound_perBx.end()) {
	      layerfound_perBx[bx] = vector<int>(23, 0);
	      layertotal_perBx[bx] = vector<int>(23, 0);
		}
		if(!badflag) layerfound_perBx[bx][layer]++;
		layertotal_perBx[bx][layer]++;

		if(!badflag) layerfound_vsLumi[layer]->Fill(instLumi);
		layertotal_vsLumi[layer]->Fill(instLumi);
		if(!badflag) layerfound_vsPU[layer]->Fill(PU);
		layertotal_vsPU[layer]->Fill(PU);

		if(_useCM){
	      if(!badflag) layerfound_vsCM[layer]->Fill(CM);
		  layertotal_vsCM[layer]->Fill(CM);
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
		if(!badflag) goodlayerfound[layer+3+nTEClayers]++;
        	goodlayertotal[layer+3+nTEClayers]++;
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
          if(!badflag) alllayerfound[layer+3+nTEClayers]++;
          alllayertotal[layer+3+nTEClayers]++;
    	}
      }  
      //At this point, both of our maps are loaded with the correct information
    }
  }// go to next CalibTreeFile

  makeHotColdMaps();
  makeTKMap();
  makeSQLite();
  totalStatistics();
  makeSummary();
  makeSummaryVsBx();
  makeSummaryVsLumi();
  if(_useCM) makeSummaryVsCM();
  
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

  // store also bad modules in log file
  ofstream badModules;
  badModules.open("BadModules.log");
  badModules << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers Apvs\n----------------------------------------------------------------";
  for (int i=1;i<5;++i)
    badModules << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  badModules << "\n";
  for (int i=1;i<4;++i)
    badModules << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i=4;i<7;++i)
    badModules << "\nTID- Disk " << i-3 << " :" << ssV[1][i].str();
  badModules << "\n";
  for (int i=1;i<7;++i)
    badModules << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  badModules << "\n";
  for (int i=1;i<10;++i)
    badModules << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i=10;i<19;++i)
    badModules << "\nTEC- Disk " << i-9 << " :" << ssV[3][i].str();
  badModules.close();
  
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
  tkmapeff = new TrackerMap(_title.Data());
  tkmapnum = new TrackerMap(" Detector numerator   ");
  tkmapden = new TrackerMap(" Detector denominator ");
  
  double myeff, mynum, myden;
  
  for(Long_t i = 1; i <= 22; i++) {
	//Loop over every layer, extracting the information from
	//the map of the efficiencies
	layertotal[i] = 0;
	layerfound[i] = 0;
    map<unsigned int, pair<unsigned int, unsigned int> >::const_iterator ih;
    for( ih = modCounter[i].begin(); ih != modCounter[i].end(); ih++) {
      //We should be in the layer in question, and looping over all of the modules in said layer
      //Generate the list for the TKmap, and the bad module list
	  mynum = (double)(((*ih).second).second);
	  myden = (double)(((*ih).second).first);
      if(myden>0) myeff = mynum/myden;
	  else myeff=0;
      if ( (myden >= nModsMin) && (myeff < threshold) ) {
        //We have a bad module, put it in the list!
		BadModules[(*ih).first] = myeff;
		tkmapbad->fillc((*ih).first,255,0,0);
		cout << "Layer " << i << " module " << (*ih).first << " efficiency " << myeff << " " << (((*ih).second).second) << "/" << (((*ih).second).first) << endl;
      }
      else {
        //Fill the bad list with empty results for every module
        tkmapbad->fillc((*ih).first,255,255,255);
      }
      if(myden < 50 ) {
        cout << "Module " << (*ih).first << " layer " << i << " is under occupancy at " << (((*ih).second).first) << endl;
      }
      //Put any module into the TKMap
      //Should call module ID, and then 1- efficiency for that module
      //if((*ih).first == 369137820) {
      //  cout << "Module 369137820 has 1-eff of " << 1.-myeff << endl;
	//cout << "Which is " << ((*ih).second).second << "/" << ((*ih).second).first << endl;
      //}
      tkmap->fill((*ih).first,1.-myeff);
      tkmapeff->fill((*ih).first,myeff);
      tkmapnum->fill((*ih).first,mynum);
      tkmapden->fill((*ih).first,myden);
      //Find the total number of hits in the module
      layertotal[i] += int(myden);
      layerfound[i] += int(mynum);
    }
  }
  tkmap->save(true, 0, 0, "SiStripHitEffTKMap.png");
  tkmapbad->save(true, 0, 0, "SiStripHitEffTKMapBad.png");
  tkmapeff->save(true, _tkMapMin, 1., "SiStripHitEffTKMapEff.png");
  tkmapnum->save(true, 0, 0, "SiStripHitEffTKMapNum.png");
  tkmapden->save(true, 0, 0, "SiStripHitEffTKMapDen.png");
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
  int subdetfound[5];
  int subdettotal[5];
  
  for(Long_t i=1; i<5; i++) {subdetfound[i]=0; subdettotal[i]=0;}
  
  for(Long_t i=1; i<=22; i++) {
    layereff = double(layerfound[i])/double(layertotal[i]);
    cout << "Layer " << i << " has total efficiency " << layereff << " " << layerfound[i] << "/" << layertotal[i] << endl;
    totalfound += layerfound[i];
    totaltotal += layertotal[i];
	if(i<5) {subdetfound[1]+=layerfound[i]; subdettotal[1]+=layertotal[i];}
	if(i>=5 && i<11) {subdetfound[2]+=layerfound[i]; subdettotal[2]+=layertotal[i];}
	if(i>=11 && i<14) {subdetfound[3]+=layerfound[i]; subdettotal[3]+=layertotal[i];}
	if(i>=14) {subdetfound[4]+=layerfound[i]; subdettotal[4]+=layertotal[i];}
	
  }
  
  cout << "The total efficiency is " << double(totalfound)/double(totaltotal) << endl;
  cout << "      TIB: " << double(subdetfound[1])/subdettotal[1] << endl;
  cout << "      TOB: " << double(subdetfound[2])/subdettotal[2] << endl;
  cout << "      TID: " << double(subdetfound[3])/subdettotal[3] << endl;
  cout << "      TEC: " << double(subdetfound[4])/subdettotal[4] << endl;
}

void SiStripHitEffFromCalibTree::makeSummary() {
  //setTDRStyle();
  
  int nLayers = 34;
  if(_showRings) nLayers = 30;
  if(!_showEndcapSides) {
    if(!_showRings) nLayers=22;
    else nLayers=20;
  }
  
  TH1F *found = fs->make<TH1F>("found","found",nLayers+1,0,nLayers+1);
  TH1F *all = fs->make<TH1F>("all","all",nLayers+1,0,nLayers+1);
  TH1F *found2 = fs->make<TH1F>("found2","found2",nLayers+1,0,nLayers+1);
  TH1F *all2 = fs->make<TH1F>("all2","all2",nLayers+1,0,nLayers+1);
  // first bin only to keep real data off the y axis so set to -1
  found->SetBinContent(0,-1);
  all->SetBinContent(0,1);
  
  // new ROOT version: TGraph::Divide don't handle null or negative values
  for (Long_t i=1; i< nLayers+2; ++i) {
  	found->SetBinContent(i,1e-6);
	all->SetBinContent(i,1);
  	found2->SetBinContent(i,1e-6);
	all2->SetBinContent(i,1);
  }
  
  TCanvas *c7 =new TCanvas("c7"," test ",10,10,800,600);
  c7->SetFillColor(0);
  c7->SetGrid();

  int nLayers_max=nLayers+1; // barrel+endcap
  if(!_showEndcapSides) nLayers_max=11; // barrel 
  for (Long_t i=1; i< nLayers_max; ++i) {
    cout << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i] << "    B = " << goodlayertotal[i] << endl;
    if (goodlayertotal[i] > 5) {
      found->SetBinContent(i,goodlayerfound[i]);
      all->SetBinContent(i,goodlayertotal[i]);
    }
    
    cout << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] << "    B = " << alllayertotal[i] << endl;
    if (alllayertotal[i] > 5) {
      found2->SetBinContent(i,alllayerfound[i]);
      all2->SetBinContent(i,alllayertotal[i]);
    }

  }

  // endcap - merging sides
  if(!_showEndcapSides) {
    for (Long_t i=11; i< 14; ++i) { // TID disks
      cout << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i]+goodlayerfound[i+3] << "    B = " << goodlayertotal[i]+goodlayertotal[i+3] << endl;
      if (goodlayertotal[i]+goodlayertotal[i+3] > 5) {
        found->SetBinContent(i,goodlayerfound[i]+goodlayerfound[i+3]);
        all->SetBinContent(i,goodlayertotal[i]+goodlayertotal[i+3]);
      }
      cout << "Filling all modules layer " << i << ":  S = " << alllayerfound[i]+alllayerfound[i+3] << "    B = " << alllayertotal[i]+alllayertotal[i+3] << endl;
      if (alllayertotal[i]+alllayertotal[i+3] > 5) {
        found2->SetBinContent(i,alllayerfound[i]+alllayerfound[i+3]);
        all2->SetBinContent(i,alllayertotal[i]+alllayertotal[i+3]);
      }
	}
    for (Long_t i=17; i< 17+nTEClayers; ++i) { // TEC disks
      cout << "Fill only good modules layer " << i-3 << ":  S = " << goodlayerfound[i]+goodlayerfound[i+nTEClayers] << "    B = " << goodlayertotal[i]+goodlayertotal[i+nTEClayers] << endl;
      if (goodlayertotal[i]+goodlayertotal[i+nTEClayers] > 5) {
        found->SetBinContent(i-3,goodlayerfound[i]+goodlayerfound[i+nTEClayers]);
        all->SetBinContent(i-3,goodlayertotal[i]+goodlayertotal[i+nTEClayers]);
      }
      cout << "Filling all modules layer " << i-3 << ":  S = " << alllayerfound[i]+alllayerfound[i+nTEClayers] << "    B = " << alllayertotal[i]+alllayertotal[i+nTEClayers] << endl;
      if (alllayertotal[i]+alllayertotal[i+nTEClayers] > 5) {
        found2->SetBinContent(i-3,alllayerfound[i]+alllayerfound[i+nTEClayers]);
        all2->SetBinContent(i-3,alllayertotal[i]+alllayertotal[i+nTEClayers]);
      }
	}
  }

  found->Sumw2();
  all->Sumw2();

  found2->Sumw2();
  all2->Sumw2();

  TGraphAsymmErrors *gr = fs->make<TGraphAsymmErrors>(nLayers+1);
  gr->SetName("eff_good");
  gr->BayesDivide(found,all); 

  TGraphAsymmErrors *gr2 = fs->make<TGraphAsymmErrors>(nLayers+1);
  gr2->SetName("eff_all");
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
  gr->SetMinimum(_effPlotMin);
  gr->SetMaximum(1.001);
  gr->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleBorderSize(0);
  gr->SetTitle(_title);

  gr2->GetXaxis()->SetLimits(0,nLayers);
  gr2->SetMarkerColor(1);
  gr2->SetMarkerSize(1.2);
  gr2->SetLineColor(1);
  gr2->SetLineWidth(4);
  gr2->SetMarkerStyle(21);
  gr2->SetMinimum(_effPlotMin);
  gr2->SetMaximum(1.001);
  gr2->GetYaxis()->SetTitle("Efficiency");
  gr2->SetTitle(_title);

  for ( Long_t k=1; k<nLayers+1; k++) {
        TString label;
	if(_showEndcapSides) label = GetLayerSideName(k);
	else label = GetLayerName(k);
	if(!_showTOB6TEC9) {
	  if(k==10) label="";
	  if(!_showRings && k==nLayers) label="";
	  if(!_showRings && _showEndcapSides && k==25) label="";
        }
	if(!_showRings) {
	  if(_showEndcapSides) {
            gr->GetXaxis()->SetBinLabel(((k+1)*100+2)/(nLayers)-4,label);
            gr2->GetXaxis()->SetBinLabel(((k+1)*100+2)/(nLayers)-4,label);
	  }
	  else {
            gr->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-6,label);
            gr2->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-6,label);
	  }
	}
	else {
	  if(_showEndcapSides) {
	    gr->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-4,label);
            gr2->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-4,label);
	  }
	  else {
	    gr->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-7,label);
            gr2->GetXaxis()->SetBinLabel((k+1)*100/(nLayers)-7,label);
	  }
	}
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
  if(!_showOnlyGoodModules) gr2->Draw("AP");

  TLegend *leg = new TLegend(0.70,0.27,0.88,0.40);
  leg->AddEntry(gr,"Good Modules","p");
  if(!_showOnlyGoodModules) leg->AddEntry(gr2,"All Modules","p");
  leg->SetTextSize(0.020);
  leg->SetFillColor(0);
  leg->Draw("same");
  
  c7->SaveAs("Summary.png");
}


void SiStripHitEffFromCalibTree::makeSummaryVsBx() {
  cout<<"Computing efficiency vs bx"<<endl;
  
  unsigned int nLayers = 22;
  if(_showRings) nLayers = 20;
  
  for(unsigned int ilayer=1; ilayer<nLayers; ilayer++) {
    TH1F *hfound = fs->make<TH1F>(Form("foundVsBx_layer%i", ilayer),Form("layer %i", ilayer),3565,0,3565);
    TH1F *htotal = fs->make<TH1F>(Form("totalVsBx_layer%i", ilayer),Form("layer %i", ilayer),3565,0,3565);
	
	for(unsigned int ibx=0; ibx<3566; ibx++){
	  hfound->SetBinContent(ibx, 1e-6);
	  htotal->SetBinContent(ibx, 1);
	}
	map<unsigned int, vector<int> >::iterator iterMapvsBx;
	for(iterMapvsBx=layerfound_perBx.begin(); iterMapvsBx!=layerfound_perBx.end(); ++iterMapvsBx) 
	  hfound->SetBinContent( iterMapvsBx->first, iterMapvsBx->second[ilayer]);
	for(iterMapvsBx=layertotal_perBx.begin(); iterMapvsBx!=layertotal_perBx.end(); ++iterMapvsBx)
	  if(iterMapvsBx->second[ilayer]>0) htotal->SetBinContent( iterMapvsBx->first, iterMapvsBx->second[ilayer]);
		
	hfound->Sumw2();
	htotal->Sumw2();
	
	TGraphAsymmErrors *geff = fs->make<TGraphAsymmErrors>(3564);
	geff->SetName(Form("effVsBx_layer%i", ilayer));
	geff->SetTitle("Hit Efficiency vs bx - "+GetLayerName(ilayer));
	geff->BayesDivide(hfound,htotal);
	
	//Average over trains
	TGraphAsymmErrors *geff_avg = fs->make<TGraphAsymmErrors>();
	geff_avg->SetName(Form("effVsBxAvg_layer%i", ilayer));
	geff_avg->SetTitle("Hit Efficiency vs bx - "+GetLayerName(ilayer));
	geff_avg->SetMarkerStyle(20);
	int ibx=0;
	int previous_bx=-80;
	int delta_bx=0;
	int nbx=0;
	int found=0;
	int total=0;
	double sum_bx=0;
	int ipt=0;
	float low, up, eff;
	int firstbx=0;
	for(iterMapvsBx=layertotal_perBx.begin(); iterMapvsBx!=layertotal_perBx.end(); ++iterMapvsBx){
	  ibx=iterMapvsBx->first;
	  delta_bx=ibx-previous_bx;
	  // consider a new train
	  if(delta_bx>(int)_spaceBetweenTrains && nbx>0 && total>0){
	    eff=found/(float)total;
	    //cout<<"new train "<<ipt<<" "<<sum_bx/nbx<<" "<<eff<<endl;
	    geff_avg->SetPoint(ipt, sum_bx/nbx, eff);
	    low = TEfficiency::Bayesian(total, found, .683, 1, 1, false);
	    up = TEfficiency::Bayesian(total, found, .683, 1, 1, true);
	    geff_avg->SetPointError(ipt, sum_bx/nbx-firstbx, previous_bx-sum_bx/nbx, eff-low, up-eff);
	    ipt++;
	    sum_bx=0;
	    found=0;
	    total=0;
	    nbx=0;
	    firstbx=ibx;
	  }
	  sum_bx+=ibx;
	  found+=hfound->GetBinContent(ibx);
	  total+=htotal->GetBinContent(ibx);
	  nbx++;
	  
	  previous_bx=ibx;
	}
	//last train
	eff=found/(float)total;
	//cout<<"new train "<<ipt<<" "<<sum_bx/nbx<<" "<<eff<<endl;
	geff_avg->SetPoint(ipt, sum_bx/nbx, eff);
	low = TEfficiency::Bayesian(total, found, .683, 1, 1, false);
	up = TEfficiency::Bayesian(total, found, .683, 1, 1, true);
	geff_avg->SetPointError(ipt, sum_bx/nbx-firstbx, previous_bx-sum_bx/nbx, eff-low, up-eff);
  }
}


TString SiStripHitEffFromCalibTree::GetLayerName(Long_t k) {

    TString layername="";
    TString ringlabel="D";
    if(_showRings) ringlabel="R";
    if (k>0 && k<5) {
      layername = TString("TIB L") + k;
    } else if (k>4 && k<11) {
      layername = TString("TOB L")+(k-4);
    } else if (k>10 && k<14) {
      layername = TString("TID ")+ringlabel+(k-10);
    } else if (k>13 && k<14+nTEClayers) {
      layername = TString("TEC ")+ringlabel+(k-13);
	}
   
    return layername;
}

void SiStripHitEffFromCalibTree::ComputeEff(vector< TH1F* > &vhfound, vector< TH1F* > &vhtotal, string name) {

  unsigned int nLayers = 22;
  if(_showRings) nLayers = 20;
  
  TH1F* hfound;
  TH1F* htotal;
  
  for(unsigned int ilayer=1; ilayer<nLayers; ilayer++) {
	
	hfound = vhfound[ilayer];
	htotal = vhtotal[ilayer];
	
	hfound->Sumw2();
	htotal->Sumw2();	
	
	// new ROOT version: TGraph::Divide don't handle null or negative values
	for (Long_t i=0; i< hfound->GetNbinsX()+1; ++i) {
	  if( hfound->GetBinContent(i)==0) hfound->SetBinContent(i,1e-6);
	  if( htotal->GetBinContent(i)==0) htotal->SetBinContent(i,1);
	}
		
	TGraphAsymmErrors *geff = fs->make<TGraphAsymmErrors>(hfound->GetNbinsX());
	geff->SetName(Form("%s_layer%i", name.c_str(), ilayer));
	geff->BayesDivide(hfound, htotal);
	if(name=="effVsLumi") geff->SetTitle("Hit Efficiency vs inst. lumi. - "+GetLayerName(ilayer));
	if(name=="effVsPU") geff->SetTitle("Hit Efficiency vs pileup - "+GetLayerName(ilayer));
	if(name=="effVsCM") geff->SetTitle("Hit Efficiency vs common Mode - "+GetLayerName(ilayer));
	geff->SetMarkerStyle(20);
	
  }

}

void SiStripHitEffFromCalibTree::makeSummaryVsLumi() {
  cout<<"Computing efficiency vs lumi"<<endl;
  
  unsigned int nLayers = 22;
  if(_showRings) nLayers = 20;
  unsigned int nLayersForAvg = 0;
  float layerLumi = 0;
  float layerPU = 0;
  float avgLumi = 0;
  float avgPU = 0;
  
  cout<<"Lumi summary:  (avg over trajectory measurements)"<<endl; 
  for(unsigned int ilayer=1; ilayer<nLayers; ilayer++) {
	layerLumi=layertotal_vsLumi[ilayer]->GetMean();
	layerPU=layertotal_vsPU[ilayer]->GetMean();
	//cout<<" layer "<<ilayer<<"  lumi: "<<layerLumi<<"  pu: "<<layerPU<<endl;	
	if(layerLumi!=0 && layerPU!=0) {
	  avgLumi+=layerLumi;
	  avgPU+=layerPU;
	  nLayersForAvg++;
	}
  }
  avgLumi/=nLayersForAvg;
  avgPU/=nLayersForAvg;
  cout<<"Avg conditions:   lumi :"<<avgLumi<<"  pu: "<<avgPU<<endl;
  
  ComputeEff(layerfound_vsLumi, layertotal_vsLumi, "effVsLumi");
  ComputeEff(layerfound_vsPU, layertotal_vsPU, "effVsPU");
  
}

void SiStripHitEffFromCalibTree::makeSummaryVsCM() {
  cout<<"Computing efficiency vs CM"<<endl;
  ComputeEff(layerfound_vsCM, layertotal_vsCM, "effVsCM");
}

TString SiStripHitEffFromCalibTree::GetLayerSideName(Long_t k) {

    TString layername="";
    TString ringlabel="D";
    if(_showRings) ringlabel="R";
    if (k>0 && k<5) {
      layername = TString("TIB L") + k;
    } else if (k>4&&k<11) {
      layername = TString("TOB L")+(k-4);
    } else if (k>10&&k<14) {
      layername = TString("TID- ")+ringlabel+(k-10);
    } else if (k>13&&k<17) {
      layername = TString("TID+ ")+ringlabel+(k-13);
    } else if (k>16&&k<17+nTEClayers) {
      layername = TString("TEC- ")+ringlabel+(k-16);
    } else if (k>16+nTEClayers) {
      layername = TString("TEC+ ")+ringlabel+(k-16-nTEClayers);
    }
   
    return layername;
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
