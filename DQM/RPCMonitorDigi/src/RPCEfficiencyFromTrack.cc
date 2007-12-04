/**********************************************
 *                                            *
 *           Giuseppe Roselli                 *
 *         INFN, Sezione di Bari              *
 *      Via Amendola 173, 70126 Bari          *
 *         Phone: +390805443218               *
 *      giuseppe.roselli@ba.infn.it           *
 *                                            *
 *                                            *
 **********************************************/


#include "DQM/RPCMonitorDigi/interface/RPCEfficiencyFromTrack.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"


#include <memory>
#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TAxis.h"

//
// class decleration
//
using namespace edm;
using namespace reco;
using namespace std;


RPCEfficiencyFromTrack::RPCEfficiencyFromTrack(const edm::ParameterSet& iConfig){
  std::map<RPCDetId, int> buff;
  counter.clear();
  counter.reserve(3);
  counter.push_back(buff);
  counter.push_back(buff);
  counter.push_back(buff);
  totalcounter.clear();
  totalcounter.reserve(3);
  totalcounter[0]=0;
  totalcounter[1]=0;
  totalcounter[2]=0;
  
  MeasureEndCap = iConfig.getParameter<bool>("ReadEndCap");
  MeasureBarrel = iConfig.getParameter<bool>("ReadBarrel");
  maxRes = iConfig.getParameter<double>("EfficiencyCut");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", true); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 1000); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiencyFromTrack.root"); 
  TjInput  = iConfig.getUntrackedParameter<std::string>("trajectoryInput");
  RPCDataLabel = iConfig.getUntrackedParameter<std::string>("rpcRecHitLabel");
  GlobalRootLabel= iConfig.getUntrackedParameter<std::string>("GlobalRootFileName","GlobalEfficiencyFromTrack.root");
  fOutputFile  = new TFile(GlobalRootLabel.c_str(), "RECREATE" );
  hRecPt = new TH1F("RecPt","ReconstructedPt",100,0.5,100.5);
  hGlobalRes = new TH1F("GlobalResiduals","GlobalRPCResiduals",50,-15.,15.);
  hGlobalPull = new TH1F("GlobalPull","GlobalRPCPull",50,-15.,15.);
  histoMean = new TH1F("MeanEfficincy","MeanEfficiency_vs_Ch",100,20.5,120.5);
  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  _idList.clear(); 

}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack()
{
  fOutputFile->Write();
  fOutputFile->Close();
}


void RPCEfficiencyFromTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  char layerLabel[128];
  char meIdRPC [128];
  char meIdTrack [128];
  std::map<RPCDetId, int> buff;

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(RPCDataLabel,rpcHits);

  ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  Handle<Trajectories> trajectories;
  iEvent.getByLabel(TjInput,trajectories);

  for(Trajectories::const_iterator tj = trajectories->begin(); tj != trajectories->end(); ++tj){
    std::vector<TrajectoryMeasurement> tmColl = tj->measurements();

    for(vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); itTraj++){
      for (GlobalTrackingGeometry::DetContainer::const_iterator itDet=rpcGeo->dets().begin();itDet<rpcGeo->dets().end();itDet++){
	if( dynamic_cast< RPCChamber* >( *itDet ) != 0 ){
	  RPCChamber* ch = dynamic_cast< RPCChamber* >( *itDet ); 
	  std::vector< const RPCRoll*> roles = (ch->rolls());
	  for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	    RPCDetId rpcId = (*r)->id();
	    const BoundPlane & RPCSur=(*r)->surface();
	    const LocalPoint LTj=itTraj->updatedState().localPosition();
	    const GlobalPoint GTj=itTraj->updatedState().globalPosition();

	    float Gx1=RPCSur.toGlobal(LTj).x();
	    float Gy1=RPCSur.toGlobal(LTj).y();
	    float Gz1=RPCSur.toGlobal(LTj).z();

	    float Gx2=GTj.x();
	    float Gy2=GTj.y();
	    float Gz2=GTj.z();
	    
	    if((fabs(Gx1-Gx2)<0.01 && fabs(Gy1-Gy2)<0.01 && fabs(Gz1-Gz2)<0.01 && rpcId.region()==0 && MeasureBarrel==true) ||
	       (fabs(Gx1-Gx2)<0.01 && fabs(Gy1-Gy2)<0.01 && fabs(Gz1-Gz2)<0.01 && rpcId.region()!=0 && MeasureEndCap==true)){



	      RPCDetId rollId = (*r)->id();
	      RPCGeomServ RPCname(rollId);
	      std::string nameRoll = RPCname.name();
	      _idList.push_back(nameRoll);
	      

	      std::cout<<"RPC -- Candidate"<<nameRoll<<std::endl;
	      

	      char detUnitLabel[128];
	      sprintf(detUnitLabel ,"%s",nameRoll.c_str());
	      sprintf(layerLabel ,"%s",nameRoll.c_str());
	      std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
	      if (meItr == meCollection.end()){
		meCollection[nameRoll] = bookDetUnitTrackEff(rollId);
	      }
	      std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];

	      totalcounter[0]++;
	      buff=counter[0];
	      buff[rollId]++;
	      counter[0]=buff;

	      RPCRecHitCollection::range rpcRecHitRange = rpcHits->get((*r)->id());
	      RPCRecHitCollection::const_iterator recIt;
	      
	      int stripPredicted = (int)((*r)->strip(itTraj->updatedState().localPosition()));
	      double xextrap = itTraj->updatedState().localPosition().x();

	      sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	      meMap[meIdTrack]->Fill(stripPredicted);

	      bool recFound=false;
	      bool anycoincidence=false;
	      std::vector<double> ResVec;
	      ResVec.clear();
	      std::vector<double> RecErr;
	      RecErr.clear();
	      std::vector<double> extrVec;
	      extrVec.clear();
	      std::vector<double> posVec;
	      posVec.clear();
	      std::vector<double> stripD;
	      stripD.clear();
	      std::vector<double> stripPr;
	      stripPr.clear();

	      double res=0.;

	      for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
		recFound=true;
		LocalPoint rhitlocal = (*recIt).localPosition();
		double rhitpos = rhitlocal.x();  
		double stripDetected = (int)((*r)->strip(rhitlocal));
		LocalError RecError = (*recIt).localPositionError();
		double sigmaRec = RecError.xx();
		res = (double)(xextrap - rhitpos);

		ResVec.push_back(res);		
		RecErr.push_back(sigmaRec);
		extrVec.push_back(xextrap);	
		posVec.push_back(rhitpos);	
		stripD.push_back(stripDetected);	
		stripPr.push_back(stripPredicted);
	      }

	      //Find the minimum residual

	      res = ResVec[0];
	      int rpos=0;
	      if(ResVec.size()>1)std::cout<<"Best Residual Finder.............. "<<std::endl;
	      for(unsigned int rs=0;rs<ResVec.size();rs++){
		if(ResVec.size()>1)std::cout<<"Residual candidate"<<rs<<"\t= "<<ResVec[rs]<<std::endl;
		if(fabs(ResVec[rs]) < fabs(res)){
		  res = ResVec[rs];
		  rpos=rs;
		}
	      }


	      if(recFound==true){
		std::cout<<"**********************************************"<<std::endl;
		std::cout<<"\t                                   "<<std::endl;
		std::cout<<"Point Extrapolated                   "<<extrVec[rpos]<<std::endl;
		std::cout<<"Real Point                           "<<posVec[rpos]<<std::endl;
		std::cout<<"**********************************************"<<std::endl;
		std::cout<<"Strip Extrapolated "<<stripPr[rpos]<<" Strip Detected "<<stripD[rpos]<<std::endl;
		std::cout<<"**********************************************"<<std::endl;
		
		sprintf(meIdRPC,"Residuals_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(res);
		sprintf(meIdRPC,"Residuals_VS_RecPt_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(itTraj->updatedState().globalMomentum().perp(),res);

		hGlobalRes->Fill(res);
		hGlobalPull->Fill(res/RecErr[rpos]);
		hRecPt->Fill(itTraj->updatedState().globalMomentum().perp());
	      }
	      if(fabs(res)<maxRes){
		anycoincidence=true;
		std::cout<<"Good Match "<<"\t"<<"Residuals = "<<res<<"\t"<<(*r)->id()<<std::endl;
	      }
	      else{
		anycoincidence=false;
		std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<(*r)->id()<<std::endl;
	      }
	      


	      if(anycoincidence==true){
		  totalcounter[1]++;
		  buff=counter[1];
		  buff[rollId]++;
		  counter[1]=buff;
	
		  sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
		  meMap[meIdRPC]->Fill(stripD[rpos]);

		  sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
		  meMap[meIdRPC]->Fill(stripPr[rpos]);

	      }else{
		  totalcounter[2]++;
		  buff=counter[2];
		  buff[rollId]++;
		  counter[2]=buff;
	      }
	    }
	  }
	}
      }
    }
  }
}


void RPCEfficiencyFromTrack::beginJob(const edm::EventSetup&)
{
}

void RPCEfficiencyFromTrack::endJob() {

  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];
  std::map<RPCDetId, int>::iterator irpc;
  
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    assert(p==o+r);
   
    if(p!=0){
      float ef = float(o)/float(p); 
      float er = sqrt(ef*(1.-ef)/float(p));
      std::cout <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
      if(ef<0.8){
	std::cout<<"\t \t Warning!";
      } 
    }
    else{
      std::cout<<"No predictions in this file predicted=0"<<std::endl;
    }
  }
  if(totalcounter[0]!=0){
    float tote = float(totalcounter[1])/float(totalcounter[0]);
    float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
    std::cout <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[0]<<"\t Observed "<<totalcounter[1]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<"%"<<std::endl;
    std::cout <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;
    histoMean->Fill(tote*100.);
  }
  else{
    std::cout<<"No predictions in this file = 0!!!"<<std::endl;
  }

  std::vector<std::string>::iterator meIt;
  int id=0;
  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){
    id++;

    char detUnitLabel[128];
    char meIdRPC [128];
    char meIdTrack [128];
    char meIdRes [128];
    char effIdRPC [128];
    sprintf(detUnitLabel ,"%s",(*meIt).c_str());
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
    sprintf(meIdRes,"Residuals_%s",detUnitLabel);
    sprintf(effIdRPC,"EfficienyFromTrackExtrapolation_%s",detUnitLabel);
    std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

    for(unsigned int i=1;i<=100;++i){
      if(meMap[meIdTrack]->getBinContent(i) != 0){
	float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdTrack]->getBinContent(i);
	float erreff = sqrt(eff*(1-eff)/meMap[meIdTrack]->getBinContent(i));
	meMap[effIdRPC]->setBinContent(i,eff*100.);
	meMap[effIdRPC]->setBinError(i,erreff*100.);
      }
    }
  }

  
  if(EffSaveRootFile) dbe->save(EffRootFileName);
}


DEFINE_FWK_MODULE(RPCEfficiencyFromTrack);
