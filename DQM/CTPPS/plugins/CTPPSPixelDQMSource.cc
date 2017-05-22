/******************************************
*
* This is a part of CTPPSDQM software.
* Authors:
*   F.Ferro INFN Genova
*   Vladimir Popov (vladimir.popov@cern.ch)
*
*******************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"

#include <string>

//-----------------------------------------------------------------------------
 
class CTPPSPixelDQMSource: public DQMEDAnalyzer
{
 public:
   CTPPSPixelDQMSource(const edm::ParameterSet& ps);
   virtual ~CTPPSPixelDQMSource();
  
 protected:
   void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
   void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
   void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
   void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
   void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
   void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

 private:
   unsigned int verbosity;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tokenDigi;
//  edm::EDGetTokenT< edm::DetSetVector<CTPPSPixelCluster> > tokenCluster;

 static constexpr int NArms=2;
 static constexpr int NStationMAX=3;  // in an arm
 static constexpr int NRPotsMAX=6;	// per station
 static constexpr int NplaneMAX=6;	// per RPot
 static constexpr int NROCsMAX = 6;	// per plane
 const int RPn_first = 3, RPn_last = 4;
 const int hitMultMAX = 300;
 const int ClusMultMAX = 90; // tuned
 const int CluSizeMAX = 25;  // tuned

 CTPPSPixelIndices thePixIndices;
 int pixRowMAX = 160;  // defaultDetSizeInX, CMS Y-axis
 int pixColMAX = 156;  // defaultDetSizeInY, CMS X-axis
 int ROCSizeInX = pixRowMAX/2;  // ROC row size in pixels = 80
 int ROCSizeInY = pixColMAX/3;  // ROC col size in pixels = 52
 long int nEvents = 0;

  MonitorElement *hBX, *hBXshort;

  MonitorElement *hp2HitsOcc[NArms][NStationMAX];
  MonitorElement *h2HitsMultipl[NArms][NStationMAX];
  MonitorElement *h2PlaneActive[NArms][NStationMAX];
  MonitorElement *h2ClusMultipl[NArms][NStationMAX];
  MonitorElement *h2CluSize[NArms][NStationMAX];

  static constexpr int RPotsTotalNumber=NArms*NStationMAX*NRPotsMAX;

  int	          RPindexValid[RPotsTotalNumber];
  MonitorElement *hRPotActivPlanes[RPotsTotalNumber];
  MonitorElement *hRPotActivBX[RPotsTotalNumber];
  MonitorElement *hRPotActivBXroc[RPotsTotalNumber];
  MonitorElement *h2HitsMultROC[RPotsTotalNumber];
  MonitorElement *hRPotActivROCs[RPotsTotalNumber];
  MonitorElement *hRPotActivROCsMax[RPotsTotalNumber];
  MonitorElement   *hHitsMult[RPotsTotalNumber][NplaneMAX];
  MonitorElement    *h2xyHits[RPotsTotalNumber][NplaneMAX];
  MonitorElement    *hp2xyADC[RPotsTotalNumber][NplaneMAX];
  MonitorElement *h2xyROCHits[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  MonitorElement  *h2xyROCadc[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  int		  HitsMultROC[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  int           HitsMultPlane[RPotsTotalNumber][NplaneMAX];


  unsigned int rpStatusWord = 0x8000; // 220 fr_hr (stn2rp3)
  int RPstatus[NStationMAX][NRPotsMAX]; // symmetric in both arms
  int StationStatus[NStationMAX]; // symmetric in both arms
  const int IndexNotValid = 0;

  int getRPindex(int arm, int station, int rp) {
	if(arm<0 || station<0 || rp<0) return(IndexNotValid);
	if(arm>1 || station>=NStationMAX || rp>=NRPotsMAX) return(IndexNotValid);
	int rc = (arm*NStationMAX+station)*NRPotsMAX + rp;
	return(rc);
  }

  int getPlaneIndex(int arm, int station, int rp, int plane) {
    if(plane<0 || plane>=NplaneMAX) return(IndexNotValid);
    int rc = getRPindex(arm, station, rp);
    if(rc == IndexNotValid) return(IndexNotValid);
    return(rc*NplaneMAX + plane);
  }

  int prIndex(int rp, int plane) // plane index in station
	{return((rp - RPn_first)*NplaneMAX + plane);}
  int getDet(int id) 
	{ return (id>>DetId::kDetOffset)&0xF; }
  int getPixPlane(int id)
	{ return ((id>>16)&0x7); }
//  int getSubdet(int id) { return ((id>>kSubdetOffset)&0x7); }

 int multHits, multClus, cluSizeMaxData; // for tuning

};

constexpr int CTPPSPixelDQMSource::NplaneMAX;
constexpr int CTPPSPixelDQMSource::NROCsMAX;

//----------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//-------------------------------------------------------------------------------

CTPPSPixelDQMSource::CTPPSPixelDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  rpStatusWord(ps.getUntrackedParameter<unsigned int>("RPStatusWord",0x8000))
{
 tokenDigi = consumes<DetSetVector<CTPPSPixelDigi> >(ps.getParameter<edm::InputTag>("tagRPixDigi"));
// tokenCluster=consumes<DetSetVector<CTPPSPixelCluster>>(ps.getParameter<edm::InputTag>("tagRPixCluster"));

}

//----------------------------------------------------------------------------------

CTPPSPixelDQMSource::~CTPPSPixelDQMSource()
{
}

//--------------------------------------------------------------------------

void CTPPSPixelDQMSource::dqmBeginRun(edm::Run const &run, edm::EventSetup const &)
{
  if(verbosity) LogPrint("CTPPSPixelDQMSource") <<"RPstatusWord= "<<rpStatusWord;
  nEvents = 0;

  pixRowMAX = thePixIndices.getDefaultRowDetSize();
  pixColMAX = thePixIndices.getDefaultColDetSize();
  ROCSizeInX = pixRowMAX/2;  // ROC row size in pixels = 80
  ROCSizeInY = pixColMAX/3;

 unsigned int rpSts = rpStatusWord<<1;
 for(int stn=0; stn<3; stn++) {
   int stns = 0;
   for(int rp=0; rp<NRPotsMAX; rp++) {
     rpSts = (rpSts >> 1); RPstatus[stn][rp] = rpSts&1;
     if(RPstatus[stn][rp] > 0) stns = 1;
   }
   StationStatus[stn]=stns;
 }
 
 for(int ind=0; ind<2*3*NRPotsMAX; ind++) RPindexValid[ind] = 0;

 multHits = multClus = cluSizeMaxData = -1;
}

//-------------------------------------------------------------------------------------

void CTPPSPixelDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, 
edm::EventSetup const &)
{
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS/TrackingPixel");
  char s[50];
  hBX = ibooker.book1D("events per BX", "ctpps_pixel;Event.BX", 4002, -1.5, 4000. +0.5);
  hBXshort = ibooker.book1D("events per BX(short)", "ctpps_pixel;Event.BX", 102, -1.5, 100. + 0.5);

 for(int arm=0; arm<2; arm++) {
   CTPPSDetId ID(CTPPSDetId::sdTrackingPixel,arm,0);
   string sd, armTitle;
   ID.armName(sd, CTPPSDetId::nPath);
   ID.armName(armTitle, CTPPSDetId::nFull);

   ibooker.setCurrentFolder(sd);

 for(int stn=2; stn<NStationMAX; stn++) {
   if(StationStatus[stn]==0) continue;
   ID.setStation(stn);
   string stnd, stnTitle;

   CTPPSDetId(ID.getStationId()).stationName(stnd, CTPPSDetId::nPath);
   CTPPSDetId(ID.getStationId()).stationName(stnTitle, CTPPSDetId::nFull);

   ibooker.setCurrentFolder(stnd);

   string st = "planes activity";
   string st2 = ": " + stnTitle;

   int rpnbins = RPn_last-RPn_first; 
  
   h2PlaneActive[arm][stn] = ibooker.book2DD(st,st+st2+";Plane #",
		NplaneMAX,0,NplaneMAX, rpnbins, RPn_first,RPn_last);
   TH2D *h = h2PlaneActive[arm][stn]->getTH2D();
   h->SetOption("colz");
   TAxis *yah = h->GetYaxis();
   
   st = "hit average multiplicity in planes";

   hp2HitsOcc[arm][stn]= ibooker.bookProfile2D(st,st+st2+";Plane #",
     NplaneMAX, 0, NplaneMAX, rpnbins, RPn_first,RPn_last,0,20000);
   TProfile2D *h2 = hp2HitsOcc[arm][stn]->getTProfile2D();
   h2->SetOption("textcolz");
   TAxis *yah2 = h2->GetYaxis();

   int xmax = NplaneMAX*rpnbins;

   st = "hit multiplicity in planes";
   string st3 = ";PlaneIndex(=pixelPot*PlaneMAX + plane)";
   h2HitsMultipl[arm][stn]= ibooker.book2DD(st,st+st2+st3+";multiplicity",
	xmax,0,xmax,hitMultMAX,0,hitMultMAX);
   h2HitsMultipl[arm][stn]->getTH2D()->SetOption("colz");

   st = "cluster multiplicity in planes";
   h2ClusMultipl[arm][stn] = ibooker.book2DD(st,st+st2+st3+";multiplicity",
	xmax,0,xmax, ClusMultMAX,0,ClusMultMAX);
   h2ClusMultipl[arm][stn]->getTH2D()->SetOption("colz");

   st = "cluster size in planes";
  h2CluSize[arm][stn] = ibooker.book2D(st,st+st2+st3+";Cluster size",
	xmax,0,xmax, CluSizeMAX,0,CluSizeMAX);
   h2CluSize[arm][stn]->getTH2F()->SetOption("colz");

//--------- Hits ---
   int pixBinW = 4;
     for(int rp=RPn_first; rp<RPn_last; rp++) { // only installed pixel pots
       ID.setRP(rp);
       string rpd, rpTitle;
       CTPPSDetId(ID.getRPId()).rpName(rpTitle, CTPPSDetId::nShort);
	yah->SetBinLabel(rp - RPn_first +1, rpTitle.c_str()); // h
       yah2->SetBinLabel(rp - RPn_first +1, rpTitle.c_str()); //h2

       if(RPstatus[stn][rp]==0) continue;
       int indexP = getRPindex(arm,stn,rp);
       RPindexValid[indexP] = 1;

       CTPPSDetId(ID.getRPId()).rpName(rpTitle, CTPPSDetId::nFull);
       CTPPSDetId(ID.getRPId()).rpName(rpd, CTPPSDetId::nPath);

       ibooker.setCurrentFolder(rpd);

       hRPotActivPlanes[indexP] = 
        ibooker.book1D("number of fired planes per event", rpTitle+";nPlanes",
	 NplaneMAX, -0.5, NplaneMAX+0.5);
       hRPotActivBX[indexP] = 
        ibooker.book1D("5 fired planes per BX", rpTitle+";Event.BX", 4002, -1.5, 4000.+0.5);
       hRPotActivBXroc[indexP] = 
        ibooker.book1D("4 fired ROCs per BX", rpTitle+";Event.BX", 4002, -1.5, 4000.+0.5);

       h2HitsMultROC[indexP] = ibooker.bookProfile2D("ROCs hits multiplicity per event",
       rpTitle+";plane # ;ROC #", NplaneMAX,-0.5,NplaneMAX-0.5, NROCsMAX,-0.5,NROCsMAX-0.5, 100,0,1.e3);

       h2HitsMultROC[indexP]->getTProfile2D()->SetOption("colztext");
       h2HitsMultROC[indexP]->getTProfile2D()->SetMinimum(1.e-10);

       hRPotActivROCs[indexP] = ibooker.book2D("number of fired aligned_ROCs per event",
 	 rpTitle+";ROC ID;number of fired ROCs", NROCsMAX,-0.5,NROCsMAX-0.5, 7,-0.5,6.5);
       hRPotActivROCs[indexP]->getTH2F()->SetOption("colz");

       hRPotActivROCsMax[indexP]= ibooker.book2D("max number of fired aligned_ROCs per event",
	 rpTitle+";ROC ID;number of fired ROCs", NROCsMAX,-0.5,NROCsMAX-0.5, 7,-0.5,6.5);
       hRPotActivROCsMax[indexP]->getTH2F()->SetOption("colz");

       int nbins = pixRowMAX/pixBinW;

       for(int p=0; p<NplaneMAX; p++) {
         sprintf(s,"plane_%d",p);
         string pd = rpd+"/"+string(s);
         ibooker.setCurrentFolder(pd);
         string st1 = ": "+rpTitle+"_"+string(s);
         st = "hits position";
         h2xyHits[indexP][p]=ibooker.book2DD(st,st1+";pix col;pix row", 
	   nbins,0,pixRowMAX,nbins,0,pixRowMAX);
	 h2xyHits[indexP][p]->getTH2D()->SetOption("colz");

 	 st = "adc average value";
	 hp2xyADC[indexP][p]=ibooker.bookProfile2D(st,st1+";pix col;pix row",
	   nbins,0,pixRowMAX,nbins,0,pixRowMAX,100,0,1.e10,"");
	 hp2xyADC[indexP][p]->getTProfile2D()->SetOption("colz");

         st = "hits multiplicity";
         hHitsMult[indexP][p]=ibooker.book1DD(st,st1+";number of hits;N / 1 hit",
	   hitMultMAX,0,hitMultMAX);

         ibooker.setCurrentFolder(pd + "/ROCs");
         int index = getPlaneIndex(arm,stn,rp,p);

         for(int roc=0; roc<6; roc++) {
	   sprintf(s,"ROC_%d",roc);
	   string st2 = st1 + "_" + string(s);
           ibooker.setCurrentFolder(pd + "/ROCs/" + string(s));

           h2xyROCHits[index][roc]=ibooker.book2DD("hits",st2+";pix row;pix col",
		ROCSizeInX,0,ROCSizeInX,ROCSizeInY,0,ROCSizeInY);
           h2xyROCHits[index][roc]->getTH2D()->SetOption("colz");

           string st = "adc average value";
           h2xyROCadc[index][roc]=ibooker.bookProfile2D(st,st2+";pix row;pix col",
		ROCSizeInX,0,ROCSizeInX,ROCSizeInY,0,ROCSizeInY, 0,512);
           h2xyROCadc[index][roc]->getTProfile2D()->SetOption("colz");
         }
       } // end of for(int p=0; p<NplaneMAX;..

     } // end for(int rp=0; rp<NRPotsMAX;...
   } // end of for(int stn=0; stn<
  } //end of for(int arm=0; arm<2;...

 return;
}

//-------------------------------------------------------------------------------
void CTPPSPixelDQMSource::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
}
//---------------------------------------------------------------------------------

void CTPPSPixelDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  ++nEvents;
  int RPactivity[2][NRPotsMAX];
  for(int arm = 0; arm <2; arm++) { 
    for(int rp=0; rp<NRPotsMAX; rp++) {
      RPactivity[arm][rp] = 0;
    }
  }
  for(int ind=0; ind<2*3*NRPotsMAX; ind++) { 
    for(int p=0; p<NplaneMAX; p++) {
      HitsMultPlane[ind][p] = 0;
    }
  }
  for(int ind=0; ind<2*3*NRPotsMAX*NplaneMAX; ind++)  {
    for(int rp=0; rp<NROCsMAX; rp++) {
      HitsMultROC[ind][rp] = 0;
    }
  }
  Handle< DetSetVector<CTPPSPixelDigi> > pixDigi;
  event.getByToken(tokenDigi, pixDigi);

  hBX->Fill(event.bunchCrossing());
  hBXshort->Fill(event.bunchCrossing());

  bool valid = false;
  valid |= pixDigi.isValid();
//  valid |= Clus.isValid();

  if(!valid && verbosity) LogPrint("CTPPSPixelDQMSource") <<"No valid data in Event "<<nEvents;

  if(pixDigi.isValid()) {
    for(const auto &ds_digi : *pixDigi) {
      int idet = getDet(ds_digi.id);
      if(idet != DetId::VeryForward) {
        if(verbosity>1) LogPrint("CTPPSPixelDQMSource") <<"not CTPPS: ds_digi.id"<<ds_digi.id;
        continue;
      }
      //   int subdet = getSubdet(ds_digi.id);
      
      int plane = getPixPlane(ds_digi.id);
      
      CTPPSDetId theId(ds_digi.id);
      int arm = theId.arm()&0x1;
      int station = theId.station()&0x3;
      int rpot = theId.rp()&0x7;
      RPactivity[arm][rpot] = 1;
      
      if(StationStatus[station] && RPstatus[station][rpot]) {
        
        hp2HitsOcc[arm][station]->Fill(plane,rpot,(int)ds_digi.data.size());
        h2HitsMultipl[arm][station]->Fill(prIndex(rpot,plane),ds_digi.data.size());
        h2PlaneActive[arm][station]->Fill(plane,rpot);
        
        int index = getRPindex(arm,station,rpot);
        HitsMultPlane[index][plane] += ds_digi.data.size();
        if(RPindexValid[index]) {
          hHitsMult[index][plane]->Fill(ds_digi.data.size());
        }
        int rocHistIndex = getPlaneIndex(arm,station,rpot,plane);
        
        for(DetSet<CTPPSPixelDigi>::const_iterator dit = ds_digi.begin();
            dit != ds_digi.end(); ++dit) {
          int row = dit->row();
          int col = dit->column();
          int adc = dit->adc();
          
          if(RPindexValid[index]) {
            h2xyHits[index][plane]->Fill(col,row);
            hp2xyADC[index][plane]->Fill(col,row,adc);
            int colROC, rowROC;
            int trocId;
            if(!thePixIndices.transformToROC(col,row, trocId, colROC, rowROC)) {
              if(trocId>=0 && trocId<NROCsMAX) {
                h2xyROCHits[rocHistIndex][trocId]->Fill(rowROC,colROC);
                h2xyROCadc[rocHistIndex][trocId]->Fill(rowROC,colROC,adc);
                ++HitsMultROC[rocHistIndex][trocId];
              }
            }
          } //end if(RPindexValid[index]) {
        }
      
        if(int(ds_digi.data.size()) > multHits) multHits = ds_digi.data.size();
      } // end  if(StationStatus[station]) {
    } // end for(const auto &ds_digi : *pixDigi)
  } //if(pixDigi.isValid()) {

  for(int arm=0; arm<2; arm++) {
    for(int stn=0; stn<NStationMAX; stn++) {
      for(int rp=0; rp<NRPotsMAX; rp++) {
        int index = getRPindex(arm,stn,rp);
        if(RPindexValid[index]==0) continue;
        if(RPactivity[arm][rp]==0) continue;
        
        int np = 0; 
        for(int p=0; p<NplaneMAX; p++) {
          if(HitsMultPlane[index][p]>0) np++;
        }
        hRPotActivPlanes[index]->Fill(np);
        if(np>5) { hRPotActivBX[index]->Fill(event.bunchCrossing());}
        
        int rocf[NplaneMAX];
        for(int r=0; r<NROCsMAX; r++) { rocf[r]=0; }
        for(int p=0; p<NplaneMAX; p++) {
          int indp = getPlaneIndex(arm,stn,rp,p);
          for(int r=0; r<NROCsMAX; r++) {
            if(HitsMultROC[indp][r] > 0) ++rocf[r];
          }
          for(int r=0; r<NROCsMAX; r++) { 
            h2HitsMultROC[index]->Fill(p,r,HitsMultROC[indp][r]);
          }
        }
        int max = 0;
        for(int r=0; r<NROCsMAX; r++) {
          if(max < rocf[r]) { max = rocf[r]; }
        }
        for(int r=0; r<NROCsMAX; r++) {
          hRPotActivROCs[index]->Fill(r,rocf[r]); 
        }
        for(int r=0; r<NROCsMAX; r++) {
          if(rocf[r] == max) {hRPotActivROCsMax[index]->Fill(r,max);}
        }
        if(max > 4) hRPotActivBXroc[index]->Fill(event.bunchCrossing());
      }
    }
  }

  if((nEvents % 100)) return;
  if(verbosity) LogPrint("CTPPSPixelDQMSource")<<"analyze event "<<nEvents;
}

//--------------------------------------------------------------
void CTPPSPixelDQMSource::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//-----------------------------------------------------------------------------
void CTPPSPixelDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
 if(!verbosity) return; 
 LogPrint("CTPPSPixelDQMSource") 
  <<"end of Run "<<run.run()<<": "<<nEvents<<" events\n"
  <<"mult Hits/Clus: "<<multHits<<" / "<<multClus
  <<"   cluSizeMaxData= "<<cluSizeMaxData;
}

//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(CTPPSPixelDQMSource);

