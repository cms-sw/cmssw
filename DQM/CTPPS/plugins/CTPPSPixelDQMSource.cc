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
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelIndices.h"

#include <string>

//-----------------------------------------------------------------------------
 
class CTPPSPixelDQMSource: public DQMEDAnalyzer
{
 public:
   CTPPSPixelDQMSource(const edm::ParameterSet& ps);
   ~CTPPSPixelDQMSource() override;
  
 protected:
   void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
   void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
   void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
   void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

 private:
   unsigned int verbosity;
   long int nEvents = 0;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tokenDigi;
  edm::EDGetTokenT< edm::DetSetVector<CTPPSPixelCluster> > tokenCluster;

 static constexpr int NArms=2;
 static constexpr int NStationMAX=3;  // in an arm
 static constexpr int NRPotsMAX=6;	// per station
 static constexpr int NplaneMAX=6;	// per RPot
 static constexpr int NROCsMAX = 6;	// per plane
 static constexpr int RPn_first = 3, RPn_last = 4;
 static constexpr int ADCMax = 256;
 static constexpr int StationIDMAX=4;  // possible range of ID
 static constexpr int RPotsIDMAX=8;    // possible range of ID
 const int hitMultMAX = 300; // tuned
 const int ClusMultMAX = 10; // tuned

 CTPPSPixelIndices thePixIndices;
 int pixRowMAX = 160;  // defaultDetSizeInX, CMS Y-axis
 int pixColMAX = 156;  // defaultDetSizeInY, CMS X-axis
 int ROCSizeInX = pixRowMAX/2;  // ROC row size in pixels = 80
 int ROCSizeInY = pixColMAX/3;  // ROC col size in pixels = 52

 static constexpr int NRPotBinsInStation = RPn_last-RPn_first;
 static constexpr int NPlaneBins = NplaneMAX*NRPotBinsInStation;

  MonitorElement *hBX, *hBXshort, *h2AllPlanesActive;

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
  MonitorElement *hp2HitsMultROC_LS[RPotsTotalNumber];
  MonitorElement *hRPotActivROCs[RPotsTotalNumber];
  MonitorElement   *hHitsMult[RPotsTotalNumber][NplaneMAX];
  MonitorElement    *h2xyHits[RPotsTotalNumber][NplaneMAX];
  MonitorElement    *hp2xyADC[RPotsTotalNumber][NplaneMAX];
  MonitorElement *h2xyROCHits[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  MonitorElement  *hROCadc[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  MonitorElement *hRPotActivBXall[RPotsTotalNumber];
  int		  HitsMultROC[RPotsTotalNumber*NplaneMAX][NROCsMAX];
  int           HitsMultPlane[RPotsTotalNumber][NplaneMAX];
  int           ClusMultPlane[RPotsTotalNumber][NplaneMAX];


  unsigned int rpStatusWord = 0x8008; //220_fr_hr(stn2rp3)+ 210_fr_hr
  int RPstatus[StationIDMAX][RPotsIDMAX]; // symmetric in both arms
  int StationStatus[StationIDMAX]; // symmetric in both arms
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

  int getRPInStationBin(int rp) { return(rp - RPn_first +1); }

  
  static constexpr int NRPglobalBins = 4; //2 arms w. 2 stations w. 1 RP
  int getRPglobalBin(int arm, int stn) { 
   static constexpr int stationBinOrder[NStationMAX] = {0, 4, 1};
   return( arm*2 + stationBinOrder[stn] +1 );
  }

  int prIndex(int rp, int plane) // plane index in station
	{return((rp - RPn_first)*NplaneMAX + plane);}
  int getDet(int id) 
	{ return (id>>DetId::kDetOffset)&0xF; }
  int getPixPlane(int id)
	{ return ((id>>16)&0x7); }
//  int getSubdet(int id) { return ((id>>kSubdetOffset)&0x7); }

 int multHitsMax, cluSizeMax; // for tuning

};

//----------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//-------------------------------------------------------------------------------

CTPPSPixelDQMSource::CTPPSPixelDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  rpStatusWord(ps.getUntrackedParameter<unsigned int>("RPStatusWord",0x8008))
{
 tokenDigi = consumes<DetSetVector<CTPPSPixelDigi> >(ps.getParameter<edm::InputTag>("tagRPixDigi"));
 tokenCluster=consumes<DetSetVector<CTPPSPixelCluster>>(ps.getParameter<edm::InputTag>("tagRPixCluster"));

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

 for(int stn=0; stn<StationIDMAX; stn++) {
  StationStatus[stn]=0;
  for(int rp=0; rp<RPotsIDMAX; rp++) RPstatus[stn][rp]=0;
 } 

 unsigned int rpSts = rpStatusWord<<1;
 for(int stn=0; stn<NStationMAX; stn++) {
   int stns = 0;
   for(int rp=0; rp<NRPotsMAX; rp++) {
     rpSts = (rpSts >> 1); RPstatus[stn][rp] = rpSts&1;
     if(RPstatus[stn][rp] > 0) stns = 1;
   }
   StationStatus[stn]=stns;
 }
 
 for(int ind=0; ind<2*3*NRPotsMAX; ind++) RPindexValid[ind] = 0;

 multHitsMax = cluSizeMax = -1;
}

//-------------------------------------------------------------------------------------

void CTPPSPixelDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, 
edm::EventSetup const &)
{
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS/TrackingPixel");
  char s[50];
  string armTitleShort, stnTitleShort;
  hBX = ibooker.book1D("events per BX", "ctpps_pixel;Event.BX", 4002, -1.5, 4000. +0.5);
  hBXshort = ibooker.book1D("events per BX(short)", "ctpps_pixel;Event.BX", 102, -1.5, 100. + 0.5);

  string str1st = "Pixel planes activity";
  h2AllPlanesActive = ibooker.book2DD(str1st,str1st+";Plane #",
                NplaneMAX,0,NplaneMAX, NRPglobalBins, 0.5, NRPglobalBins+0.5);
  TH2D *h1st = h2AllPlanesActive->getTH2D();
  h1st->SetOption("colz");
  TAxis *yah1st = h1st->GetYaxis();

 for(int arm=0; arm<2; arm++) {
   CTPPSDetId ID(CTPPSDetId::sdTrackingPixel,arm,0);
   string sd, armTitle;
   ID.armName(sd, CTPPSDetId::nPath);
   ID.armName(armTitle, CTPPSDetId::nFull);
   ID.armName(armTitleShort, CTPPSDetId::nShort);

   ibooker.setCurrentFolder(sd);

 for(int stn=0; stn<NStationMAX; stn++) {
   if(StationStatus[stn]==0) continue;
   ID.setStation(stn);
   string stnd, stnTitle;

   CTPPSDetId(ID.getStationId()).stationName(stnd, CTPPSDetId::nPath);
   CTPPSDetId(ID.getStationId()).stationName(stnTitle, CTPPSDetId::nFull);
   CTPPSDetId(ID.getStationId()).stationName(stnTitleShort, CTPPSDetId::nShort);

   ibooker.setCurrentFolder(stnd);

   string st = "planes activity";
   string st2 = ": " + stnTitle;

   h2PlaneActive[arm][stn] = ibooker.book2DD(st,st+st2+";Plane #",
		NplaneMAX,0,NplaneMAX, NRPotBinsInStation, 0.5, NRPotBinsInStation+0.5);
   TH2D *h = h2PlaneActive[arm][stn]->getTH2D();
   h->SetOption("colz");
   TAxis *yah = h->GetYaxis();
   
   st = "hit average multiplicity in planes";
   int PlaneMultCut = 20;
   hp2HitsOcc[arm][stn]= ibooker.bookProfile2D(st,st+st2+";Plane #",
     NplaneMAX, 0, NplaneMAX, NRPotBinsInStation, RPn_first,RPn_last,-1,PlaneMultCut,"");
   TProfile2D *h2 = hp2HitsOcc[arm][stn]->getTProfile2D();
   h2->SetOption("textcolz");
   TAxis *yah2 = h2->GetYaxis();

   st = "hit multiplicity in planes";
   string st3 = ";PlaneIndex(=pixelPot*PlaneMAX + plane)";
   h2HitsMultipl[arm][stn]= ibooker.book2DD(st,st+st2+st3+";multiplicity",
	NPlaneBins,0,NPlaneBins,hitMultMAX,0,hitMultMAX);
   h2HitsMultipl[arm][stn]->getTH2D()->SetOption("colz");

   st = "cluster multiplicity in planes";
   h2ClusMultipl[arm][stn] = ibooker.book2DD(st,st+st2+st3+";multiplicity",
	NPlaneBins,0,NPlaneBins, ClusMultMAX,0,ClusMultMAX);
   h2ClusMultipl[arm][stn]->getTH2D()->SetOption("colz");

   st = "cluster span in planes";
  const int nyClus = 9; //18;
  float xCh[NPlaneBins+1];
  float yClus[nyClus+1];
  for(int i=0; i<=NPlaneBins; i++) xCh[i]=i;
  double n0 = 1; //1./CluSizeMAX;
  double lnA = log(2.);
  yClus[0] = n0; //yClus[1] = n0;
  for(int j=0; j<nyClus; j++) yClus[j+1] = n0*exp(j*lnA);

  h2CluSize[arm][stn] = ibooker.book2D(st,st+st2+st3+";Cluster size",
	NPlaneBins,xCh,nyClus,yClus);
   h2CluSize[arm][stn]->getTH2F()->SetOption("colz");

//--------- RPots ---
   int pixBinW = 4;
     for(int rp=RPn_first; rp<RPn_last; rp++) { // only installed pixel pots
       ID.setRP(rp);
       string rpd, rpTitle;
       CTPPSDetId(ID.getRPId()).rpName(rpTitle, CTPPSDetId::nShort);
	yah->SetBinLabel(getRPInStationBin(rp), rpTitle.c_str()); // h
       yah2->SetBinLabel(getRPInStationBin(rp), rpTitle.c_str()); //h2
       string rpBinName = armTitleShort + "_" + stnTitleShort+"_"+rpTitle;
       yah1st->SetBinLabel(getRPglobalBin(arm,stn), rpBinName.c_str());

       if(RPstatus[stn][rp]==0) continue;
       int indexP = getRPindex(arm,stn,rp);
       RPindexValid[indexP] = 1;

       CTPPSDetId(ID.getRPId()).rpName(rpTitle, CTPPSDetId::nFull);
       CTPPSDetId(ID.getRPId()).rpName(rpd, CTPPSDetId::nPath);

       ibooker.setCurrentFolder(rpd);

       hRPotActivPlanes[indexP] = 
        ibooker.bookProfile("number of fired planes per event", rpTitle+";nPlanes;Probability",
	 NplaneMAX+1, -0.5, NplaneMAX+0.5, -0.5,NplaneMAX+0.5,"");
       hRPotActivBX[indexP] = 
        ibooker.book1D("5 fired planes per BX", rpTitle+";Event.BX", 4002, -1.5, 4000.+0.5);

       h2HitsMultROC[indexP] = ibooker.bookProfile2D("ROCs hits multiplicity per event",
       rpTitle+";plane # ;ROC #", NplaneMAX,-0.5,NplaneMAX-0.5, NROCsMAX,-0.5,NROCsMAX-0.5,
	0.,ROCSizeInX*ROCSizeInY,"");
       h2HitsMultROC[indexP]->getTProfile2D()->SetOption("colztext");
       h2HitsMultROC[indexP]->getTProfile2D()->SetMinimum(1.e-10);

	
       hp2HitsMultROC_LS[indexP]=ibooker.bookProfile2D("ROCs_hits_multiplicity_per_event vs LS",
	 rpTitle+";LumiSection;Plane#___ROC#", 1000,0.,1000.,
	 NplaneMAX*NROCsMAX,0.,double(NplaneMAX*NROCsMAX),0.,ROCSizeInX*ROCSizeInY,"");
       hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetOption("colz");
       hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetMinimum(1.0e-10);
       hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetCanExtend(TProfile2D::kXaxis);
       TAxis *yahp2 = hp2HitsMultROC_LS[indexP]->getTProfile2D()->GetYaxis();
       for(int p=0; p<NplaneMAX; p++) {
	 sprintf(s,"plane%d_0",p);
	 yahp2->SetBinLabel(p*NplaneMAX+1,s); 
	 for(int r=1; r<NROCsMAX; r++) {
	   sprintf(s,"   %d_%d",p,r);
	   yahp2->SetBinLabel(p*NplaneMAX+r+1,s);
	 }
       }

       hRPotActivROCs[indexP] = ibooker.bookProfile2D("number of fired aligned_ROCs per event",
 	 rpTitle+";ROC ID;number of fired aligned ROCs", NROCsMAX,-0.5,NROCsMAX-0.5, 
	 7,-0.5,6.5, 0,NROCsMAX,"");
       hRPotActivROCs[indexP]->getTProfile2D()->SetOption("colz");

       ibooker.setCurrentFolder(rpd+"/latency");

       hRPotActivBXroc[indexP] = 
        ibooker.book1D("4 fired ROCs per BX", rpTitle+";Event.BX", 4002, -1.5, 4000.+0.5);

       hRPotActivBXall[indexP] =
        ibooker.book1D("hits per BX", rpTitle+";Event.BX", 4002, -1.5, 4000.+0.5);

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
	   nbins,0,pixRowMAX,nbins,0,pixRowMAX, 0.,512.,"");
	 hp2xyADC[indexP][p]->getTProfile2D()->SetOption("colz");

         st = "hits multiplicity";
         hHitsMult[indexP][p]=ibooker.book1DD(st,st1+";number of hits;N / 1 hit",
	   hitMultMAX,-0.5,hitMultMAX-0.5);

         ibooker.setCurrentFolder(pd + "/ROCs");
         int index = getPlaneIndex(arm,stn,rp,p);

         for(int roc=0; roc<NROCsMAX; roc++) {
	   sprintf(s,"ROC_%d",roc);
	   string st2 = st1 + "_" + string(s);
           ibooker.setCurrentFolder(pd + "/ROCs/" + string(s));

           h2xyROCHits[index][roc]=ibooker.book2DD("hits",st2+";pix col;pix row",
		ROCSizeInY,0,ROCSizeInY, ROCSizeInX,0,ROCSizeInX);
           h2xyROCHits[index][roc]->getTH2D()->SetOption("colz");
           hROCadc[index][roc]=ibooker.book1D("adc value",st2+";ADC;number of ROCs",
		ADCMax, 0.,float(ADCMax));
         }
       } // end of for(int p=0; p<NplaneMAX;..

     } // end for(int rp=0; rp<NRPotsMAX;...
   } // end of for(int stn=0; stn<
  } //end of for(int arm=0; arm<2;...

 return;
}

//-------------------------------------------------------------------------------

void CTPPSPixelDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  ++nEvents;
  int lumiId = event.getLuminosityBlock().id().luminosityBlock();
  if(lumiId<0) lumiId=0;

  int RPactivity[RPotsTotalNumber], RPdigiSize[RPotsTotalNumber];

  for(int rp=0; rp<RPotsTotalNumber; rp++)
    { RPactivity[rp] = RPdigiSize[rp] = 0;}

  for(int ind=0; ind<RPotsTotalNumber; ind++) { 
    for(int p=0; p<NplaneMAX; p++) {
      HitsMultPlane[ind][p] = 0;
      ClusMultPlane[ind][p] = 0;
    }
  }
  for(int ind=0; ind<RPotsTotalNumber*NplaneMAX; ind++)  {
    for(int roc=0; roc<NROCsMAX; roc++) {
      HitsMultROC[ind][roc] = 0;
    }
  }
  Handle< DetSetVector<CTPPSPixelDigi> > pixDigi;
  event.getByToken(tokenDigi, pixDigi);

  Handle< DetSetVector<CTPPSPixelCluster> > pixClus;
  event.getByToken(tokenCluster, pixClus);

  hBX->Fill(event.bunchCrossing());
  hBXshort->Fill(event.bunchCrossing());

  bool valid = false;
  valid |= pixDigi.isValid();
//  valid |= Clus.isValid();

  if(!valid && verbosity) LogPrint("CTPPSPixelDQMSource") <<"No valid data in Event "<<nEvents;

  if(pixDigi.isValid()) {
    for(const auto &ds_digi : *pixDigi)
    {
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
      int rpInd = getRPindex(arm,station,rpot);
      RPactivity[rpInd] = 1;
      ++RPdigiSize[rpInd];

      if(StationStatus[station] && RPstatus[station][rpot]) {

        hp2HitsOcc[arm][station]->Fill(plane,rpot,(int)ds_digi.data.size());
        h2HitsMultipl[arm][station]->Fill(prIndex(rpot,plane),ds_digi.data.size());
        h2PlaneActive[arm][station]->Fill(plane,getRPInStationBin(rpot));
        h2AllPlanesActive->Fill(plane,getRPglobalBin(arm,station));

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
                h2xyROCHits[rocHistIndex][trocId]->Fill(colROC,rowROC);
                hROCadc[rocHistIndex][trocId]->Fill(adc);
                ++HitsMultROC[rocHistIndex][trocId];
              }
            }
          } //end if(RPindexValid[index]) {
        }
        if(int(ds_digi.data.size()) > multHitsMax) multHitsMax = ds_digi.data.size();

      } // end  if(StationStatus[station]) {
    } // end for(const auto &ds_digi : *pixDigi)
  } //if(pixDigi.isValid()) {

  if(pixClus.isValid()) 
    for(const auto &ds : *pixClus)
    {
      int idet = getDet(ds.id);
      if(idet != DetId::VeryForward) {
        if(verbosity>1) LogPrint("CTPPSPixelDQMSource") <<"not CTPPS: cluster.id"<<ds.id;
        continue;
      }
      //   int subdet = getSubdet(ds.id);

      int plane = getPixPlane(ds.id);

      CTPPSDetId theId(ds.id);
      int arm = theId.arm()&0x1;
      int station = theId.station()&0x3;
      int rpot = theId.rp()&0x7;

      if((StationStatus[station]==0) || (RPstatus[station][rpot]==0)) continue;

        int index = getRPindex(arm,station,rpot);
        ++ClusMultPlane[index][plane];

      unsigned int minRow=pixRowMAX, maxRow=0;
      unsigned int minCol=pixColMAX, maxCol=0;
      for (const auto &p : ds) {
        unsigned int max= p.minPixelRow() + p.rowSpan()+1;
        if(minRow > p.minPixelRow()) minRow = p.minPixelRow();
        if(maxRow < max) maxRow = max;
        max= p.minPixelCol() + p.colSpan()+1;
        if(minCol > p.minPixelCol()) minCol = p.minPixelCol();
        if(maxCol < max) maxCol = max;

      }
      int drow = maxRow - minRow;
      int dcol = maxCol - minCol;
      float clusize= sqrt(drow*drow + dcol*dcol);
      if(cluSizeMax < int(clusize)) cluSizeMax = clusize;
      h2CluSize[arm][station]->Fill(prIndex(rpot,plane),clusize);

   } // end if(pixClus.isValid()) for(const auto &ds : *pixClus)

  for(int arm=0; arm<2; arm++) {
    for(int stn=0; stn<NStationMAX; stn++) {
      for(int rp=0; rp<NRPotsMAX; rp++) {
        int index = getRPindex(arm,stn,rp);
        if(RPindexValid[index]==0) continue;
        if(RPactivity[index]==0) continue;

        for(int p=0; p<NplaneMAX; p++)
          h2ClusMultipl[arm][stn]->Fill(prIndex(rp,p),ClusMultPlane[index][p]);

        int np = 0; 
        for(int p=0; p<NplaneMAX; p++) if(HitsMultPlane[index][p]>0) np++;
	for(int p=0; p<=NplaneMAX; p++) {
	  if(p == np) hRPotActivPlanes[index]->Fill(p,1.);
	  else hRPotActivPlanes[index]->Fill(p,0.);
	}
        if(np>5) hRPotActivBX[index]->Fill(event.bunchCrossing());
        hRPotActivBXall[index]->Fill(event.bunchCrossing(),float(RPdigiSize[index]));

        int rocf[NplaneMAX];
        for(int r=0; r<NROCsMAX; r++) rocf[r]=0;
        for(int p=0; p<NplaneMAX; p++) {
          int indp = getPlaneIndex(arm,stn,rp,p);
          for(int r=0; r<NROCsMAX; r++) if(HitsMultROC[indp][r] > 0) ++rocf[r];
          for(int r=0; r<NROCsMAX; r++) { 
            h2HitsMultROC[index]->Fill(p,r,HitsMultROC[indp][r]);
            hp2HitsMultROC_LS[index]->Fill(lumiId,p*NROCsMAX+r,HitsMultROC[indp][r]);
          }
        }
        int max = 0;
        for(int r=0; r<NROCsMAX; r++) 
          if(max < rocf[r]) { max = rocf[r]; }

        for(int r=0; r<NROCsMAX; r++) {
	  for(int p=0; p<=NplaneMAX; p++)
	    if(p==rocf[r]) hRPotActivROCs[index]->Fill(r,rocf[r],1.);
	    else hRPotActivROCs[index]->Fill(r,p,0);
	}

        if(max > 4) hRPotActivBXroc[index]->Fill(event.bunchCrossing());
      } //end for(int rp=0; rp<NRPotsMAX; rp++) {
    }
  } //end for(int arm=0; arm<2; arm++) {

  if((nEvents % 100)) return;
  if(verbosity) LogPrint("CTPPSPixelDQMSource")<<"analyze event "<<nEvents;
}

//-----------------------------------------------------------------------------
void CTPPSPixelDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
 if(!verbosity) return; 
 LogPrint("CTPPSPixelDQMSource") 
  <<"end of Run "<<run.run()<<": "<<nEvents<<" events\n"
  <<"multHitsMax= "<<multHitsMax <<"   cluSizeMax= "<<cluSizeMax;
}

//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(CTPPSPixelDQMSource);

