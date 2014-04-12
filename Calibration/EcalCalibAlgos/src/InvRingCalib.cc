#include <memory>
#include <math.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Calibration/EcalCalibAlgos/interface/InvRingCalib.h"
#include "Calibration/EcalCalibAlgos/interface/IMACalibBlock.h"
#include "Calibration/EcalCalibAlgos/interface/L3CalibBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Calibration/EcalCalibAlgos/interface/MatrixFillMap.h"
#include "Calibration/EcalCalibAlgos/interface/ClusterFillMap.h"

//Not to remain in the final version
#include "TH2.h"
#include "TFile.h"
#include <iostream>
//----------------------------------------------------------------
//ctor

InvRingCalib::InvRingCalib (const edm::ParameterSet& iConfig) :
  m_barrelAlCa (iConfig.getParameter<edm::InputTag>("barrelAlca")),
  m_endcapAlCa (iConfig.getParameter<edm::InputTag>("endcapAlca")) ,
  m_ElectronLabel (iConfig.getParameter<edm::InputTag>("ElectronLabel")),
  m_recoWindowSidex (iConfig.getParameter<int>("recoWindowSidex")),
  m_recoWindowSidey (iConfig.getParameter<int>("recoWindowSidey")),
  m_minEnergyPerCrystal (iConfig.getParameter<double>("minEnergyPerCrystal")) ,
  m_maxEnergyPerCrystal (iConfig.getParameter<double>("maxEnergyPerCrystal")) ,
  m_etaStart (iConfig.getParameter<int>("etaStart")),
  m_etaEnd (iConfig.getParameter<int>("etaEnd")),
  m_etaWidth (iConfig.getParameter<int>("etaWidth")),
  m_maxSelectedNumPerRing (iConfig.getParameter<int>("maxNumPerRing")),
  m_minCoeff (iConfig.getParameter<double>("minCoeff")),
  m_maxCoeff (iConfig.getParameter<double>("maxCoeff")),
  m_usingBlockSolver(iConfig.getParameter<int>("usingBlockSolver")),
  m_startRing (iConfig.getParameter<int>("startRing")),
  m_endRing (iConfig.getParameter<int>("endRing")),
  m_EBcoeffFile (iConfig.getParameter<std::string>("EBcoeffs")),
  m_EEcoeffFile (iConfig.getParameter<std::string>("EEcoeffs")),
  m_EEZone (iConfig.getParameter<int>("EEZone"))
{
  //controls if the parameters inputed are correct
  if ((m_etaEnd*m_etaStart)>0)
    assert (!((m_etaEnd - m_etaStart )%m_etaWidth)); 
  if ((m_etaEnd*m_etaStart)<0)
    assert (!((m_etaEnd - m_etaStart-1 )%m_etaWidth)); 

  assert (m_etaStart >=-85 && m_etaStart <= 86);
  assert (m_etaEnd >= m_etaStart && m_etaEnd <= 86);
  assert (m_startRing>-1 && m_startRing<= 40);
  assert (m_endRing>=m_startRing && m_endRing<=40);

  assert (!((m_endRing - m_startRing)%m_etaWidth));
  assert (( abs(m_EEZone)<=1));
  
  m_loops = (unsigned int) iConfig.getParameter<int>("loops")- 1;
  //LP CalibBlock vector instantiation
  edm::LogInfo ("IML") << "[InvRingCalib][ctor] Calib Block" ;
  std::string algorithm = iConfig.getParameter<std::string> ("algorithm") ;
  m_mapFillerType = iConfig.getParameter<std::string> ("FillType");
  int eventWeight = iConfig.getUntrackedParameter<int> ("L3EventWeight",1) ;
  
  for (int i = 0 ; i < EBRegionNum () ; ++i)
   {
  if (algorithm == "IMA")
        m_IMACalibBlocks.push_back (
            new IMACalibBlock (m_etaWidth)
          ) ; 
      else if (algorithm == "L3")
        m_IMACalibBlocks.push_back (
            new L3CalibBlock (m_etaWidth, eventWeight)
          ) ; 
      else
        {
          edm::LogError ("building") << algorithm 
                          << " is not a valid calibration algorithm" ;
          exit (1) ;    
        }      
   }   
  int EEBlocks = 0 ;
  if (m_EEZone == 0) EEBlocks = 2 * EERegionNum () ;
  if (m_EEZone == 1 || m_EEZone == -1) EEBlocks = EERegionNum () ;

  for (int i = 0; i < EEBlocks ; ++i)
   {
  
  if (algorithm == "IMA")
        m_IMACalibBlocks.push_back (
            new IMACalibBlock (m_etaWidth)
          ) ; 
      else if (algorithm == "L3")
        m_IMACalibBlocks.push_back (
            new L3CalibBlock (m_etaWidth, eventWeight)
          ) ; 
      else
        {
          edm::LogError ("building") << algorithm 
                          << " is not a valid calibration algorithm" ;
          exit (1) ;    
        }      
   }   
  edm::LogInfo ("IML") <<" [InvRingCalib][ctor] end of creator";
}


//-------------------------------------------------------------- end ctor
//!destructor


InvRingCalib::~InvRingCalib ()
{
}


//---------------------------------------------------



//!BeginOfJob
void 
InvRingCalib::beginOfJob () 
{
  isfirstcall_=true;

 
}


//--------------------------------------------------------


//!startingNewLoop
void InvRingCalib::startingNewLoop (unsigned int ciclo) 
{
    edm::LogInfo ("IML") << "[InvMatrixCalibLooper][Start] entering loop " << ciclo;
    for (std::vector<VEcalCalibBlock *>::iterator calibBlock = m_IMACalibBlocks.begin () ;
         calibBlock != m_IMACalibBlocks.end () ;
         ++calibBlock)
      {
        //LP empties the energies vector, to fill DuringLoop.
        (*calibBlock)->reset () ;
      }
   for (std::map<int,int>::const_iterator ring=m_xtalRing.begin();
        ring!=m_xtalRing.end();
        ++ring)
	   m_RingNumOfHits[ring->second]=0;
   return ;
}


//--------------------------------------------------------


//!duringLoop
edm::EDLooper::Status 
InvRingCalib::duringLoop (const edm::Event& iEvent,
                          const edm::EventSetup& iSetup) 
{


   if (isfirstcall_){
    edm::LogInfo ("IML") << "[InvRingCalib][beginOfJob]" ;
    //gets the geometry from the event setup
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle);
    const CaloGeometry& geometry = *geoHandle;
    edm::LogInfo ("IML") <<"[InvRingCalib] Event Setup read";
    //fills a vector with all the cells
    m_barrelCells = geometry.getValidDetIds(DetId::Ecal, EcalBarrel);
    m_endcapCells = geometry.getValidDetIds(DetId::Ecal, EcalEndcap);
    //Defines the EB regions
    edm::LogInfo ("IML") <<"[InvRingCalib] Defining Barrel Regions";
    EBRegionDef();
    //Defines what is a ring in the EE
    edm::LogInfo ("IML") <<"[InvRingCalib] Defining endcap Rings";
    EERingDef(iSetup);
    //Defines the regions in the EE
    edm::LogInfo ("IML") <<"[InvRingCalib] Defining endcap Regions";
    EERegionDef();
    if (m_mapFillerType == "Cluster") m_MapFiller= new ClusterFillMap (
								       m_recoWindowSidex ,m_recoWindowSidey ,
								       m_xtalRegionId ,m_minEnergyPerCrystal ,
								       m_maxEnergyPerCrystal , m_RinginRegion ,
								       & m_barrelMap ,
								       & m_endcapMap ); 
    if (m_mapFillerType == "Matrix") m_MapFiller = new MatrixFillMap (
								      m_recoWindowSidex ,m_recoWindowSidey ,
								      m_xtalRegionId , m_minEnergyPerCrystal ,
								      m_maxEnergyPerCrystal , m_RinginRegion ,
								      & m_barrelMap ,
								      & m_endcapMap); 
    edm::LogInfo ("IML") <<"[InvRingCalib] Initializing the coeffs";
    //Sets the initial coefficients to 1.
    //Graphs to check ring, regions and so on, not needed in the final version
    TH2F EBRegion ("EBRegion","EBRegion",171,-85,86,360,1,361);
    TH2F EBRing ("EBRing","EBRing",171,-85,86,360,1,361);
    for (std::vector<DetId>::const_iterator it= m_barrelCells.begin();
	 it!= m_barrelCells.end(); 
	 ++it )
      {
	EBDetId eb (*it);
	EBRing.Fill(eb.ieta(),eb.iphi(),m_RinginRegion[it->rawId()]);
	EBRegion.Fill(eb.ieta(),eb.iphi(),m_xtalRegionId[it->rawId()]);
      }

    TH2F EEPRegion ("EEPRegion", "EEPRegion",100,1,101,100,1,101);
    TH2F EEPRing ("EEPRing", "EEPRing",100,1,101,100,1,101);
    TH2F EEPRingReg ("EEPRingReg", "EEPRingReg",100,1,101,100,1,101);
    TH2F EEMRegion ("EEMRegion", "EEMRegion",100,1,101,100,1,101);
    TH2F EEMRing ("EEMRing", "EEMRing",100,1,101,100,1,101);
    TH2F EEMRingReg ("EEMRingReg", "EEMRingReg",100,1,101,100,1,101);
    //  TH1F eta ("eta","eta",250,-85,165);
    for (std::vector<DetId>::const_iterator it = m_endcapCells.begin();
	 it!= m_endcapCells.end();
	 ++it)
      {
	EEDetId ee (*it);
	if (ee.zside()>0)
	  {
	    EEPRegion.Fill(ee.ix(),ee.iy(),m_xtalRegionId[ee.rawId()]);
	    EEPRing.Fill(ee.ix(),ee.iy(),m_xtalRing[ee.rawId()]);
	    EEPRingReg.Fill(ee.ix(),ee.iy(),m_RinginRegion[ee.rawId()]);
	  }
	if (ee.zside()<0)
	  {
	    EEMRegion.Fill(ee.ix(),ee.iy(),m_xtalRegionId[ee.rawId()]);
	    EEMRing.Fill(ee.ix(),ee.iy(),m_xtalRing[ee.rawId()]);
	    EEMRingReg.Fill(ee.ix(),ee.iy(),m_RinginRegion[ee.rawId()]);
	  }    
      } 

    //  for (std::map<int,float>::iterator it=m_eta.begin();
    //        it!=m_eta.end();++it)
    //   	   eta.Fill(it->first,it->second);
    TFile out ("EBZone.root", "recreate");
    EBRegion.Write();
    EBRing.Write();
    EEPRegion.Write();
    EEPRing.Write();
    EEPRingReg.Write();
    EEMRegion.Write();
    EEMRing.Write();
    //  eta.Write();
    EEMRingReg.Write();
    out.Close();
    edm::LogInfo ("IML") <<"[InvRingCalib] Start to acquire the coeffs";
    CaloMiscalibMapEcal EBmap;
    EBmap.prefillMap ();
    MiscalibReaderFromXMLEcalBarrel barrelreader (EBmap);
    if (!m_EBcoeffFile.empty()) barrelreader.parseXMLMiscalibFile (m_EBcoeffFile);
    EcalIntercalibConstants costants (EBmap.get());
    m_barrelMap = costants.getMap();
    CaloMiscalibMapEcal EEmap ;   
    EEmap.prefillMap ();
    MiscalibReaderFromXMLEcalEndcap endcapreader (EEmap);
    if (!m_EEcoeffFile.empty()) endcapreader.parseXMLMiscalibFile (m_EEcoeffFile) ;
    EcalIntercalibConstants EEcostants (EEmap.get());
    m_endcapMap = EEcostants.getMap();

    isfirstcall_=false;
  } // if isfirstcall






  //gets the barrel recHits
  double pSubtract = 0.;
  double pTk = 0.;
  const EcalRecHitCollection* barrelHitsCollection = 0;
  edm::Handle<EBRecHitCollection> barrelRecHitsHandle ;
  iEvent.getByLabel (m_barrelAlCa, barrelRecHitsHandle) ;
  barrelHitsCollection = barrelRecHitsHandle.product () ;

 if (!barrelRecHitsHandle.isValid ()) {
     edm::LogError ("IML") << "[EcalEleCalibLooper] barrel rec hits not found" ;
     return  kContinue ;
    }
  //gets the endcap recHits
  const EcalRecHitCollection* endcapHitsCollection = 0;
  edm::Handle<EERecHitCollection> endcapRecHitsHandle ;
  iEvent.getByLabel (m_endcapAlCa, endcapRecHitsHandle) ;
  endcapHitsCollection = endcapRecHitsHandle.product () ;

 if (!endcapRecHitsHandle.isValid ()) {  
     edm::LogError ("IML") << "[EcalEleCalibLooper] endcap rec hits not found" ; 
     return kContinue;
   }

  //gets the electrons
  edm::Handle<reco::GsfElectronCollection> pElectrons;
  iEvent.getByLabel(m_ElectronLabel,pElectrons);

 if (!pElectrons.isValid ()) {
     edm::LogError ("IML")<< "[EcalEleCalibLooper] electrons not found" ;
     return kContinue;
   }

  //loops over the electrons in the event
  for (reco::GsfElectronCollection::const_iterator eleIt = pElectrons->begin();
       eleIt != pElectrons->end();
       ++eleIt )
    {
      pSubtract =0;
      pTk=eleIt->trackMomentumAtVtx().R();
      std::map<int , double> xtlMap;
      DetId Max=0; 
      if (fabs(eleIt->eta()<1.49))
	     Max = EcalClusterTools::getMaximum(eleIt->superCluster()->hitsAndFractions(),barrelHitsCollection).first;
      else 
	     Max = EcalClusterTools::getMaximum(eleIt->superCluster()->hitsAndFractions(),endcapHitsCollection).first;
      if (Max.det()==0) continue;
       m_MapFiller->fillMap(eleIt->superCluster ()->hitsAndFractions (),Max, 
                           barrelHitsCollection,endcapHitsCollection, xtlMap,pSubtract);
      if (m_xtalRegionId[Max.rawId()]==-1) continue;
      pSubtract += eleIt->superCluster()->preshowerEnergy() ;
      ++m_RingNumOfHits[m_xtalRing[Max.rawId()]];
      //fills the calibBlock 
      m_IMACalibBlocks.at(m_xtalRegionId[Max.rawId()])->Fill (
          xtlMap.begin(), xtlMap.end(),pTk,pSubtract
        ) ;
    }
  return  kContinue;
} //end of duringLoop


//-------------------------------------


//EndOfLoop
edm::EDLooper::Status 
InvRingCalib::endOfLoop (const edm::EventSetup& dumb, 
                         unsigned int iCounter)
{
   std::map<int,double> InterRings;
  edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfLoop] Start to invert the matrixes" ;
  //call the autoexplaining "solve" method for every calibBlock
  for (std::vector<VEcalCalibBlock *>::iterator calibBlock=m_IMACalibBlocks.begin();
       calibBlock!=m_IMACalibBlocks.end();
       ++calibBlock)
    (*calibBlock)->solve(m_usingBlockSolver,m_minCoeff,m_maxCoeff);

  edm::LogInfo("IML") << "[InvRingLooper][endOfLoop] Starting to write the coeffs";
  TH1F *coeffDistr = new TH1F("coeffdistr","coeffdistr",100 ,0.7,1.4);
  TH1F *coeffMap = new TH1F("coeffRingMap","coeffRingMap",250,-85,165);
  TH1F *ringDistr = new TH1F("ringDistr","ringDistr",250,-85,165);
  TH1F *RingFill = new TH1F("RingFill","RingFill",250,-85,165);
  for(std::map<int,int>::const_iterator it=m_xtalRing.begin();
      it!=m_xtalRing.end();
      ++it)
    ringDistr->Fill(it->second+0.1);

  int ID;
  std::map<int,int> flag;
  for(std::map<int,int>::const_iterator it=m_xtalRing.begin();
      it!=m_xtalRing.end();
      ++it)
    flag[it->second]=0;
  
  for (std::vector<DetId>::const_iterator it=m_barrelCells.begin();
       it!=m_barrelCells.end();
       ++it)
    { 
      ID= it->rawId();
      if (m_xtalRegionId[ID]==-1) continue;
      if (flag[m_xtalRing[ID]]) continue;
      flag[m_xtalRing[ID]] =1;
      RingFill->Fill(m_xtalRing[ID],m_RingNumOfHits[m_xtalRing[ID]]);
      InterRings[m_xtalRing[ID]] = m_IMACalibBlocks.at(m_xtalRegionId[ID])->at(m_RinginRegion[ID]);
      coeffMap->Fill (m_xtalRing[ID]+0.1,InterRings[m_xtalRing[ID]]);
      coeffDistr->Fill(InterRings[m_xtalRing[ID]]);
    }

  for (std::vector<DetId>::const_iterator it=m_endcapCells.begin();
       it!=m_endcapCells.end();
       ++it)
    { 
      ID= it->rawId();
      if (m_xtalRegionId[ID]==-1) continue;
      if (flag[m_xtalRing[ID]]) continue;
      flag[m_xtalRing[ID]]= 1;
      InterRings[m_xtalRing[ID]] = m_IMACalibBlocks.at(m_xtalRegionId[ID])->at(m_RinginRegion[ID]);
      RingFill->Fill(m_xtalRing[ID],m_RingNumOfHits[m_xtalRing[ID]]);
      coeffMap->Fill (m_xtalRing[ID],InterRings[m_xtalRing[ID]]);
      coeffDistr->Fill(InterRings[m_xtalRing[ID]]);
		
    } 
   
  char filename[80];
  sprintf(filename,"coeff%d.root",iCounter);
  TFile out(filename,"recreate");    
  coeffDistr->Write();
  coeffMap->Write();
  ringDistr->Write();
  RingFill->Write();
  out.Close();
  for (std::vector<DetId>::const_iterator it=m_barrelCells.begin();
       it!=m_barrelCells.end();
       ++it){
	 m_barrelMap[*it]*=InterRings[m_xtalRing[it->rawId()]];
  }
  for (std::vector<DetId>::const_iterator it=m_endcapCells.begin();
       it!=m_endcapCells.end();
       ++it)
	  m_endcapMap[*it]*=InterRings[m_xtalRing[it->rawId()]];
  if (iCounter < m_loops-1 ) return kContinue ;
  else return kStop; 
}


//---------------------------------------


//LP endOfJob
void 
InvRingCalib::endOfJob ()
{

 edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfJob] saving calib coeffs" ;
 calibXMLwriter barrelWriter(EcalBarrel);
 calibXMLwriter endcapWriter(EcalEndcap);
 for (std::vector<DetId>::const_iterator barrelIt =m_barrelCells.begin(); 
       barrelIt!=m_barrelCells.end(); 
       ++barrelIt) {
	    EBDetId eb (*barrelIt);
	    barrelWriter.writeLine(eb,m_barrelMap[eb]);
          }
 for (std::vector<DetId>::const_iterator endcapIt = m_endcapCells.begin();
     endcapIt!=m_endcapCells.end();
     ++endcapIt) {
	  EEDetId ee (*endcapIt);
	  endcapWriter.writeLine(ee,m_endcapMap[ee]);
	}
}


//------------------------------------//
//      definition of functions       //
//------------------------------------//

//------------------------------------------------------------


//!EE ring definition
void InvRingCalib::EERingDef(const edm::EventSetup& iSetup)  
{
 //Gets the Handle for the geometry from the eventSetup
 edm::ESHandle<CaloGeometry> geoHandle;
 iSetup.get<CaloGeometryRecord>().get(geoHandle);
 //Gets the geometry of the endcap
 const CaloGeometry& geometry = *geoHandle;
 const CaloSubdetectorGeometry *endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
// const CaloSubdetectorGeometry *barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
 //for every xtal gets the position Vector and the phi position
 
// for (std::vector<DetId>::const_iterator barrelIt = m_barrelCells.begin();
//    barrelIt!=m_barrelCells.end();
//    ++barrelIt) {
//     const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(*barrelIt);
//     GlobalPoint point;
//     EBDetId eb (*barrelIt);
//     point=cellGeometry->getPosition();
//     m_eta[eb.ieta()]=point.eta() ;    //cellGeometry->getPosition().eta();
//    }
 for (std::vector<DetId>::const_iterator endcapIt = m_endcapCells.begin();
    endcapIt!=m_endcapCells.end();
    ++endcapIt) {
     const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt);
     m_cellPos[endcapIt->rawId()] = cellGeometry->getPosition();
     m_cellPhi[endcapIt->rawId()] = cellGeometry->getPosition().phi();
  }
 //takes the first 39 xtals at a fixed y varying the x coordinate and saves their eta coordinate 
 float eta_ring[39];
 for (int ring=0; ring<39; ring++) 
	if (EEDetId::validDetId(ring,50,1)){  
	  EEDetId det = EEDetId (ring,50,1,EEDetId::XYMODE);
          eta_ring[ring]=m_cellPos[det.rawId()].eta();
	  }
 //defines the bonduary of the rings as the average eta of a xtal
 double etaBonduary[40];
 etaBonduary[0]=1.49;
 etaBonduary[39]=4.0;
 for (int ring=1; ring<39; ++ring)
       etaBonduary[ring]=(eta_ring[ring]+eta_ring[ring-1])/2.;
 //assign to each xtal a ring number
 int CRing;
 for (int ring=0; ring<39; ring++) 
   for (std::vector<DetId>::const_iterator endcapIt=m_endcapCells.begin();
        endcapIt!=m_endcapCells.end();++endcapIt){
     if (fabs(m_cellPos[endcapIt->rawId()].eta())>etaBonduary[ring] &&
         fabs(m_cellPos[endcapIt->rawId()].eta())<etaBonduary[ring+1])
	  {
	      EEDetId ee(*endcapIt);
	      if (ee.zside()>0) CRing=ring + 86; 
	      else CRing = ring + 125;
              m_xtalRing[endcapIt->rawId()]=CRing;
//              m_eta[CRing]=m_cellPos[endcapIt->rawId()].eta();
	  }    
      }
 return;
}


//------------------------------------------------------------


//!Gives the Id of the region given the id of the xtal
int InvRingCalib::EERegId( int id) 
{
   int reg;
   int ring;
   EEDetId ee (id);
  //sets the reg to -1 if the ring doesn't exist or is outside the region of interest 
   if (m_xtalRing[id] == -1) return -1;
  //avoid the calibration in the wrong zside
   if (m_EEZone == 1 ){
   if (ee.zside()<0) return -1;
   ring = m_xtalRing[id]-86;
   if(ring >=m_endRing) return -1;
   if (ring<m_startRing) return -1;
   reg = (ring -m_startRing) / m_etaWidth;
   m_RinginRegion[id]=(ring -m_startRing) % m_etaWidth;
   return reg;
   }
   if (m_EEZone == -1){
   if (ee.zside()>0) return -1;
   ring = m_xtalRing[id] -125;
   if(ring >=m_endRing) return -1;
   if (ring<m_startRing) return -1;
   reg = (ring -m_startRing) / m_etaWidth;
   m_RinginRegion[id]=(ring -m_startRing) % m_etaWidth;
   return reg;
   }
   if (ee.zside()>0) ring=m_xtalRing[id]-86;
     else ring = m_xtalRing[id]-125;
   if(ring >=m_endRing) return -1;
   if (ring<m_startRing) return -1;
   reg = (ring -m_startRing) / m_etaWidth;
   m_RinginRegion[id]=(ring -m_startRing) % m_etaWidth;
   return reg;
}
//----------------------------------------
//!Loops over all the endcap xtals and sets for each xtal the value of the region
//!the xtal is in, and the ringNumber inside the region 
void InvRingCalib::EERegionDef ()
{
int reg;
for (std::vector<DetId>::const_iterator endcapIt=m_endcapCells.begin();
     endcapIt!=m_endcapCells.end();++endcapIt){
      EEDetId ee(*endcapIt);
      reg = EERegId(endcapIt->rawId());
      //If the ring is not of interest saves only the region Id(-1)
      if(reg==-1) 
         m_xtalRegionId[endcapIt->rawId()]=reg;
      //sums the number of region in EB or EB+EE to have different regionsId in different regions 
      else {
      if (ee.zside()>0)reg += EBRegionNum();
      else reg += EBRegionNum()+EERegionNum();
      m_xtalRegionId[endcapIt->rawId()]=reg;
   }
  }
}


//------------------------------------------------------------


//!Number of Regions in EE 
inline int InvRingCalib::EERegionNum () const 
{
  return ((m_endRing - m_startRing)/m_etaWidth);
}


//! number of Ring in EB
int InvRingCalib::EBRegionNum () const 
{
  if ((m_etaEnd*m_etaStart)>0)
   return ((m_etaEnd - m_etaStart )/m_etaWidth); 
  
  if ((m_etaEnd*m_etaStart)<0)
   return ((m_etaEnd - m_etaStart-1 )/m_etaWidth); 
  
  return 0;
}
//!Divides the barrel in region, necessary to take into
//! account the missing 0 xtal
void InvRingCalib::RegPrepare()
{
 int k=0;
 for (int i = m_etaStart;i<m_etaEnd;++i)
 {
  if (i==0) continue;
  m_Reg[i]=k/m_etaWidth;
  ++k;
 }
}
//! gives the region Id given ieta
int InvRingCalib::EBRegId(const int ieta) 
{
 if (ieta<m_etaStart || ieta>=m_etaEnd) return -1;
 else return (m_Reg[ieta]);
}


//------------------------------------------------------------


//EB Region Definition
void InvRingCalib::EBRegionDef()
{
  RegPrepare();
  for (std::vector<DetId>::const_iterator it=m_barrelCells.begin();
  	it!=m_barrelCells.end();++it)
  {
    EBDetId eb (it->rawId());
    m_xtalRing[eb.rawId()] = eb.ieta() ;
    m_xtalRegionId[eb.rawId()] = EBRegId (eb.ieta()); 
    if (m_xtalRegionId[eb.rawId()]==-1) continue;
    m_RinginRegion[eb.rawId()] = (eb.ieta() - m_etaStart)% m_etaWidth; 
  }
}
//------------------------------------------------------------
