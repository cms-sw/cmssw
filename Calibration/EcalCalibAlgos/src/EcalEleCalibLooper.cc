#include <memory>
#include <cmath>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Calibration/EcalCalibAlgos/interface/IMACalibBlock.h"
#include "Calibration/EcalCalibAlgos/interface/L3CalibBlock.h"
#include "Calibration/EcalCalibAlgos/interface/EcalEleCalibLooper.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "Calibration/EcalCalibAlgos/interface/MatrixFillMap.h"
#include "Calibration/EcalCalibAlgos/interface/ClusterFillMap.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"



//----------------------------------------------------------------


//!LP ctor
EcalEleCalibLooper::EcalEleCalibLooper (const edm::ParameterSet& iConfig) :
      m_barrelAlCa (iConfig.getParameter<edm::InputTag> ("alcaBarrelHitCollection")) ,
      m_endcapAlCa (iConfig.getParameter<edm::InputTag> ("alcaEndcapHitCollection")) ,
      m_recoWindowSidex (iConfig.getParameter<int> ("recoWindowSidex")) ,
      m_recoWindowSidey (iConfig.getParameter<int> ("recoWindowSidey")) ,
      m_etaWidth (iConfig.getParameter<int> ("etaWidth")) ,
      m_phiWidthEB (iConfig.getParameter<int> ("phiWidthEB")) ,
      m_etaStart (etaShifter (iConfig.getParameter<int> ("etaStart"))) , 
      m_etaEnd (etaShifter (iConfig.getParameter<int> ("etaEnd"))) ,
      m_phiStartEB (iConfig.getParameter<int> ("phiStartEB")) , 
      m_phiEndEB (iConfig.getParameter<int> ("phiEndEB")),
      m_radStart (iConfig.getParameter<int> ("radStart")) , 
      m_radEnd (iConfig.getParameter<int> ("radEnd")) ,
      m_radWidth (iConfig.getParameter<int> ("radWidth")) ,
      m_phiStartEE (iConfig.getParameter<int> ("phiStartEE")) ,
      m_phiEndEE (iConfig.getParameter<int> ("phiEndEE")) ,
      m_phiWidthEE (iConfig.getParameter<int> ("phiWidthEE")) ,
      m_maxSelectedNumPerXtal (iConfig.getParameter<int> ("maxSelectedNumPerCrystal")) ,  
      m_minEnergyPerCrystal (iConfig.getParameter<double> ("minEnergyPerCrystal")),
      m_maxEnergyPerCrystal (iConfig.getParameter<double> ("maxEnergyPerCrystal")) ,
      m_minCoeff (iConfig.getParameter<double> ("minCoeff")) ,
      m_maxCoeff (iConfig.getParameter<double> ("maxCoeff")) ,
      m_usingBlockSolver (iConfig.getParameter<int> ("usingBlockSolver")) ,
      m_loops (iConfig.getParameter<int> ("loops")),
      m_ElectronLabel (iConfig.getParameter<edm::InputTag> ("electronLabel"))
{
  edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] asserts" ;
  assert (!((m_etaEnd - m_etaStart )%m_etaWidth)); 

  assert (m_etaStart >=0 && m_etaStart <= 171);
  assert (m_etaEnd >= m_etaStart && m_etaEnd <= 171);
  assert ( (m_radEnd - m_radStart)%m_radWidth == 0) ; 
  assert (m_radStart >=0 && m_radStart <= 50);
  assert (m_radEnd >= m_radStart && m_radEnd <= 50);
  edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] entering " ;
  edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] region definition" ;
  EBRegionDefinition () ;
  EERegionDefinition () ;
///Graphs to ckeck the region definition
  TH2F * EBRegion = new TH2F ("EBRegion","EBRegion",170,0,170,360,0,360) ;
  for (int eta = 0; eta<170; ++eta)
     for (int phi = 0; phi <360; ++phi){
  	EBRegion->Fill (eta, phi,m_xtalRegionId[EBDetId::unhashIndex(eta*360+phi).rawId()] );
      }
  TH2F * EERegion = new TH2F ("EERegion", "EERegion",100,0,100,100,0,100);
  for (int x = 0; x<100; ++x)
     for (int y = 0; y<100;++y){
   if(EEDetId::validDetId(x+1,y+1,1))
      	     EERegion->Fill(x,y,m_xtalRegionId[EEDetId(x+1,y+1,-1).rawId()]);
     }
         
  TFile out ("EBZone.root", "recreate");
    EBRegion->Write ();
    EERegion->Write ();
  out.Close ();
  delete EERegion;
  delete EBRegion;
  ///End of Graphs

  //PG build the calibration algorithms for the regions
  //PG ------------------------------------------------

  edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] Calib Block" ;
  std::string algorithm = iConfig.getParameter<std::string> ("algorithm") ;
  int eventWeight = iConfig.getUntrackedParameter<int> ("L3EventWeight",1) ;

  //PG loop over the regions set
  for (int region = 0 ; 
       region < EBregionsNum () + 2 * EEregionsNum () ; 
       ++region)
    {   
      if (algorithm == "IMA")
        m_EcalCalibBlocks.push_back (
            new IMACalibBlock (m_regions.at (region))
          ) ; 
      else if (algorithm == "L3")
        m_EcalCalibBlocks.push_back (
            new L3CalibBlock (m_regions.at (region), eventWeight)
          ) ; 
      else
        {
          edm::LogError ("building") << algorithm 
                          << " is not a valid calibration algorithm" ;
          exit (1) ;    
        }    
    } //PG loop over the regions set
  std::string mapFiller = iConfig.getParameter<std::string> ("FillType");
  if (mapFiller == "Cluster") m_MapFiller= new ClusterFillMap (
	m_recoWindowSidex ,m_recoWindowSidey ,
        m_xtalRegionId ,m_minEnergyPerCrystal ,
        m_maxEnergyPerCrystal , m_xtalPositionInRegion ,
        & m_barrelMap ,
        & m_endcapMap ); 
  if (mapFiller == "Matrix") m_MapFiller = new MatrixFillMap (
	m_recoWindowSidex ,m_recoWindowSidey ,
        m_xtalRegionId , m_minEnergyPerCrystal ,
        m_maxEnergyPerCrystal , m_xtalPositionInRegion ,
        & m_barrelMap ,
        & m_endcapMap); 
 } //end ctor


//---------------------------------------------------------------------------


//!LP destructor
EcalEleCalibLooper::~EcalEleCalibLooper ()
{
  edm::LogInfo ("IML") << "[EcalEleCalibLooper][dtor]" ;
  for (std::vector<VEcalCalibBlock *>::iterator calibBlock = m_EcalCalibBlocks.begin () ;
       calibBlock != m_EcalCalibBlocks.end () ;
       ++calibBlock) 
    delete (*calibBlock) ;

}


//---------------------------------------------------------------------------


//!BeginOfJob
void 
EcalEleCalibLooper::beginOfJob () 
{
  isfirstcall_=true;
 
}


//----------------------------------------------------------------------


//!startingNewLoop
//!empties the map of the calibBlock so that it can be filled
void EcalEleCalibLooper::startingNewLoop (unsigned int ciclo) 
{
  edm::LogInfo ("IML") << "[InvMatrixCalibLooper][Start] entering loop " << ciclo;

 

  for (std::vector<VEcalCalibBlock *>::iterator calibBlock = m_EcalCalibBlocks.begin () ;
       calibBlock != m_EcalCalibBlocks.end () ;
       ++calibBlock) 
          (*calibBlock)->reset ();
  for (std::map<int,int>::iterator it= m_xtalNumOfHits.begin();
       it!=m_xtalNumOfHits.end();
       ++it)
    it->second = 0 ;
 return ;
}


//----------------------------------------------------------------------


//!duringLoop
//!return the status Kcontinue, fills the calibBlock with the recHits
edm::EDLooper::Status
EcalEleCalibLooper::duringLoop (const edm::Event& iEvent,
                             const edm::EventSetup& iSetup) 
{

  // this chunk used to belong to beginJob(isetup). Moved here
  // with the beginJob without arguments migration
  
  if (isfirstcall_){
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord> ().get (geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  m_barrelCells = geometry.getValidDetIds (DetId::Ecal, EcalBarrel);
  m_endcapCells = geometry.getValidDetIds (DetId::Ecal, EcalEndcap);
    for (std::vector<DetId>::const_iterator barrelIt=m_barrelCells.begin();
	 barrelIt!=m_barrelCells.end();++barrelIt){
      m_barrelMap[*barrelIt]=1;
      m_xtalNumOfHits[barrelIt->rawId()]=0;
    }
    for (std::vector<DetId>::const_iterator endcapIt=m_endcapCells.begin();
	 endcapIt!=m_endcapCells.end();++endcapIt){
      m_endcapMap[*endcapIt]=1;
      m_xtalNumOfHits[endcapIt->rawId()]=0;
    }
    
    isfirstcall_=false; 
  }
  


 //take the collection of recHits in the barrel
 const EBRecHitCollection* barrelHitsCollection = 0;
 edm::Handle<EBRecHitCollection> barrelRecHitsHandle ;
 iEvent.getByLabel (m_barrelAlCa, barrelRecHitsHandle) ;
 barrelHitsCollection = barrelRecHitsHandle.product () ;
 if (!barrelRecHitsHandle.isValid ()) {
     edm::LogError ("reading") << "[EcalEleCalibLooper] barrel rec hits not found" ;
     return  kContinue ;
    }

 //take the collection of rechis in the endcap
 const EERecHitCollection * endcapHitsCollection = 0 ;
 edm::Handle<EERecHitCollection> endcapRecHitsHandle ;
 iEvent.getByLabel (m_endcapAlCa, endcapRecHitsHandle) ;
 endcapHitsCollection = endcapRecHitsHandle.product () ;
 if (!endcapRecHitsHandle.isValid ()) {  
     edm::LogError ("reading") << "[EcalEleCalibLooper] endcap rec hits not found" ; 
     return kContinue;
   }

 //Takes the electron collection of the pixel detector
 edm::Handle<reco::GsfElectronCollection> pElectrons;
 iEvent.getByLabel (m_ElectronLabel,pElectrons);
 if (!pElectrons.isValid ()) {
     edm::LogError ("reading")<< "[EcalEleCalibLooper] electrons not found" ;
     return kContinue;
   }

 //Start the loop over the electrons 
 for (reco::GsfElectronCollection::const_iterator eleIt = pElectrons->begin ();
      eleIt != pElectrons->end ();
      ++eleIt )
   {
     double pSubtract = 0 ;
     double pTk = 0 ;
     std::map<int , double> xtlMap;
     DetId Max =0;
     if (fabs(eleIt->eta()<1.49))
	     Max = EcalClusterTools::getMaximum(eleIt->superCluster()->hitsAndFractions(),barrelHitsCollection).first;
     else 
	     Max = EcalClusterTools::getMaximum(eleIt->superCluster()->hitsAndFractions(),endcapHitsCollection).first;
     if (Max.det()==0) continue;
     m_MapFiller->fillMap(eleIt->superCluster ()->hitsAndFractions (),Max, 
                           barrelHitsCollection,endcapHitsCollection, xtlMap,pSubtract);
     if (m_maxSelectedNumPerXtal > 0 && 
        m_xtalNumOfHits[Max.rawId ()] > m_maxSelectedNumPerXtal ) continue;
     ++m_xtalNumOfHits[Max.rawId()];
     if (m_xtalRegionId[Max.rawId()]==-1) continue;
     pTk = eleIt->trackMomentumAtVtx ().R ();
     m_EcalCalibBlocks.at (m_xtalRegionId[Max.rawId()])->Fill (xtlMap.begin (), 
		                                   xtlMap.end (),pTk,pSubtract) ;
   } //End of the loop over the electron collection

  return  kContinue;
} //end of duringLoop


//----------------------------------------------------------------------


//! EndOfLoop
//!Return kContinue if there's still another loop to be done; otherwise stops returnig kStop;
//!Takes the coefficients solving the calibBlock;
edm::EDLooper::Status EcalEleCalibLooper::endOfLoop (const edm::EventSetup& dumb,unsigned int iCounter)
{
 edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfLoop] entering..." ;
 for (std::vector<VEcalCalibBlock *>::iterator calibBlock = m_EcalCalibBlocks.begin ();
       calibBlock!=m_EcalCalibBlocks.end ();
       ++calibBlock) 
   (*calibBlock)->solve (m_usingBlockSolver, m_minCoeff,m_maxCoeff);

  TH1F * EBcoeffEnd = new TH1F ("EBRegion","EBRegion",100,0.5,2.1) ;
  TH2F * EBcoeffMap = new TH2F ("EBcoeff","EBcoeff",171,-85,85,360,1,361);
  TH1F * EEPcoeffEnd = new TH1F ("EEPRegion", "EEPRegion",100,0.5,2.1);
  TH1F * EEMcoeffEnd = new TH1F ("EEMRegion", "EEMRegion",100,0.5,2.1);
  TH2F * EEPcoeffMap = new TH2F ("EEPcoeffMap","EEPcoeffMap",101,1,101,101,0,101);
  TH2F * EEMcoeffMap = new TH2F ("EEMcoeffMap","EEMcoeffMap",101,1,101,101,0,101);
 //loop over the barrel xtals to get the coeffs
 for (std::vector<DetId>::const_iterator barrelIt=m_barrelCells.begin();
       barrelIt!=m_barrelCells.end();++barrelIt)
        {
          EBDetId ee (*barrelIt);
          int index= barrelIt->rawId();
          if(m_xtalRegionId[index]==-1)continue;
          m_barrelMap[*barrelIt] *= 
              m_EcalCalibBlocks.at(m_xtalRegionId[index])->at(m_xtalPositionInRegion[index]);
          EBcoeffEnd->Fill(m_barrelMap[*barrelIt]);
          EBcoeffMap->Fill(ee.ieta(),ee.iphi(),m_barrelMap[*barrelIt]);
        } //PG loop over phi

  // loop over the EndCap to get the recalib coefficients
    for(std::vector<DetId>::const_iterator endcapIt=m_endcapCells.begin();
         endcapIt!=m_endcapCells.end();++endcapIt)
    {
     EEDetId ee (*endcapIt);
     int index =endcapIt->rawId(); 
     if (ee.zside()>0) 
        { 
          if (m_xtalRegionId[index]==-1) continue ;
          m_endcapMap[*endcapIt] *= 
             m_EcalCalibBlocks.at (m_xtalRegionId[index])->at (m_xtalPositionInRegion[index]);
          EEPcoeffEnd->Fill (m_endcapMap[*endcapIt]) ;
          EEPcoeffMap->Fill (ee.ix(),ee.iy(),m_endcapMap[*endcapIt]) ;
        }
      else
        {
          m_endcapMap[*endcapIt] *= 
            m_EcalCalibBlocks.at (m_xtalRegionId[index])->at (m_xtalPositionInRegion[index]);
          EEMcoeffEnd->Fill (m_endcapMap[*endcapIt]) ;
          EEMcoeffMap->Fill (ee.ix(),ee.iy(),m_endcapMap[*endcapIt]) ;
        }
    } // loop over the EndCap to get the recalib coefficients

  edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfLoop] End of endOfLoop" ;

  char filename[80];
  sprintf(filename,"coeffs%d.root",iCounter);
  TFile zout (filename, "recreate");
  EBcoeffEnd->Write () ;
  EBcoeffMap->Write () ;
  EEPcoeffEnd->Write () ;
  EEPcoeffMap->Write () ;
  EEMcoeffEnd->Write () ;
  EEMcoeffMap->Write () ;
  zout.Close () ;
  delete EBcoeffEnd;
  delete EBcoeffMap;
  delete EEPcoeffEnd;
  delete EEMcoeffEnd;
  delete EEPcoeffMap;
  delete EEMcoeffMap;
  if (iCounter < m_loops-1 ) return kContinue ;
  else return kStop; 
}


//-------------------------------------------------------------------


//!LP endOfJob
//!writes the coefficients in the xml format and exits
void 
EcalEleCalibLooper::endOfJob ()
{
 edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfJob] saving calib coeffs" ;

//Writes the coeffs 
 calibXMLwriter barrelWriter (EcalBarrel);
 calibXMLwriter endcapWriter (EcalEndcap);
 for (std::vector<DetId>::const_iterator barrelIt = m_barrelCells.begin (); 
       barrelIt!=m_barrelCells.end (); 
       ++barrelIt) 
   {
     EBDetId eb (*barrelIt);
     barrelWriter.writeLine (eb,m_barrelMap[*barrelIt]);
   }
 for (std::vector<DetId>::const_iterator endcapIt = m_endcapCells.begin ();
      endcapIt!=m_endcapCells.end ();
      ++endcapIt) 
   {
     EEDetId ee (*endcapIt);
     endcapWriter.writeLine (ee,m_endcapMap[*endcapIt]);
   }
 edm::LogInfo ("IML") << "[InvMatrixCalibLooper][endOfJob] Exiting" ;    
}


//------------------------------------//
//      definition of functions       //
//------------------------------------//


//! Tells if you are in the region to be calibrated
int 
EcalEleCalibLooper::EBregionCheck (const int eta, const int phi) const 
 {
   if (eta < m_etaStart) return 1 ;
   if (eta >= m_etaEnd)   return 2 ;
   if (phi < m_phiStartEB) return 3 ;
   if (phi >= m_phiEndEB)   return 4 ;
   return 0 ;
 }


//--------------------------------------------


//! def degrees
inline double degrees (double radiants)
 {
  return radiants * 180 * (1/M_PI) ;
 }


//--------------------------------------------


//!DS def radiants
inline double radiants (int degrees)
 {
   return degrees * M_PI * (1 / 180) ;  
 }    


//--------------------------------------------


//! copes with the infinitives of the tangent
double EcalEleCalibLooper::giveLimit (int degrees)
  {
    //PG 200 > atan (50/0.5)
    if (degrees == 90) return 90 ; 
    return tan (radiants (degrees)) ;      
  } 


//--------------------------------------------


//! autoexplaining
inline double Mod (double phi)
 {
  if (phi>=360 && phi<720) return phi-360;
  if (phi>=720) return phi-720;
  return phi;
 } 


//----------------------------------------


//!Reg Id generator EB ----- for the barrel
int EcalEleCalibLooper::EBRegionId (const int etaXtl,const int phiXtl) const 
{
 if (EBregionCheck(etaXtl,phiXtl)) return -1;
 int phifake = m_phiStartEB;
 if (m_phiStartEB>m_phiEndEB) phifake = m_phiStartEB - 360;
 int Nphi = (m_phiEndEB-phifake)/m_phiWidthEB ;
 int etaI = (etaXtl-m_etaStart) / m_etaWidth ;  
 int phiI = (phiXtl-m_phiStartEB) / m_phiWidthEB ; 
 int regionNumEB = phiI + Nphi*etaI ;
 return (int) regionNumEB;
}


//----------------------------------------


//! Gives the id of the region
int EcalEleCalibLooper::EERegionId (const int ics, const int ips) const
{
 if (EEregionCheck(ics,ips)) return -1;
 int phifake = m_phiStartEE;
 if (m_phiStartEE>m_phiEndEE) phifake = m_phiStartEE - 360;
 double radius = (ics-50) * (ics-50) + (ips-50) * (ips-50) ;
 radius = sqrt (radius) ;
 int Nphi = (m_phiEndEE - phifake)/m_phiWidthEE ;
 double phi = atan2 (static_cast<double> (ips-50), 
                     static_cast<double> (ics-50)) ;
 phi = degrees (phi);
 if (phi < 0) phi += 360; 
 int radI = static_cast<int> ((radius-m_radStart) / m_radWidth) ;
 int phiI = static_cast<int> ((m_phiEndEE-phi) / m_phiWidthEE) ;
 int regionNumEE = phiI + Nphi*radI ;
 return  regionNumEE ;
}


//----------------------------------------


//!DS Number of regions in EE 
inline int EcalEleCalibLooper::EEregionsNum () const 
{
  int phifake = m_phiStartEE;
  if (m_phiStartEE>m_phiEndEE) phifake = m_phiStartEE - 360;
  return ( (m_radEnd - m_radStart)/m_radWidth) * ( (m_phiEndEE - phifake)/m_phiWidthEE) ;
}


//----------------------------------------


//!DS number of regions in EB
inline int EcalEleCalibLooper::EBregionsNum () const 
{
  int phi = m_phiStartEB;
  if (m_phiStartEB>m_phiEndEB) phi = m_phiStartEB - 360;
  return ( (m_etaEnd - m_etaStart)/m_etaWidth) * ( (m_phiEndEB - phi)/m_phiWidthEB) ; 
}


//----------------------------------------


//!DS EB Region Definition
void EcalEleCalibLooper::EBRegionDefinition ()
{
 int reg=-1;
 for (int it = 0 ; it < EBregionsNum () ; ++it) m_regions.push_back (0) ;   
 for (int eta = 0 ; eta < 170  ; ++eta)
   for (int phi = 0 ; phi < 360 ; ++phi)
      {
        reg = EBRegionId (eta,phi) ;
        m_xtalRegionId[EBDetId::unhashIndex (eta*360+phi).rawId ()] = reg ; 
	if (reg==-1) continue;
        m_xtalPositionInRegion[EBDetId::unhashIndex (eta*360+phi).rawId ()] = m_regions.at (reg) ;
        ++m_regions.at (reg);
      }
}


//----------------------------------------


//DS EE Region Definition 
void EcalEleCalibLooper::EERegionDefinition ()
{
 // reset
 int EBnum=EBregionsNum();
 int EEnum=EEregionsNum();
 for (int it = 0 ; it < 2* EEnum ; ++it) m_regions.push_back (0) ;   
 // loop sui xtl 
 int reg=-1;
 for (int ics = 0 ; ics < 100 ; ++ics)
  for (int ips = 0 ; ips < 100 ; ++ips)
    {
     int ireg = EERegionId(ics, ips);
     if (ireg==-1) reg =-1;
     else reg = EBnum + ireg;
     if (EEDetId::validDetId (ics+1, ips+1, 1))
      {
        m_xtalRegionId[EEDetId (ics+1, ips+1, 1).rawId ()] = reg ; 
        if (reg==-1) continue;
        m_xtalPositionInRegion[EEDetId (ics+1, ips+1, 1).rawId ()] = m_regions.at (reg) ;
        ++m_regions.at(reg);
      }
     if (reg!=-1) reg += EEnum; 
     if (EEDetId::validDetId (ics+1, ips+1, -1))
      {
        m_xtalRegionId[EEDetId (ics+1, ips+1, -1).rawId ()] = reg ; 
        if (reg==-1) continue;
        m_xtalPositionInRegion[EEDetId (ics+1, ips+1, -1).rawId ()] = m_regions.at (reg) ;
        ++m_regions.at (reg) ;
       }
    }
}


//-----------------------------------------


//!returns zero if the coordinates are in the right place.
int EcalEleCalibLooper::EEregionCheck (const int ics, const int ips)  const
{
  int x = ics-50;
  int y = ips-50;
  double radius2 = x*x + y*y ;
  if (radius2 < 10*10) return 1;  //center of the donut
  if (radius2 > 50*50) return 1;  //outer part of the donut
  if (radius2 < m_radStart * m_radStart) return 2 ;
  if (radius2 >= m_radEnd * m_radEnd) return 2 ;
  double phi = atan2 (static_cast<double> (y),static_cast<double> (x));
  phi = degrees (phi);
  if (phi < 0) phi += 360; 
  if (m_phiStartEE < m_phiEndEE 
     && phi > m_phiStartEE && phi < m_phiEndEE ) return 0; 
  if (m_phiStartEE > m_phiEndEE 
      && (phi > m_phiStartEE || phi < m_phiEndEE )) return 0; 
   return 3;
}


//--------------------------------------------


//Shifts eta in other coordinates (from 0 to 170)
int EcalEleCalibLooper::etaShifter (const int etaOld) const
   {
     if (etaOld < 0) return etaOld + 85;
     else if (etaOld > 0) return etaOld + 84;
     assert(0!=etaOld); // etaOld = 0, apparently not a foreseen value, so fail
     return 999; // dummy statement to silence compiler warning
   }


//--------------------------------------------
