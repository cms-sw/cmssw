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
//LP includes to read/write the original coeff
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
//DS verify all these include
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
//DS LogMessages
#include "FWCore/MessageLogger/interface/MessageLogger.h" 



//----------------------------------------------------------------


//!LP ctor
EcalEleCalibLooper::EcalEleCalibLooper (const edm::ParameterSet& iConfig) :
      m_barrelAlCa (iConfig.getParameter<edm::InputTag> ("alcaBarrelHitCollection")) ,
      m_endcapAlCa (iConfig.getParameter<edm::InputTag> ("alcaEndcapHitCollection")) ,
      m_recoWindowSide (iConfig.getParameter<int> ("recoWindowSide")) ,
      m_etaWidth (iConfig.getParameter<int> ("etaWidth")) ,
 //PG fin dove andare a cercare cristalli da aggiungere a pSubtract
 //PG fuori dalla regione da calibrare     
      //m_etaBorder (iConfig.getParameter<int> ("etaBorder")) ,
      m_phiWidthEB (iConfig.getParameter<int> ("phiWidthEB")) ,
 //PG fin dove andare a cercare cristalli da aggiungere a pSubtract
 //PG fuori dalla regione da calibrare (ci vuole anche per EE)    
      //m_phiBorderEB (iConfig.getParameter<int> ("phiBorderEB")) ,
      m_etaStart (etaShifter (iConfig.getParameter<int> ("etaStart"))) , 
      m_etaEnd (etaShifter (iConfig.getParameter<int> ("etaEnd"))) ,
      m_phiStartEB (iConfig.getParameter<int> ("phiStartEB")) , 
      m_phiEndEB (iConfig.getParameter<int> ("phiEndEB")),
      m_radStart (iConfig.getParameter<int> ("radStart")) , 
      m_radEnd (iConfig.getParameter<int> ("radEnd")) ,
      m_radWidth (iConfig.getParameter<int> ("radWidth")) ,
 //PG fin dove andare a cercare cristalli da aggiungere a pSubtract
 //PG fuori dalla regione da calibrare     
      //m_radBorder (iConfig.getParameter<int> ("radBorder")),
      m_phiStartEE (iConfig.getParameter<int> ("phiStartEE")) ,
      m_phiEndEE (iConfig.getParameter<int> ("phiEndEE")) ,
      m_phiWidthEE (iConfig.getParameter<int> ("phiWidthEE")) ,
//PG per applicare tagli al punto di impatto
//PG sulla faccia frontale del cristallo, importante per il testbeam      
//FIXME      m_halfXBand (iConfig.getParameter<double> ("halfXBand")) ,
//FIXME      m_halfYBand (iConfig.getParameter<double> ("halfYBand")) ,
      m_maxSelectedNumPerXtal (iConfig.getParameter<int> ("maxSelectedNumPerCrystal")) ,  
      m_minEnergyPerCrystal (iConfig.getParameter<double> ("minEnergyPerCrystal")),
      m_maxEnergyPerCrystal (iConfig.getParameter<double> ("maxEnergyPerCrystal")) ,
      m_minCoeff (iConfig.getParameter<double> ("minCoeff")) ,
      m_maxCoeff (iConfig.getParameter<double> ("maxCoeff")) ,
      m_usingBlockSolver (iConfig.getParameter<int> ("usingBlockSolver")) ,
      //m_minAccept (iConfig.getParameter<double> ("minAccept")) ,
      //m_maxAccept (iConfig.getParameter<double> ("maxAccept")) ,
      m_loops (iConfig.getParameter<int> ("loops")),
      m_ElectronLabel (iConfig.getParameter<edm::InputTag> ("electronLabel"))
  //Controls the parameters and their conversions
{
   edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] asserts" ;
   assert ( (m_radEnd - m_radStart)%m_radWidth == 0) ; 
   assert ( (m_etaEnd-m_etaStart)%m_etaWidth == 0) ; 
   assert (m_etaStart >=0 && m_etaStart < 170);
   assert (m_etaEnd >= m_etaStart && m_etaEnd <= 170);
//        assert (m_phiStartEB >=0 && m_phiStartEB < 360);
//        assert (m_phiEndEB >= m_phiStartEB && m_phiEndEB <= 360);
//PG questi due si possono sostituire con 
//PG m_phiStartEE %= 360 ;
//PG if (m_phiStartEE < 0) m_phiStartEE += 360 ;
//PG ed analogo per l'end, credo
//        assert (m_phiStartEE >=0); // || m_phiStartEE < 360);
//        assert (m_phiEndEE >=0); // m_phiStartEE || m_phiEndEE < 360);
   assert (m_radStart >=0 && m_radStart <= 50);
   assert (m_radEnd >= m_radStart && m_radEnd <= 50);
   edm::LogInfo ("IML") << "[EcalEleCalibLooper][ctor] entering " ;
   //PG FIXME questi posso farli direttamente con barrelcells
   int index;
   for (int a=0; a<170; ++a)
     for (int b=0; b<360; ++b)
       {
         index = EBDetId::unhashIndex (a*360+b).rawId ();
         m_recalibMap[index] = 1. ;
         m_xtalNumOfHits[index] = 0 ;
       }
   for (int i=0; i<100; ++i)
    for (int j=0; j<100; ++j)
      {
        if (EEDetId::validDetId (i+1, j+1, 1))
          {
            index = EEDetId (i+1, j+1, 1).rawId ();
            m_xtalNumOfHits[index]=0;          
            m_recalibMap[index] = 1. ;
          }
        if (EEDetId::validDetId (i+1, j+1, -1))
          {
            index = EEDetId (i+1, j+1, -1).rawId ();
            m_recalibMap[index]= 1. ;
            m_xtalNumOfHits[index]=0;
          }
      }
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
EcalEleCalibLooper::beginOfJob (const edm::EventSetup & iSetup) 
{
  edm::LogInfo ("IML") << "[EcalEleCalibLooper][beginOfJob]" ;
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord> ().get (geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  m_barrelCells = geometry.getValidDetIds (DetId::Ecal, EcalBarrel);
  m_endcapCells = geometry.getValidDetIds (DetId::Ecal, EcalEndcap);
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
    (*calibBlock)->reset () ;
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
                             const edm::EventSetup&) 
{
 const EBRecHitCollection* barrelHitsCollection = 0;
 edm::Handle<EBRecHitCollection> barrelRecHitsHandle ;
 iEvent.getByLabel (m_barrelAlCa, barrelRecHitsHandle) ;
 barrelHitsCollection = barrelRecHitsHandle.product () ;
 if (!barrelRecHitsHandle.isValid ()) {
     edm::LogError ("reading") << "[EcalEleCalibLooper] barrel rec hits not found" ;
     return  kContinue ;//maybe FIXME not with a kContinue but a skip only on the barrel part;
    }

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
 for (eleIterator eleIt = pElectrons->begin ();
      eleIt != pElectrons->end ();
      ++eleIt )
   {
     double pSubtract = 0 ;
     double pTk = 0 ;
     DetId Max = findMaxHit (eleIt->superCluster ()->getHitsByDetId (), 
                             barrelHitsCollection,  endcapHitsCollection) ;
     // Continues if the findMaxHit doesn't find anything
     if (Max.det()==0) continue; 

     if (m_maxSelectedNumPerXtal > 0 && 
        m_xtalNumOfHits[Max.rawId ()] > m_maxSelectedNumPerXtal ) continue;
     ++m_xtalNumOfHits[Max.rawId()];
     std::map<int , double> xtlMap;
     int blockIndex =  m_xtalRegionId[Max.rawId ()] ;
     pTk = eleIt->trackMomentumAtVtx ().R ();
     if  ( Max.subdetId () == EcalBarrel  )
       {
         EBDetId EBmax = Max;
         if (EBregionCheck (etaShifter (EBmax.ieta ()), EBmax.iphi ()-1)) continue;//IN the future FIXME
         fillEBMap (EBmax, barrelHitsCollection, xtlMap,
                    blockIndex, pSubtract );
       }
     else 
       {
         EEDetId EEmax = Max;
	       if (EEregionCheck (EEmax.ix ()-1, EEmax.iy ()-1)) continue ;
         fillEEMap (EEmax, endcapHitsCollection, xtlMap,
                    blockIndex, pSubtract ) ;
         pSubtract += eleIt->superCluster ()->preshowerEnergy () ;          
       }
     m_EcalCalibBlocks.at (blockIndex)->Fill (xtlMap.begin (), xtlMap.end (),pTk,pSubtract) ;
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
          m_recalibMap[index] *= 
              m_EcalCalibBlocks.at(m_xtalRegionId[index])->at(m_xtalPositionInRegion[index]);
          EBcoeffEnd->Fill(m_recalibMap[index]);
          EBcoeffMap->Fill(ee.ieta(),ee.iphi(),m_recalibMap[index]);
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
          m_recalibMap[index] *= 
             m_EcalCalibBlocks.at (m_xtalRegionId[index])->at (m_xtalPositionInRegion[index]);
          EEPcoeffEnd->Fill (m_recalibMap[index]) ;
          EEPcoeffMap->Fill (ee.ix(),ee.iy(),m_recalibMap[index]) ;
        }
      else
        {
          m_recalibMap[index] *= 
            m_EcalCalibBlocks.at (m_xtalRegionId[index])->at (m_xtalPositionInRegion[index]);
          EEMcoeffEnd->Fill (m_recalibMap[index]) ;
          EEMcoeffMap->Fill (ee.ix(),ee.iy(),m_recalibMap[index]) ;
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
     barrelWriter.writeLine (eb,m_recalibMap[barrelIt->rawId()]);
   }
 for (std::vector<DetId>::const_iterator endcapIt = m_endcapCells.begin ();
      endcapIt!=m_endcapCells.end ();
      ++endcapIt) 
   {
     EEDetId ee (*endcapIt);
     endcapWriter.writeLine (ee,m_recalibMap[endcapIt->rawId()]);
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
inline int EcalEleCalibLooper::etaShifter (const int etaOld) const
   {
     if (etaOld < 0) return etaOld + 85;
     else if (etaOld > 0) return etaOld + 84;
   }


//--------------------------------------------


//!find the most energetic Xtal
DetId  EcalEleCalibLooper::findMaxHit (const std::vector<DetId> & v1,
				    const EBRecHitCollection* EBhits,
				    const EERecHitCollection* EEhits) 
{
 double currEnergy = 0.;
 DetId maxHit;
 for (std::vector<DetId>::const_iterator idsIt = v1.begin () ; 
      idsIt != v1.end () ; ++idsIt)
   {
    if (idsIt->subdetId () == EcalBarrel) 
       {              
         EBRecHitCollection::const_iterator itrechit;
         itrechit = EBhits->find (*idsIt) ;
         if (itrechit == EBhits->end () )
           {
            edm::LogWarning("IML") <<"max hit not found";
            continue;
           }
         if (itrechit->energy () > currEnergy)
           {
             currEnergy = itrechit->energy () ;
             maxHit= *idsIt ;
           }
       } //barrel part ends
    else 
       {     
         EERecHitCollection::const_iterator itrechit;
         itrechit = EEhits->find (*idsIt) ;
         if (itrechit == EEhits->end () )
           {
             edm::LogWarning("IML") <<"max hit not found";
             continue;
           }
              
         if (itrechit->energy () > currEnergy)
           {
            currEnergy=itrechit->energy ();
            maxHit= *idsIt;
           }
       } //ends the barrel part
    } //end of the loop over the detId
 return maxHit;
}


//---------------------------------------------------


//!Fills the map to be sent to the IMACalibBlock
void EcalEleCalibLooper::fillEBMap (EBDetId EBmax,
                const EcalRecHitCollection * barrelHitsCollection,
                std::map<int,double> & EBxtlMap,
                int EBNumberOfRegion, double & pSubtract )
{
 int curr_eta;
 int curr_phi;
 double dummy;
 pSubtract=0;
//Maybe updated differently for eta and phi
 for (int ii = 0 ; ii < m_recoWindowSide ; ++ii)
  for (int ij = 0 ; ij < m_recoWindowSide ; ++ij)
    {
     //PG CMS official reference system
     curr_eta = EBmax.ieta () - m_recoWindowSide / 2 + ii ;
     curr_phi = EBmax.iphi () - m_recoWindowSide / 2 + ij ;
     if (abs (curr_eta) > 85) continue;
     //Skpis over the zero between EB+ and EB-
    if (curr_eta * EBmax.ieta () <= 0) 
       {
         if (EBmax.ieta () > 0) --curr_eta ; 
         else curr_eta++; 
       }
     if (curr_phi < 1) curr_phi += 360;
     if (curr_phi >= 360) curr_phi -= 360;
     if (EBDetId::validDetId (curr_eta,curr_phi))
      {
       EBDetId det = EBDetId (curr_eta,curr_phi,EBDetId::ETAPHIMODE);
       EcalRecHitCollection::const_iterator curr_recHit = barrelHitsCollection->find (det) ;
       if (curr_recHit == barrelHitsCollection->end ()) continue;
       dummy = curr_recHit->energy () ;
       if ( dummy > m_minEnergyPerCrystal && dummy < m_maxEnergyPerCrystal)
            dummy *= m_recalibMap[det.rawId ()] ;     
       else continue;
       if (m_xtalRegionId[det.rawId ()] == EBNumberOfRegion)
            EBxtlMap[m_xtalPositionInRegion[det.rawId ()]] = dummy ;
       else pSubtract += dummy;
//PG FIXME qui bisognera' inserire il ctrl per non essere troppo lontano, il bordo insomma
      }
    }
}

//---------------------------------------------------


//!Fills the map to be sent to the calibBlock for the endcap
void EcalEleCalibLooper::fillEEMap (EEDetId EEmax,
                const EcalRecHitCollection * endcapHitsCollection,
                std::map<int,double> & EExtlMap,
                int EENumberOfRegion, double & pSubtract )
  
{
 int curr_x=0;
 int curr_y=0;
 double dummy=0.;
 pSubtract=0;
 int ecalZone=EEmax.zside();
 //Loop on the energy reconstruction window
 for (int ii=0;ii<m_recoWindowSide;++ii)
 for (int ij=0;ij<m_recoWindowSide;++ij)
  {
     curr_x=EEmax.ix ()-m_recoWindowSide/2+ii;
     curr_y=EEmax.iy ()-m_recoWindowSide/2+ij;
     if (EEDetId::validDetId (curr_x,curr_y,EEmax.zside ()))
     {
      EEDetId det = EEDetId (curr_x,curr_y,ecalZone,EEDetId::XYMODE);
      EcalRecHitCollection::const_iterator curr_recHit = endcapHitsCollection->find (det) ;
      if (curr_recHit == endcapHitsCollection->end ()) continue;
      dummy = curr_recHit->energy () ;
      if ( dummy < m_minEnergyPerCrystal || dummy > m_maxEnergyPerCrystal ) continue ; 
      dummy *= m_recalibMap[det.rawId ()] ;     
       if (m_xtalRegionId[det.rawId ()] == EENumberOfRegion)
            EExtlMap[m_xtalPositionInRegion[det.rawId ()]] = dummy ;
      else pSubtract += dummy;
      //PG FIXME qui bisognera' inserire il ctrl per non essere troppo lontano, il bordo insomma
     } 
   } //PG loop on the energy reconstruction window
 } 
   


