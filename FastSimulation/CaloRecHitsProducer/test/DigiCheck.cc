#include "FastSimulation/CaloRecHitsProducer/test/DigiCheck.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <iostream>

DigiCheck::DigiCheck(const edm::ParameterSet&){;}
DigiCheck::~DigiCheck(){;}
typedef math::XYZVector XYZPoint;

void  DigiCheck::beginRun(edm::Run const& run, edm::EventSetup const& es){

  m_firstTimeAnalyze = true ;

  dbe = edm::Service<DQMStore>().operator->();
  h0b = dbe->book2D("h0b","Gain vs Gev",100,0.,1700.,5,-0.5,4.5);
  h0e = dbe->book2D("h0e","Gain vs Gev",100,0.,3000.,5,-0.5,4.5);
  h1b = dbe->book2D("h1b","Gain 3 ADC vs GeV - Barrel",140,0.,140.,1000,0,5000);
  h2b = dbe->book2D("h2b","Gain 2 ADC vs GeV - Barrel",140,0,820,1000,0,5000);
  h3b = dbe->book2D("h3b","Gain 1 ADC vs GeV - Barrel",140,0,1700,1000,0,5000);
  h1e = dbe->book2D("h1e","Gain 3 ADC vs GeV- Endcap",140,0,250,1000,0,5000);
  h2e = dbe->book2D("h2e","Gain 2 ADC vs GeV- Endcap",140,0,1400,1000,0,5000);
  h3e = dbe->book2D("h3e","Gain 1 ADC vs GeV- Endcap",140,0,3000,1000,0,5000);
  h4  = dbe->book2D("h4","HBHE GeV adc",1000,0,1000,1000,0,1000);
  h5  = dbe->book2D("h5","digis vs rechits ",400,0,200,400,0,200);
  h6  = dbe->book2D("h6","digis vs rechits ",400,0,200,400,0,200);
  h7  = dbe->book1D("h7","TP digis vs calohits ",100,-10,10);
  h8  = dbe->book2D("h8","ieta vs TP/calohits ",64,-31.5,31.5,200,-2,2);
  h9  = dbe->book2D("h9","iphi vs TP/calohits ",100,-0.5,99.5,200,-2,2);
}

void  DigiCheck::beginJobAnalyze(const edm::EventSetup & c){

  edm::ESHandle<CaloTopology> theCaloTopology;
  c.get<CaloTopologyRecord>().get(theCaloTopology);       
  edm::ESHandle<CaloGeometry> pG;
  c.get<CaloGeometryRecord>().get(pG);     
  // Setup the tools
  double bField000 = 4.;
  myGeometry.setupGeometry(*pG);
  myGeometry.setupTopology(*theCaloTopology);
  myGeometry.initialize(bField000);
    
  edm::ESHandle<EcalTrigTowerConstituentsMap> hetm;
  c.get<IdealGeometryRecord>().get(hetm);
  eTTmap_ = &(*hetm);
}

void  DigiCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if( m_firstTimeAnalyze )
   {
      beginJobAnalyze( iSetup ) ;
      m_firstTimeAnalyze = false ;
   }

  // builds the tower-cell map
  if(mapTow_sintheta.size()==0)
    {
      const std::vector<DetId>& vec(myGeometry.getEcalBarrelGeometry()->getValidDetIds(DetId::Ecal,EcalBarrel));
      std::map<EcalTrigTowerDetId,std::vector<GlobalPoint> > mapTow_Centers;
      //  std::map<EcalTrigTowerDetId,GlobalPoint> mapTow_Pos;
      unsigned size=vec.size();
      for(unsigned ic=0;ic<size;++ic)
	{
	  const CaloCellGeometry * geom=myGeometry.getEcalBarrelGeometry()->getGeometry(vec[ic]);
	  GlobalPoint p=geom->getPosition();
	  EcalTrigTowerDetId towid1= eTTmap_->towerOf(vec[ic]);
	  mapTow_Centers[towid1].push_back(p);
	}
      
      const std::vector<DetId>& vece(myGeometry.getEcalEndcapGeometry()->getValidDetIds(DetId::Ecal,EcalEndcap));
      size=vece.size();
      for(unsigned ic=0;ic<size;++ic)
	{
	  const CaloCellGeometry * geom=myGeometry.getEcalEndcapGeometry()->getGeometry(vece[ic]);
	  GlobalPoint p=geom->getPosition();
	  EcalTrigTowerDetId towid1= eTTmap_->towerOf(vece[ic]);
	  mapTow_Centers[towid1].push_back(p);
	}
  
      std::map<EcalTrigTowerDetId,std::vector<GlobalPoint> >::const_iterator ittow=mapTow_Centers.begin();
      std::map<EcalTrigTowerDetId,std::vector<GlobalPoint> >::const_iterator ittowend=mapTow_Centers.end();
      for(;ittow!=ittowend;++ittow)
	{
	  XYZPoint bar(0.,0.,0.);
	  for(unsigned itower=0;itower<ittow->second.size();++itower)
	    {
	      bar+=XYZPoint(ittow->second[itower].x(),ittow->second[itower].y(),ittow->second[itower].z());
	    }
	  bar/=ittow->second.size();
	  mapTow_sintheta[ittow->first]=sin(bar.theta());
	}
    }

  // Retrieve the SimHits.
  edm::Handle<edm::PCaloHitContainer> pcalohits;
  iEvent.getByLabel("famosSimHits","EcalHitsEB",pcalohits);
  std::vector<float> simHitsBarrel;
  simHitsBarrel.resize(62000,0.);
  edm::PCaloHitContainer::const_iterator it=pcalohits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend=pcalohits.product()->end();
  for(;it!=itend;++it)
    {
      simHitsBarrel[EBDetId(it->id()).hashedIndex()]+=it->energy();
    }
  std::vector<float> simHitsEndcap;
  simHitsEndcap.resize(20000,0.);
  iEvent.getByLabel("famosSimHits","EcalHitsEE",pcalohits);
  it=pcalohits.product()->begin();
  itend=pcalohits.product()->end();
  for(;it!=itend;++it)
    {
      simHitsEndcap[EEDetId(it->id()).hashedIndex()]+=it->energy();
    }

  edm::Handle<EBDigiCollection> ebDigis;
  iEvent.getByLabel("caloRecHits",ebDigis);
  
  edm::Handle<EEDigiCollection> eeDigis;
  iEvent.getByLabel("caloRecHits",eeDigis);

  std::map<EcalTrigTowerDetId,double> mapTow_et;
  
  //EBDigiCollection::const_iterator i;
  //  std::cout << " Barrel digis " << std::endl;
  //for(i=ebDigis->begin();i!=ebDigis->end();++i)
  for(unsigned int idigi = 0; idigi < ebDigis->size(); ++idigi)
    {
      EBDataFrame ebdf = (*ebDigis)[idigi];
      EBDetId  ebId( ebdf.id() );
      //h0b->Fill(simHitsBarrel[i->id().hashedIndex()],i->sample(0).gainId());
      h0b->Fill(simHitsBarrel[ebId.hashedIndex()],ebdf.sample(0).gainId());
      EcalTrigTowerDetId towid1= eTTmap_->towerOf(ebId);
      //      float  theta=myGeometry.getEcalBarrelGeometry()->getGeometry(i->id())->getPosition().theta();
      float et=0.;
      
      if(ebdf.sample(0).gainId()==3)
	{
	  h1b->Fill(simHitsBarrel[ebId.hashedIndex()],ebdf.sample(0).adc());
	  et = (ebdf.sample(0).adc()-200)*0.035*12.;
	  //	  et = (ebdf.sample(0).adc()-200)*0.035;
	}
      else if(ebdf.sample(0).gainId()==2)
	{
	  h2b->Fill(simHitsBarrel[ebId.hashedIndex()],ebdf.sample(0).adc());
	  et = (ebdf.sample(0).adc()-200)*0.035*2.;
	  //	  et = (ebdf.sample(0).adc()-200)*0.035;
	}
      else if(ebdf.sample(0).gainId()==1)
	{
	  h3b->Fill(simHitsBarrel[ebId.hashedIndex()],ebdf.sample(0).adc());
	  et = (ebdf.sample(0).adc()-200)*0.035*1.;
	}
      //      et*=sin(theta);
      std::map<EcalTrigTowerDetId,double>::iterator itcheck=mapTow_et.find(towid1);
      if(itcheck==mapTow_et.end())
	mapTow_et.insert(std::pair<EcalTrigTowerDetId,double>(towid1,et));
      else
	itcheck->second+=et;
	   
    }
  //EEDigiCollection::const_iterator j;
  //  std::cout << " Encap digis " << std::endl;
  //for(j=eeDigis->begin();j!=eeDigis->end();++j)
  for(unsigned int idigi = 0; idigi < eeDigis->size(); ++idigi)
    {
      EEDataFrame eedf = (*eeDigis)[idigi];
      EEDetId  eeId( eedf.id() );
      h0e->Fill(simHitsEndcap[eeId.hashedIndex()],eedf.sample(0).gainId());
      EcalTrigTowerDetId towid1= eTTmap_->towerOf(eeId);
      //      float  theta=myGeometry.getEcalEndcapGeometry()->getGeometry(eeId)->getPosition().theta();
      float et=0.;
      if(eedf.sample(0).gainId()==3)
	{
	  h1e->Fill(simHitsEndcap[eeId.hashedIndex()],eedf.sample(0).adc());
	  et = (eedf.sample(0).adc()-200)*0.060*12.;
	  //	  et = (eedf.sample(0).adc()-200)*0.060;
	}
      else if(eedf.sample(0).gainId()==2)
	{
	  h2e->Fill(simHitsEndcap[eeId.hashedIndex()],eedf.sample(0).adc());
	  et = (eedf.sample(0).adc()-200)*0.060*2.;
	  //	  et = (eedf.sample(0).adc()-200)*0.060;
	}
      else if(eedf.sample(0).gainId()==1)
	{
	  h3e->Fill(simHitsEndcap[eeId.hashedIndex()],eedf.sample(0).adc());
	  et = (eedf.sample(0).adc()-200)*0.060;
	}
      //      et*=sin(theta);
      std::map<EcalTrigTowerDetId,double>::iterator itcheck=mapTow_et.find(towid1);
      if(itcheck==mapTow_et.end())
	mapTow_et.insert(std::pair<EcalTrigTowerDetId,double>(towid1,et));
      else
	itcheck->second+=et;
    }
 
  iEvent.getByLabel("famosSimHits","HcalHits",pcalohits);
  it=pcalohits.product()->begin();
  itend=pcalohits.product()->end();
  std::map<HcalDetId,double> hcalSimHits;
  for(;it!=itend;++it)
    {
      HcalDetId myDetId(it->id());
      std::map<HcalDetId,double>::iterator itcheck;      
      itcheck=hcalSimHits.find(myDetId);
      if(itcheck==hcalSimHits.end())
	{
	  hcalSimHits.insert(std::pair<HcalDetId,double>(myDetId,it->energy()));
	}
      else
	itcheck->second+=it->energy();
    }
  
  edm::Handle<HBHEDigiCollection> hbheDigis;
  iEvent.getByLabel("caloRecHits",hbheDigis);
  HBHEDigiCollection::const_iterator k;
  for(k=hbheDigis->begin();k!=hbheDigis->end();++k)
    {
      const HcalQIESample& mySample=k->sample(0);
      double simHitEnergy=hcalSimHits[k->id()];
      h4->Fill(simHitEnergy,mySample.adc());
//      if(simHitEnergy>2.&&mySample.adc()==0)
//	std::cout << k->id() << " "  << simHitEnergy << std::endl;
    }
  edm::Handle<EBRecHitCollection> hrechit_EB_col;
  iEvent.getByLabel("caloRecHits","EcalRecHitsEB",hrechit_EB_col);

  edm::Handle<EERecHitCollection> hrechit_EE_col;
  iEvent.getByLabel("caloRecHits", "EcalRecHitsEE",hrechit_EE_col);

  std::map<EcalTrigTowerDetId,double> mapTow_et_rechits;
  
  EcalRecHitCollection::const_iterator rhit=hrechit_EB_col.product()->begin();
  EcalRecHitCollection::const_iterator rhitend=hrechit_EB_col.product()->end();
  for(;rhit!=rhitend;++rhit)
    {
      EcalTrigTowerDetId towid1= eTTmap_->towerOf(rhit->id());
      //      float  theta=myGeometry.getEcalBarrelGeometry()->getGeometry(rhit->id())->getPosition().theta();
      //      float et=rhit->energy()*sin(theta);
      float et=rhit->energy();
      std::map<EcalTrigTowerDetId,double>::iterator itcheck=mapTow_et_rechits.find(towid1);
      if(itcheck==mapTow_et_rechits.end())
	mapTow_et_rechits.insert(std::pair<EcalTrigTowerDetId,double>(towid1,et));
      else
	itcheck->second+=et;
    }
  
  rhit=hrechit_EE_col.product()->begin();
  rhitend=hrechit_EE_col.product()->end();
  for(;rhit!=rhitend;++rhit)
    {
      EcalTrigTowerDetId towid1= eTTmap_->towerOf(rhit->id());
      //      float  theta=myGeometry.getEcalEndcapGeometry()->getGeometry(rhit->id())->getPosition().theta();
      //      float et=rhit->energy()*sin(theta);
      float et=rhit->energy();
      std::map<EcalTrigTowerDetId,double>::iterator itcheck=mapTow_et_rechits.find(towid1);
      if(itcheck==mapTow_et_rechits.end())
	mapTow_et_rechits.insert(std::pair<EcalTrigTowerDetId,double>(towid1,et));
      else
	itcheck->second+=et;
    }
  
 // Get input
//  edm::Handle<EcalTrigPrimDigiCollection> tp;
//  iEvent.getByLabel("ecalTriggerPrimitiveDigis",tp);
//  EcalTrigPrimDigiCollection::const_iterator ittp=tp.product()->begin();
//  EcalTrigPrimDigiCollection::const_iterator ittpend=tp.product()->end();
//  for(;ittp!=ittpend;++ittp)
//    {
//       double Et=ittp->compressedEt();
//       Et*=0.469;
//       if (ittp->id().ietaAbs()==27 || ittp->id().ietaAbs()==28)    Et*=2;
//        // Look for the same tower in the RecHits collection 
//       std::map<EcalTrigTowerDetId,double>::const_iterator theotherTower=mapTow_et.find(ittp->id());
//       if(theotherTower==mapTow_et.end())
//	 {
//	   //	   std::cout << " Strange - in one collection but not in the other " << std::endl;	  
//	   continue;
//	 }
//       double energy2=theotherTower->second*mapTow_sintheta[ittp->id()];
//       h6->Fill(energy2,Et);
//       h7->Fill((energy2-Et)/energy2);
//       if(energy2<100.&&energy2>10.)
//	 {
//	   h8->Fill(ittp->id().ieta(),(energy2-Et)/energy2);
//	   h9->Fill(ittp->id().iphi(),(energy2-Et)/energy2);
//	 }
//       if(energy2>Et+2) 
//	 std::cout << " Large difference " << energy2 << " " << Et << ittp->id() << std::endl;
//    }


  // Finally fill the histo
  std::map<EcalTrigTowerDetId,double>::const_iterator mytowerit=mapTow_et.begin();
  std::map<EcalTrigTowerDetId,double>::const_iterator mytoweritend=mapTow_et.end();
  for(;mytowerit!=mytoweritend;++mytowerit)
    {
      // Look for the same tower in the RecHits collection 
      std::map<EcalTrigTowerDetId,double>::const_iterator theotherTower=mapTow_et_rechits.find(mytowerit->first);
      if(theotherTower==mapTow_et_rechits.end())
	{
	  std::cout << " Strange - in one collection but not in the other " << std::endl;	  
	  continue;
	}
      double energy1=mytowerit->second*mapTow_sintheta[mytowerit->first];
      double energy2=theotherTower->second*mapTow_sintheta[mytowerit->first];
      h5->Fill(energy1,energy2);
    }
}

void DigiCheck::endRun()
{
  dbe->save("Digicheck.root");
}


DEFINE_FWK_MODULE(DigiCheck);
