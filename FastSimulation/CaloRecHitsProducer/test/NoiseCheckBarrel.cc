#include "FastSimulation/CaloRecHitsProducer/test/NoiseCheckBarrel.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include <iostream>
#include <sstream>

NoiseCheckBarrel::NoiseCheckBarrel(const edm::ParameterSet& p)
{
  rootFileName = p.getParameter<std::string>("OutputFile");
  threshold_ = p.getParameter<double>("Threshold");
}
NoiseCheckBarrel::~NoiseCheckBarrel(){

  // Computing the Sigma
  TF1 * f1 = new TF1("f1","[0]*exp(-0.5*((x-[1])/[2])**2)",-1.,1.);
  f1->SetParameter(1,0.01);
  f1->SetParameter(2,0.01);

  if(true)
    {
      for(unsigned ih=0;ih<EBDetId::kSizeForDenseIndexing;++ih)
	{
	  uint16_t flag=chanStatus_[ih];
	  //      if(flag==13)  continue;
	  individual_histos[ih]->getTH1F()->Fit("f1","RQ");
	  float sigma=fabs(f1->GetParameter(2));
	  NoiseSigma->Fill(sigma);
	  EBDetId myDetId(EBDetId::unhashIndex(ih));
	  NoiseSigmaMap->Fill(myDetId.ieta(),myDetId.iphi(),sigma);
	  if(sigma<0.03) 
	    {
	      std::cout << " Sigma " << sigma << " ih " << myDetId << std::endl;
	      LowSigmaChannelStatus->Fill(flag+0.);
	    }
	  if(sigma>0.05) 
	    {
	      std::cout << " Sigma " << sigma << " ih " <<  ih << " " << myDetId << " " << flag << std::endl;
	      HighSigmaChannelStatus->Fill(flag+0.);
	    }
	  if(flag!=0) continue;
	  NoiseSigmaClean->Fill(sigma);
	  NoiseSigmaMapClean->Fill(myDetId.ieta(),myDetId.iphi(),sigma);
	}
    }
  
  std::cout << " Calling save histos " << std::endl;
  dbe->save(rootFileName);
  std::cout << "done " << std::endl;

  // Computing the Sigma
  TF1 * f2 = new TF1("f2","[0]*exp(-0.5*((x-[1])/[2])**2)");
  NHits->getTH1F()->Fit("f2");
  float mean= fabs(f2->GetParameter(1));
  float hotfraction=mean/Ngood_;
  std::cout << " Hot fraction " << hotfraction << std::endl;

;}
typedef math::XYZVector XYZPoint;

void  NoiseCheckBarrel::beginRun(edm::Run const& run, edm::EventSetup const& es){

  m_firstTimeAnalyze = true ;

  dbe = edm::Service<DQMStore>().operator->();

  NChannelBad= dbe->book1D("NChannelBad","NChannelBad",10000,0.,10000.);
  // Retrieve the good/bad channels from the DB
  edm::ESHandle<EcalChannelStatus> pEcs;
  es.get<EcalChannelStatusRcd>().get(pEcs);
  const EcalChannelStatus* ecs = 0;
  if( pEcs.isValid() ) ecs = pEcs.product();
  unsigned nbad=0;
 Ngood_=0;
  for(unsigned ih=0;ih<EBDetId::kSizeForDenseIndexing;++ih)
    {
      chanStatus_.push_back(((ecs->barrelItems())[ih]).getStatusCode());
      if(((ecs->barrelItems())[ih]).getStatusCode()!=0) {
	++nbad;
      }
      else {
	++Ngood_;
      }
    }
  NChannelBad->Fill(nbad);
  std::cout << " N good " << Ngood_ << std::endl;
}

void  NoiseCheckBarrel::beginJobAnalyze(const edm::EventSetup & c){

//  edm::ESHandle<CaloTopology> theCaloTopology;
//  c.get<CaloTopologyRecord>().get(theCaloTopology);       
//  edm::ESHandle<CaloGeometry> pG;
//  c.get<CaloGeometryRecord>().get(pG);     
  // Setup the tools
  //  double bField000 = 4.;
//  myGeometry.setupGeometry(*pG);
//  myGeometry.setupTopology(*theCaloTopology);
//  myGeometry.initialize(bField000);

  individual_histos.resize(EBDetId::kSizeForDenseIndexing);
  for(unsigned ih=0;ih<EBDetId::kSizeForDenseIndexing;++ih)
    {
      std::ostringstream oss,oss2;
      oss << EBDetId::unhashIndex(ih);
      oss2 << "h" << ih;
      individual_histos[ih]= dbe->book1D(oss2.str(),oss.str(),100,-0.5,0.5);
    }
  
  NoiseSigma= dbe->book1D("NoiseSigma","Sigma",1000,0.,.1);
  RecHit= dbe->book1D("RecHit","Rechit",1000,0.,1.);
  NHits= dbe->book1D("NHits","NHits",10000,0.,10000.);

  NoiseSigmaMap= dbe->book2D("NoiseSigmaMap","Sigma",171,-85.5,85.5,360,0.5,360.5);
  NoiseSigmaClean= dbe->book1D("NoiseSigmaClean","Sigma",500,0.,.1);
  NoiseSigmaMapClean= dbe->book2D("NoiseSigmaMapClean","Sigma",171,-85.5,85.5,360,0.5,360.5);
  LowSigmaChannelStatus = dbe->book1D("LowSigmaChannelStatus","LowSigmaChannelStatus",20,-0.5,19.5);
  HighSigmaChannelStatus = dbe->book1D("HighSigmaChannelStatus","HighSigmaChannelStatus",20,-0.5,19.5);
  //  RecHitENICMCZ = dbe->book1D("RecHitENICMCZ"," RecHit E; Negative side ", 300,-1,2);
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  c.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();
  IC = (ical->barrelItems());

  edm::ESHandle<EcalIntercalibConstantsMC> pJcal;
  c.get<EcalIntercalibConstantsMCRcd>().get(pJcal); 
  const EcalIntercalibConstantsMC* jcal = pJcal.product(); 
  ICMC = jcal->barrelItems();

}


void  NoiseCheckBarrel::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if( m_firstTimeAnalyze )
   {
      beginJobAnalyze( iSetup ) ;
      m_firstTimeAnalyze = false ;
      counter=0;
      edm::ESHandle<EcalADCToGeVConstant> agc;
      iSetup.get<EcalADCToGeVConstantRcd>().get(agc);      
      adcToGeV_=   agc->getEBValue() ;
      std::cout << " ADC to GeV " << adcToGeV_ << std::endl;
   }
   
   // create two histos per event
   std::ostringstream oss,oss2;
   oss << "DP" << counter;
   oss2 << "DN" << counter;
   std::ostringstream oss3,oss4;
   oss3 << " Event " << counter << " + ";
   oss4 << " Event " << counter << " - ";

//   MonitorElement * DisplayP = dbe->book2D(oss.str(),oss3.str(),110,-55.5,54.5,110,-55.5,54.5);
//   MonitorElement * DisplayN = dbe->book2D(oss2.str(),oss4.str(),110,-55.5,54.5,110,-55.5,54.5);

   edm::Handle<EBRecHitCollection> hrechit_EB_col;
   iEvent.getByLabel("ecalRecHit", "EcalRecHitsEB",hrechit_EB_col);

   EcalRecHitCollection::const_iterator rhit=hrechit_EB_col.product()->begin();
   EcalRecHitCollection::const_iterator rhitend=hrechit_EB_col.product()->end();
   unsigned counter=0;
   for(;rhit!=rhitend;++rhit)
     {      
       EBDetId myDetId(rhit->id());
       unsigned ih=myDetId.hashedIndex();
       individual_histos[ih]->Fill(rhit->energy()/ICMC[ih]) ;
       uint16_t flag=chanStatus_[ih];
       if(flag!=0) continue;
       ++counter;
       RecHit->Fill(rhit->energy()/ICMC[ih]) ;
       std::cout << "AAA " << myDetId.ieta() << " " << myDetId.iphi() << " " << rhit->energy() << " " << IC[myDetId.hashedIndex()] << " " << ICMC[myDetId.hashedIndex()]  <<" " <<  rhit->energy()/ICMC[myDetId.hashedIndex()] << std::endl;
     }
   NHits->Fill(counter);
}

void NoiseCheckBarrel::endRun()
{

}

DEFINE_FWK_MODULE(NoiseCheckBarrel);
