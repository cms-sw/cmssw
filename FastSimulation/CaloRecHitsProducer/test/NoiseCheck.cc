#include "FastSimulation/CaloRecHitsProducer/test/NoiseCheck.h"
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
#include <iostream>
#include <sstream>

NoiseCheck::NoiseCheck(const edm::ParameterSet&){;}
NoiseCheck::~NoiseCheck(){

  // Computing the Sigma
  TF1 * f1 = new TF1("f1","[0]*exp(-0.5*((x-[1])/[2])**2)",-1.,1.);
  f1->SetParameter(1,0);
  f1->SetParameter(2,0.1);

  for(unsigned ih=0;ih<EEDetId::kSizeForDenseIndexing;++ih)
    {
      EEDetId myDetId(EEDetId::unhashIndex(ih));
      individual_histos[ih]->getTH1F()->Fit("f1","RQ");
      float sigma=fabs(f1->GetParameter(2));
      float r=std::sqrt((myDetId.ix()-50)*(myDetId.ix()-50)+(myDetId.iy()-50)*(myDetId.iy()-50));      
      bool positiveSide= (myDetId.ix()>50.5);
      if(myDetId.zside()>0)
	{
	  EEPlus->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma);
	  if(positiveSide)
	    EEPP->Fill(r,sigma);
	  else
	    EEPN->Fill(r,sigma);
	}
      else
	{
	  EEMinus->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma);
	  if(positiveSide)
	    EENP->Fill(r,sigma);
	  else
	    EENN->Fill(r,sigma);
	}      
    }
  

  std::cout << " Calling save histos " << std::endl;
  dbe->save("Noisecheck.root");
  std::cout << "done " << std::endl;
;}
typedef math::XYZVector XYZPoint;

void  NoiseCheck::beginRun(edm::Run const& run, edm::EventSetup const& es){

  m_firstTimeAnalyze = true ;

  dbe = edm::Service<DQMStore>().operator->();
  
  
}

void  NoiseCheck::beginJobAnalyze(const edm::EventSetup & c){

  edm::ESHandle<CaloTopology> theCaloTopology;
  c.get<CaloTopologyRecord>().get(theCaloTopology);       
  edm::ESHandle<CaloGeometry> pG;
  c.get<CaloGeometryRecord>().get(pG);     
  // Setup the tools
  double bField000 = 4.;
//  myGeometry.setupGeometry(*pG);
//  myGeometry.setupTopology(*theCaloTopology);
//  myGeometry.initialize(bField000);

  individual_histos.resize(EEDetId::kSizeForDenseIndexing);
  for(unsigned ih=0;ih<EEDetId::kSizeForDenseIndexing;++ih)
    {
      std::ostringstream oss,oss2;
      oss << EEDetId::unhashIndex(ih);
      oss2 << "h" << ih;
      individual_histos[ih]= dbe->book1D(oss2.str(),oss.str(),100,-1.,1.);
    }

  EEPlus= dbe->book2D("EEPlus","EE+",110,-55.5,54.5,110,-55.5,54.5);
  EEMinus= dbe->book2D("EEMinus","EE-",110,-55.7,54.5,110,-55.5,54.5);
  EEPP = dbe->bookProfile("EEPP","EE+ profile ; Positive side ",55,0,55,100,0,0.5);
  EEPN = dbe->bookProfile("EEPN","EE+ profile ; Negative side ",55,0,55,100,0,0.5);
  EENN = dbe->bookProfile("EENN","EE- profile ; Negative side ",55,0,55,100,0,0.5);
  EENP = dbe->bookProfile("EENP","EE- profile ; Positive side ",55,0,55,100,0,0.5);
}

void  NoiseCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if( m_firstTimeAnalyze )
   {
      beginJobAnalyze( iSetup ) ;
      m_firstTimeAnalyze = false ;
   }

  edm::Handle<EERecHitCollection> hrechit_EE_col;
  iEvent.getByLabel("ecalRecHit", "EcalRecHitsEE",hrechit_EE_col);

  std::map<EcalTrigTowerDetId,double> mapTow_et_rechits;
  
  EcalRecHitCollection::const_iterator rhit=hrechit_EE_col.product()->begin();
  EcalRecHitCollection::const_iterator rhitend=hrechit_EE_col.product()->end();
  for(;rhit!=rhitend;++rhit)
    {      
      EEDetId myDetId(rhit->id());
      individual_histos[myDetId.hashedIndex()]->Fill(rhit->energy());
    }
  }

void NoiseCheck::endRun()
{

}

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(NoiseCheck);
