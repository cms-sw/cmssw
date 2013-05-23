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
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CLHEP/GenericFunctions/Erf.hh"

#include <iostream>
#include <sstream>

NoiseCheck::NoiseCheck(const edm::ParameterSet& p)
{
  rootFileName = p.getParameter<std::string>("OutputFile");
  threshold_ = p.getParameter<double>("Threshold");
}
NoiseCheck::~NoiseCheck(){

  // Computing the Sigma
  TF1 * f1 = new TF1("f1","[0]*exp(-0.5*((x-[1])/[2])**2)",-1.,1.);

  for(unsigned ih=0;ih<EEDetId::kSizeForDenseIndexing;++ih)
    {
      EEDetId myDetId(EEDetId::unhashIndex(ih));
      f1->SetParameter(1,0);
      f1->SetParameter(2,0.1);
      individual_histos[ih]->getTH1F()->Fit("f1","RQ");

      float sigma=fabs(f1->GetParameter(2));
      float r=std::sqrt((myDetId.ix()-50)*(myDetId.ix()-50)+(myDetId.iy()-50)*(myDetId.iy()-50));      
      bool positiveSide= (myDetId.ix()>50.5);
      ICRatio->Fill(IC[ih]/ICMC[ih]);
      std::cout << myDetId <<" " << ih << " " << sigma << std::endl;
      if(myDetId.zside()>0)
 	{
	  EEPlus->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma);
	  ICEP->Fill(myDetId.ix()-50,myDetId.iy()-50,IC[ih]);
	  ADCP->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma/IC[ih]/adcToGeV_);
	  ADCPH->Fill(sigma/IC[ih]/adcToGeV_);
	  ADCMCPH->Fill(sigma/ICMC[ih]/adcToGeV_);
	  ICPH->Fill(IC[ih]);
	  ICMCPH->Fill(ICMC[ih]);
	  int binx=OCCPT->getTH2F()->GetXaxis()->FindBin(myDetId.ix()-50);
	  int biny=OCCPT->getTH2F()->GetXaxis()->FindBin(myDetId.iy()-50);
	  double nhits=OCCPT->getTH2F()->GetBinContent(binx,biny);
	  //	  std::cout << myDetId << " " << r << " " << nhits << std::endl;
	  if(positiveSide)
	    {
	      EEPP->Fill(r,sigma);
	      OCCPPR->Fill(r,nhits);
	    }
	  else
	    {
	      EEPN->Fill(r,sigma);
	      OCCPNR->Fill(r,nhits);
	    }
	}
      else
	{
	  EEMinus->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma);
	  ICEN->Fill(myDetId.ix()-50,myDetId.iy()-50,IC[ih]);
	  ICMCNH->Fill(ICMC[ih]);
	  ADCN->Fill(myDetId.ix()-50,myDetId.iy()-50,sigma/IC[ih]/adcToGeV_);
	  ADCNH->Fill(sigma/IC[ih]/adcToGeV_);
	  ADCMCNH->Fill(sigma/ICMC[ih]/adcToGeV_);
	  ICNH->Fill(IC[ih]);
	  int binx=OCCNT->getTH2F()->GetXaxis()->FindBin(myDetId.ix()-50);
	  int biny=OCCNT->getTH2F()->GetXaxis()->FindBin(myDetId.iy()-50);
	  double nhits=OCCNT->getTH2F()->GetBinContent(binx,biny);
	  //	  std::cout << myDetId << " " << r << " " << nhits << std::endl;
	  if(positiveSide)
	    {
	      EENP->Fill(r,sigma);
	      OCCNPR->Fill(r,nhits);
	    }
	  else
	    {
	      EENN->Fill(r,sigma);
	      OCCNNR->Fill(r,nhits);
	    }
	}      
    }
  
  std::cout << counter << " N events analyzer " << std::endl;
  std::cout << " Calling save histos " << std::endl;
  dbe->save(rootFileName);
  std::cout << " done " << std::endl;

  std::cout << " Endcap + " << std::endl;
  double nevents=HitMultiplicityP->getTH1F()->GetEntries();
  double neventspositive=HitMultiplicityP->getTH1F()->GetEntries()-HitMultiplicityP->getTH1F()->GetBinContent(1);
  double nhitsp=RecHitEPICMCZADC->getTH1F()->GetEntries();
  double nhitsperevent = nhitsp/neventspositive;
  std::cout << " N events " << nevents << " with>0 "  <<neventspositive << std::endl;
  std::cout << " N hits " << nhitsp << " N hits / event " << nhitsperevent << std::endl;
  double hotfraction = 2.*nhitsperevent/14648;
  std::cout << " Hot fraction " << hotfraction << std::endl;
  double sigma=2.7;
  double threshold=sigma*std::sqrt(2.)*TMath::ErfInverse(1.-hotfraction*2.);
  std::cout << " Threshold " << threshold << std::endl;
  Genfun::Erf myErf; 
  std::cout << " Recomputed hot fraction " << (0.5-0.5*myErf(threshold/sigma/sqrt(2.))) << std::endl;
  std::cout <<  " Old thresholds " << (0.5-0.5*myErf(5.1/sigma/sqrt(2.))) << std::endl;

  std::cout << " Endcap - " << std::endl;
  nevents=HitMultiplicityN->getTH1F()->GetEntries();
  neventspositive=HitMultiplicityN->getTH1F()->GetEntries()-HitMultiplicityN->getTH1F()->GetBinContent(1);
  nhitsp=RecHitENICMCZADC->getTH1F()->GetEntries();
  nhitsperevent = nhitsp/neventspositive;
  std::cout << " N events " << nevents << " with>0 "  <<neventspositive << std::endl;
  std::cout << " N hits " << nhitsp << " N hits / event " << nhitsperevent << std::endl;
  hotfraction = 2.*nhitsperevent/14648;
  std::cout << " Hot fraction " << hotfraction << std::endl;
  threshold=sigma*std::sqrt(2.)*TMath::ErfInverse(1.-hotfraction*2.);
  std::cout << " Threshold " << threshold << std::endl;
  std::cout << " Recomputed hot fraction " << (0.5-0.5*myErf(threshold/sigma/sqrt(2.))) << std::endl;


;}
typedef math::XYZVector XYZPoint;

void  NoiseCheck::beginRun(edm::Run const& run, edm::EventSetup const& es){

  m_firstTimeAnalyze = true ;

  dbe = edm::Service<DQMStore>().operator->();
  
  
}

void  NoiseCheck::beginJobAnalyze(const edm::EventSetup & c){

//  edm::ESHandle<CaloTopology> theCaloTopology;
//  c.get<CaloTopologyRecord>().get(theCaloTopology);       
//  edm::ESHandle<CaloGeometry> pG;
//  c.get<CaloGeometryRecord>().get(pG);     
  // Setup the tools
  //  double bField000 = 4.;
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
  OCCPPR = dbe->bookProfile("OCCPPR","EE+ profile ; Positive side ",55,0,55,100,0,1000);
  OCCPNR = dbe->bookProfile("OCCPNR","EE+ profile ; Negative side ",55,0,55,100,0,1000);
  OCCNNR = dbe->bookProfile("OCCNNR","EE- profile ; Negative side ",55,0,55,100,0,1000);
  OCCNPR = dbe->bookProfile("OCCNPR","EE- profile ; Positive side ",55,0,55,100,0,1000);
  HitMultiplicityP = dbe->book1D("HitMultiplicityP"," Multiplicity ; Positive side ", 1000,0,1000);
  HitMultiplicityN = dbe->book1D("HitMultiplicityN"," Multiplicity ; Negative side ", 1000,0,1000);
  RecHitEP = dbe->book1D("RecHitEP"," RecHit E ; Positive side ", 1100,-10,100);
  RecHitEN = dbe->book1D("RecHitEN"," RecHit E; Negative side ", 1100,-10,100);
  RecHitEPZ = dbe->book1D("RecHitEPZ"," RecHit E ; Positive side ", 300,-1,2);
  RecHitENZ = dbe->book1D("RecHitENZ"," RecHit E; Negative side ", 300,-1,2);
  RecHitEPCZ = dbe->book1D("RecHitEPCZ"," RecHit E ; Positive side ", 300,-1,2);
  RecHitENCZ = dbe->book1D("RecHitENCZ"," RecHit E; Negative side ", 300,-1,2);
  RecHitEPICMCZ = dbe->book1D("RecHitEPICMCZ"," RecHit E ; Positive side ", 300,-1,2);
  RecHitENICMCZ = dbe->book1D("RecHitENICMCZ"," RecHit E; Negative side ", 300,-1,2);
  RecHitEPICMCZADC = dbe->book1D("RecHitEPICMCZADC"," RecHit E ; Positive side ", 300,-1,9);
  RecHitENICMCZADC = dbe->book1D("RecHitENICMCZADC"," RecHit E; Negative side ", 300,-1,9);
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  c.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();
  IC = (ical->endcapItems());

  edm::ESHandle<EcalIntercalibConstantsMC> pJcal;
  c.get<EcalIntercalibConstantsMCRcd>().get(pJcal); 
  const EcalIntercalibConstantsMC* jcal = pJcal.product(); 
  ICMC = jcal->endcapItems();

  ICEP= dbe->book2D("ICEP","EE+",110,-55.5,54.5,110,-55.5,54.5);
  ICEN= dbe->book2D("ICEN","EE-",110,-55.7,54.5,110,-55.5,54.5);
  ADCP= dbe->book2D("ADCP","EE+",110,-55.5,54.5,110,-55.5,54.5);
  ADCN= dbe->book2D("ADCN","EE-",110,-55.7,54.5,110,-55.5,54.5);
  OCCP= dbe->book2D("OCCP","EE+",110,-55.5,54.5,110,-55.5,54.5);
  OCCN= dbe->book2D("OCCN","EE-",110,-55.7,54.5,110,-55.5,54.5);
  OCCPT= dbe->book2D("OCCPT","EE+",110,-55.5,54.5,110,-55.5,54.5);
  OCCNT= dbe->book2D("OCCNT","EE-",110,-55.7,54.5,110,-55.5,54.5);
  ICPH= dbe->book1D("ICPH","EE+",100,0,2.5);
  ICMCNH= dbe->book1D("ICMCNH","EE-",100,0,2.5);
  ICMCPH= dbe->book1D("ICMCPH","EE+",100,0,2.5);
  ICNH= dbe->book1D("ICNH","EE-",100,0,2.5);
  ADCPH= dbe->book1D("ADCPH","EE+",1000,0,5.);
  ADCNH= dbe->book1D("ADCNH","EE-",1000,0,5.);
  ADCMCPH= dbe->book1D("ADCMCPH","EE+",1000,0,5.);
  ADCMCNH = dbe->book1D("ADCMCNH","EE-",1000,0,5.);
  ICRatio= dbe->book1D("ICRatio","ICRatio",100,0,2.5);
}


void  NoiseCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if( m_firstTimeAnalyze )
   {
      beginJobAnalyze( iSetup ) ;
      m_firstTimeAnalyze = false ;
      counter=0;
      edm::ESHandle<EcalADCToGeVConstant> agc;
      iSetup.get<EcalADCToGeVConstantRcd>().get(agc);      
      adcToGeV_=   agc->getEEValue() ;
      std::cout << " ADC to GeV " << adcToGeV_ << std::endl;
   }
   
   // create two histos per event
   std::ostringstream oss,oss2;
   oss << "DP" << counter;
   oss2 << "DN" << counter;
   std::ostringstream oss3,oss4;
   oss3 << " Event " << counter << " + ";
   oss4 << " Event " << counter << " - ";

   MonitorElement * DisplayP = dbe->book2D(oss.str(),oss3.str(),110,-55.5,54.5,110,-55.5,54.5);
   MonitorElement * DisplayN = dbe->book2D(oss2.str(),oss4.str(),110,-55.5,54.5,110,-55.5,54.5);

   edm::Handle<EERecHitCollection> hrechit_EE_col;
   iEvent.getByLabel("ecalRecHit", "EcalRecHitsEE",hrechit_EE_col);

   EcalRecHitCollection::const_iterator rhit=hrechit_EE_col.product()->begin();
   EcalRecHitCollection::const_iterator rhitend=hrechit_EE_col.product()->end();
   unsigned hp=0;
   unsigned hn=0;
   for(;rhit!=rhitend;++rhit)
     {      
       EEDetId myDetId(rhit->id());
       individual_histos[myDetId.hashedIndex()]->Fill(rhit->energy());
       
       //       std::cout << myDetId << " " << rhit->energy() << " " << rhit->energy()/ICMC[myDetId.hashedIndex()] << std::endl;
       if(myDetId.zside()>0)
	 {
	   DisplayP->Fill(myDetId.ix()-50,myDetId.iy()-50,rhit->energy());
	   RecHitEP->Fill(rhit->energy());
	   RecHitEPZ->Fill(rhit->energy());
	   RecHitEPICMCZ->Fill(rhit->energy()/ICMC[myDetId.hashedIndex()]);
	   RecHitEPICMCZADC->Fill(rhit->energy()/ICMC[myDetId.hashedIndex()]/adcToGeV_);
	   OCCP->Fill(myDetId.ix()-50,myDetId.iy()-50);
	   if(rhit->energy()>threshold_*ICMC[myDetId.hashedIndex()])
	     {
	       OCCPT->Fill(myDetId.ix()-50,myDetId.iy()-50);
	       RecHitEPCZ->Fill(rhit->energy());
	     }
	   //	   else
	     //	     std::cout << " Cutting " << rhit->energy() << " " << ICMC[myDetId.hashedIndex()] << " " << threshold_*ICMC[myDetId.hashedIndex()] << std::endl;
	   ++hp;
	 }
       else
	 {
	   DisplayN->Fill(myDetId.ix()-50,myDetId.iy()-50,rhit->energy());
	   RecHitEN->Fill(rhit->energy());
	   RecHitENZ->Fill(rhit->energy());
	   RecHitENICMCZ->Fill(rhit->energy()/ICMC[myDetId.hashedIndex()]);
	   RecHitENICMCZADC->Fill(rhit->energy()/ICMC[myDetId.hashedIndex()]/adcToGeV_);
	   OCCN->Fill(myDetId.ix()-50,myDetId.iy()-50);
	   if(rhit->energy()>threshold_*ICMC[myDetId.hashedIndex()])
	     {
	       OCCNT->Fill(myDetId.ix()-50,myDetId.iy()-50);
	       RecHitENCZ->Fill(rhit->energy());
	     }
	   //	   else
	   //std::cout << " Cutting " << rhit->energy() << " " << ICMC[myDetId.hashedIndex()] << " " << threshold_*ICMC[myDetId.hashedIndex()] << std::endl;
	   ++hn;
	 }	       
     }
   ++counter;
   HitMultiplicityP->Fill(hp);
   HitMultiplicityN->Fill(hn);

   
   edm::Handle<reco::SuperClusterCollection> SuperClustersH;
   edm::InputTag SuperClustersTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters");
   iEvent.getByLabel(SuperClustersTag,SuperClustersH);
   
   reco::SuperClusterCollection::const_iterator scit=SuperClustersH.product()->begin();
   reco::SuperClusterCollection::const_iterator scitend=SuperClustersH.product()->end();
   for(;scit!=scitend;++scit)
     {
       if (scit->position().eta()<-1.4)
	 {
	   //	   std::cout << " Found a supercluster E = " << scit->energy();
	   reco::CaloCluster_iterator bcit=scit->clustersBegin();
	   reco::CaloCluster_iterator bcitend=scit->clustersEnd();
	   //	   std::cout << " Cells " << std::endl;
	   for(;bcit!=bcitend;++bcit)
	     {
	       //	       std::cout << " Basic cluster E " << (*bcit)->energy() << std::endl;
	       const std::vector< std::pair<DetId, float> > & hits = (*bcit)->hitsAndFractions();
	       unsigned size=hits.size();
	       for(unsigned ihit=0;ihit<size;++ihit)
		 {		   
//		   std::cout << ihit << " DetId / hash" << EEDetId(hits[ihit].first) << " " ;
//		   std::cout <<  EEDetId(hits[ihit].first).hashedIndex() << " "  << " Frac " << hits[ihit].second << " E ";
		   rhit=hrechit_EE_col.product()->begin();
		   bool found=false;
		   for(;rhit!=rhitend&&!found;++rhit)
		     {
		       if(hits[ihit].first==rhit->id())			 
			 {
			   //			   std::cout << rhit->energy() << std::endl;
			   found=true;
			 }
		     }
		 }
	     }
	 }
     }


}

void NoiseCheck::endRun()
{

}

DEFINE_FWK_MODULE(NoiseCheck);
