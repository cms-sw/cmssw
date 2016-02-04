// -*- C++ -*-
//
// Package:    EcalIsolatedParticleCandidateProducer
// Class:      EcalIsolatedParticleCandidateProducer
// 
/**\class EcalIsolatedParticleCandidateProducer EcalIsolatedParticleCandidateProducer.cc Calibration/EcalIsolatedParticleCandidateProducer/src/EcalIsolatedParticleCandidateProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory Safronov
//         Created:  Thu Jun  7 17:21:58 MSD 2007
// $Id: EcalIsolatedParticleCandidateProducer.cc,v 1.9 2010/01/13 22:25:52 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"


#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"



EcalIsolatedParticleCandidateProducer::EcalIsolatedParticleCandidateProducer(const edm::ParameterSet& conf)
{
  InConeSize_ = conf.getParameter<double>("EcalInnerConeSize");
  OutConeSize_= conf.getParameter<double>("EcalOuterConeSize");
  hitCountEthr_= conf.getParameter<double>("ECHitCountEnergyThreshold");
  hitEthr_=conf.getParameter<double>("ECHitEnergyThreshold");
  l1tausource_=conf.getParameter<edm::InputTag>("L1eTauJetsSource");
  hltGTseedlabel_=conf.getParameter<edm::InputTag>("L1GTSeedLabel");
  EBrecHitCollectionLabel_=conf.getParameter<edm::InputTag>("EBrecHitCollectionLabel");
  EErecHitCollectionLabel_=conf.getParameter<edm::InputTag>("EErecHitCollectionLabel");

   //register your products
  produces< reco::IsolatedPixelTrackCandidateCollection >();

}


EcalIsolatedParticleCandidateProducer::~EcalIsolatedParticleCandidateProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void 
EcalIsolatedParticleCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 
  using namespace edm;

//  std::cout<<"get tau"<<std::endl;

  Handle<l1extra::L1JetParticleCollection> l1Taus;
  iEvent.getByLabel(l1tausource_,l1Taus);

//  std::cout<<"get geom"<<std::endl;

  ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo = pG.product();

//  std::cout<<" get ec rechit"<<std::endl;

  Handle<EcalRecHitCollection> ecalEB;
  iEvent.getByLabel(EBrecHitCollectionLabel_,ecalEB);

  Handle<EcalRecHitCollection> ecalEE;
  iEvent.getByLabel(EErecHitCollectionLabel_,ecalEE);

//  std::cout<<"get l1 trig obj"<<std::endl;

  Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  iEvent.getByLabel(hltGTseedlabel_, l1trigobj);

  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;

  l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
  l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
   
  double ptTriggered=-10;
  double etaTriggered=-100;
  double phiTriggered=-100;

//  std::cout<<"find highest pT triggered obj"<<std::endl;

  for (unsigned int p=0; p<l1tauobjref.size(); p++)
        {
        if (l1tauobjref[p]->pt()>ptTriggered)
                {
                ptTriggered=l1tauobjref[p]->pt();
                phiTriggered=l1tauobjref[p]->phi();
                etaTriggered=l1tauobjref[p]->eta();
                }
        }
  for (unsigned int p=0; p<l1jetobjref.size(); p++)
        {
        if (l1jetobjref[p]->pt()>ptTriggered)
                {
                ptTriggered=l1jetobjref[p]->pt();
                phiTriggered=l1jetobjref[p]->phi();
                etaTriggered=l1jetobjref[p]->eta();
                }
        }

  reco::IsolatedPixelTrackCandidateCollection * iptcCollection=new reco::IsolatedPixelTrackCandidateCollection;

//  std::cout<<"loop over l1taus"<<std::endl;

  for (l1extra::L1JetParticleCollection::const_iterator tit=l1Taus->begin(); tit!=l1Taus->end(); tit++)
	{
	double dphi=fabs(tit->phi()-phiTriggered);
	if (dphi>3.1415926535) dphi=2*3.1415926535-dphi;
	double Rseed=sqrt(pow(etaTriggered-tit->eta(),2)+dphi*dphi);
	if (Rseed<1.2) continue;
	int nhitOut=0;
	int nhitIn=0;
	double OutEnergy=0;
	double InEnergy=0;
//	std::cout<<" loops over rechits"<<std::endl;
	for (EcalRecHitCollection::const_iterator eItr=ecalEB->begin(); eItr!=ecalEB->end(); eItr++)
		{
		double phiD, R;
                GlobalPoint pos = geo->getPosition(eItr->detid());
                double phihit = pos.phi();
                double etahit = pos.eta();
                phiD=fabs(phihit-tit->phi());
                if (phiD>3.1415926535) phiD=2*3.1415926535-phiD;
                R=sqrt(pow(etahit-tit->eta(),2)+phiD*phiD);
                
		if (R<OutConeSize_&&R>InConeSize_&&eItr->energy()>hitCountEthr_)
                	{
                  	nhitOut++;
                	}
		if (R<InConeSize_&&eItr->energy()>hitCountEthr_)
                        {
                        nhitIn++;
                        }

		if (R<OutConeSize_&&R>InConeSize_&&eItr->energy()>hitEthr_)
                        {
                        OutEnergy+=eItr->energy();
                        }
                if (R<InConeSize_&&eItr->energy()>hitEthr_)
                        {
                        InEnergy+=eItr->energy();
                        }

                }

	for (EcalRecHitCollection::const_iterator eItr=ecalEE->begin(); eItr!=ecalEE->end(); eItr++)
                {
                double phiD, R;
                GlobalPoint pos = geo->getPosition(eItr->detid());
                double phihit = pos.phi();
                double etahit = pos.eta();
                phiD=fabs(phihit-tit->phi());
                if (phiD>3.1415926535) phiD=2*3.1415926535-phiD;
                R=sqrt(pow(etahit-tit->eta(),2)+phiD*phiD);
                if (R<OutConeSize_&&R>InConeSize_&&eItr->energy()>hitCountEthr_)
                        {
                        nhitOut++;
                        }
                if (R<InConeSize_&&eItr->energy()>hitCountEthr_)
                        {
                        nhitIn++;
                        }
		if (R<OutConeSize_&&R>InConeSize_&&eItr->energy()>hitEthr_)
                        {
                        OutEnergy+=eItr->energy();
                        }
                if (R<InConeSize_&&eItr->energy()>hitEthr_)
                        {
                        InEnergy+=eItr->energy();
                        }

                }
//	std::cout<<"create and push_back candidate"<<std::endl;
	reco::IsolatedPixelTrackCandidate newca(l1extra::L1JetParticleRef(l1Taus,tit-l1Taus->begin()), InEnergy, OutEnergy, nhitIn, nhitOut);
        iptcCollection->push_back(newca);	
	}


  
  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  
//  std::cout<<"put cand into event"<<std::endl;
  std::auto_ptr<reco::IsolatedPixelTrackCandidateCollection> pOut(iptcCollection);
  iEvent.put(pOut);

}
// ------------ method called once each job just before starting event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::endJob() {
}

