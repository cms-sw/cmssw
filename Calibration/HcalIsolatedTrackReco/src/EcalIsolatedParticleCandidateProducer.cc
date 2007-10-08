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
// $Id: EcalIsolatedParticleCandidateProducer.cc,v 1.2 2007/10/08 13:22:25 safronov Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidateFwd.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/Framework/interface/produce_helpers.h"


EcalIsolatedParticleCandidateProducer::EcalIsolatedParticleCandidateProducer(const edm::ParameterSet& conf)
{
  InConeSize_ = conf.getParameter<double>("EcalInnerConeSize");
  OutConeSize_= conf.getParameter<double>("EcalOuterConeSize");
  hitCountEthr_= conf.getParameter<double>("ECHitCountEnergyThreshold");
  hitEthr_=conf.getParameter<double>("ECHitEnergyThreshold");
  l1tausource_=conf.getUntrackedParameter<edm::InputTag>("L1eTauJetsSource");
  hltGTseedlabel_=conf.getUntrackedParameter<edm::InputTag>("L1GTSeedLabel");
  EBrecHitCollectionLabel_=conf.getUntrackedParameter<edm::InputTag>("EBrecHitCollectionLabel");
  EErecHitCollectionLabel_=conf.getUntrackedParameter<edm::InputTag>("EErecHitCollectionLabel");

   //register your products
   produces< reco::EcalIsolatedParticleCandidateCollection >();

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

  Handle<l1extra::L1JetParticleCollection> l1Taus;
  iEvent.getByLabel(l1tausource_,l1Taus);

  ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  geo = pG.product();

  Handle<EcalRecHitCollection> ecalEB;
  iEvent.getByLabel(EBrecHitCollectionLabel_,ecalEB);

  Handle<EcalRecHitCollection> ecalEE;
  iEvent.getByLabel(EErecHitCollectionLabel_,ecalEE);

  Handle<reco::HLTFilterObjectWithRefs> l1trigobj;
  iEvent.getByLabel(hltGTseedlabel_, l1trigobj);
   
  double ptTriggered=-10;
  double etaTriggered=-100;
  double phiTriggered=-100;

  for (unsigned int p=0; p<l1trigobj->size(); p++)
	{
	const RefToBase<reco::Candidate> l1objref=l1trigobj->getParticleRef(p);
	if (l1objref.get()->pt()>ptTriggered)
		{
		ptTriggered=l1objref.get()->pt(); 
		phiTriggered=l1objref.get()->phi();
		etaTriggered=l1objref.get()->eta();
		}
	}  

  reco::EcalIsolatedParticleCandidateCollection * eipcCollection=new reco::EcalIsolatedParticleCandidateCollection;

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
	reco::EcalIsolatedParticleCandidate newca(tit->eta(), tit->phi(), InEnergy, OutEnergy, nhitIn, nhitOut);
        eipcCollection->push_back(newca);	
	}


  
  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  
  std::auto_ptr<reco::EcalIsolatedParticleCandidateCollection> pOut(eipcCollection);
  iEvent.put(pOut);

}
// ------------ method called once each job just before starting event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalIsolatedParticleCandidateProducer::endJob() {
}

