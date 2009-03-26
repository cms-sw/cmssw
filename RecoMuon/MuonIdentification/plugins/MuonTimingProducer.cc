//
// Package:    MuonTimingProducer
// Class:      MuonTimingProducer
// 
/**\class MuonTimingProducer MuonTimingProducer.cc RecoMuon/MuonIdentification/src/MuonTimingProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
// $Id: MuonTimingProducer.cc,v 1.1 2009/03/26 23:23:10 ptraczyk Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h" 
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "RecoMuon/MuonIdentification/plugins/MuonTimingProducer.h"
#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"


//
// constructors and destructor
//
MuonTimingProducer::MuonTimingProducer(const edm::ParameterSet& iConfig)
{
   produces<reco::MuonTimeExtraMap>();

   m_muonCollection = iConfig.getParameter<edm::InputTag>("MuonCollection");

   // Load parameters for the TimingExtractor
   edm::ParameterSet dtTimingParameters = iConfig.getParameter<edm::ParameterSet>("DTTimingParameters");
   theDTTimingExtractor_ = new DTTimingExtractor(dtTimingParameters);
}


MuonTimingProducer::~MuonTimingProducer()
{
   if (theDTTimingExtractor_) delete theDTTimingExtractor_;
}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
MuonTimingProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonTimingProducer::endJob() {
}

// ------------ method called to produce the data  ------------
void
MuonTimingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::auto_ptr<reco::MuonTimeExtraMap> muonTimeMap(new reco::MuonTimeExtraMap());
  reco::MuonTimeExtraMap::Filler filler(*muonTimeMap);
  
  edm::Handle<reco::MuonCollection> muons; 
  iEvent.getByLabel(m_muonCollection, muons);

  unsigned int nMuons = muons->size();
  if (!nMuons) return;
  
  vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
  vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
  vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);

  for ( unsigned int i=0; i<nMuons; ++i ) {

    reco::MuonTimeExtra dtTime;
    reco::MuonTimeExtra cscTime;
    reco::MuonTimeExtra combinedTime;

    reco::MuonRef muonr(muons,i);
    
    fillTiming(muonr, dtTime, cscTime, combinedTime, iEvent, iSetup);
    
    dtTimeColl[i] = dtTime;
    cscTimeColl[i] = cscTime;
    combinedTimeColl[i] = combinedTime;
     
  }
  
  filler.insert(muons, combinedTimeColl.begin(), combinedTimeColl.end());
  
  filler.fill();
  
  iEvent.put(muonTimeMap);

}


void 
MuonTimingProducer::fillTiming( reco::MuonRef muon, reco::MuonTimeExtra& dtTime, reco::MuonTimeExtra& cscTime, reco::MuonTimeExtra& combinedTime, edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  TimeMeasurementSequence dtTmSeq;
     
  if ( !(muon->standAloneMuon().isNull()) ) {
    theDTTimingExtractor_->fillTiming(dtTmSeq, muon->standAloneMuon(), iEvent, iSetup);
  }
     
  // Fill DT-specific timing information block     
  if (dtTmSeq.totalWeight)
    fillTimeFromMeasurements(dtTmSeq, dtTime);
       
  // TODO - combine the TimeMeasurementSequences from all subdetectors
  TimeMeasurementSequence combinedTmSeq;
          
  // Fill the master timing block
  // TEMPORARY! use DT only for now
  // in the future there will be a CSCTimingExtractor
  // and in the reco::Muon we will have ECAL and HCAL time stored
  if (dtTime.nDof())
    fillTimeFromMeasurements(dtTmSeq, combinedTime);
     
  LogTrace("MuonTime") << "Global 1/beta: " << combinedTime.inverseBeta() << " +/- " << combinedTime.inverseBetaErr()<<std::endl;
  LogTrace("MuonTime") << "  Free 1/beta: " << combinedTime.freeInverseBeta() << " +/- " << combinedTime.freeInverseBetaErr()<<std::endl;
  LogTrace("MuonTime") << "  Vertex time (in-out): " << combinedTime.timeAtIpInOut() << " +/- " << combinedTime.timeAtIpInOutErr()
                       << "  # of points: " << combinedTime.nDof() <<std::endl;
  LogTrace("MuonTime") << "  Vertex time (out-in): " << combinedTime.timeAtIpOutIn() << " +/- " << combinedTime.timeAtIpOutInErr()<<std::endl;
  LogTrace("MuonTime") << "  direction: "   << combinedTime.direction() << std::endl;
     
}


void 
MuonTimingProducer::fillTimeFromMeasurements( TimeMeasurementSequence tmSeq, reco::MuonTimeExtra &muTime ) {

  vector <double> x,y;
  double invbeta=0, invbetaerr=0;
  double vertexTime=0, vertexTimeErr=0, vertexTimeR=0, vertexTimeRErr=0;    
  double freeBeta, freeBetaErr, freeTime, freeTimeErr;

  for (unsigned int i=0;i<tmSeq.dstnc.size();i++) {
    invbeta+=(1.+tmSeq.local_t0.at(i)/tmSeq.dstnc.at(i)*30.)*tmSeq.weight.at(i)/tmSeq.totalWeight;
    x.push_back(tmSeq.dstnc.at(i)/30.);
    y.push_back(tmSeq.local_t0.at(i)+tmSeq.dstnc.at(i)/30.);
    vertexTime+=tmSeq.local_t0.at(i)*tmSeq.weight.at(i)/tmSeq.totalWeight;
    vertexTimeR+=(tmSeq.local_t0.at(i)+2*tmSeq.dstnc.at(i)/30.)*tmSeq.weight.at(i)/tmSeq.totalWeight;
  }

  double diff;
  for (unsigned int i=0;i<tmSeq.dstnc.size();i++) {
    diff=(1.+tmSeq.local_t0.at(i)/tmSeq.dstnc.at(i)*30.)-invbeta;
    invbetaerr+=diff*diff*tmSeq.weight.at(i);
    diff=tmSeq.local_t0.at(i)-vertexTime;
    vertexTimeErr+=diff*diff*tmSeq.weight.at(i);
    diff=tmSeq.local_t0.at(i)+2*tmSeq.dstnc.at(i)/30.-vertexTimeR;
    vertexTimeRErr+=diff*diff*tmSeq.weight.at(i);
  }
  
  invbetaerr=sqrt(invbetaerr/tmSeq.totalWeight);
  vertexTimeErr=sqrt(vertexTimeErr/tmSeq.totalWeight);
  vertexTimeRErr=sqrt(vertexTimeRErr/tmSeq.totalWeight);

  muTime.setInverseBeta(invbeta);
  muTime.setInverseBetaErr(invbetaerr);
  muTime.setTimeAtIpInOut(vertexTime);
  muTime.setTimeAtIpInOutErr(vertexTimeErr);
  muTime.setTimeAtIpOutIn(vertexTimeR);
  muTime.setTimeAtIpOutInErr(vertexTimeRErr);
      
  rawFit(freeBeta, freeBetaErr, freeTime, freeTimeErr, x, y);

  muTime.setFreeInverseBeta(freeBeta);
  muTime.setFreeInverseBetaErr(freeBetaErr);
    
  muTime.setNDof((int)tmSeq.totalWeight);

}



void 
MuonTimingProducer::rawFit(double &a, double &da, double &b, double &db, const vector<double> hitsx, const vector<double> hitsy) {

  double s=0,sx=0,sy=0,x,y;
  double sxx=0,sxy=0;

  a=b=0;
  if (hitsx.size()==0) return;
  if (hitsx.size()==1) {
    b=hitsy[0];
  } else {
    for (unsigned int i = 0; i != hitsx.size(); i++) {
      x=hitsx[i];
      y=hitsy[i];
      sy += y;
      sxy+= x*y;
      s += 1.;
      sx += x;
      sxx += x*x;
    }

    double d = s*sxx - sx*sx;
    b = (sxx*sy- sx*sxy)/ d;
    a = (s*sxy - sx*sy) / d;
    da = sqrt(sxx/d);
    db = sqrt(s/d);
  }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MuonTimingProducer);
