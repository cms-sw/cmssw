// -*- C++ -*-
//
// Package:    HiJetBackground/HiFJRhoProducer
// Class:      HiFJRhoProducer
// 
/**\class HiFJRhoProducer HiFJRhoProducer.cc HiJetBackground/HiFJRhoProducer/plugins/HiFJRhoProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marta Verweij
//         Created:  Thu, 16 Jul 2015 10:57:12 GMT
//
//

#include <memory>
#include <string>

#include "RecoHI/HiJetAlgos/plugins/HiFJRhoProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

//using namespace edm;
using namespace reco;
//using namespace pat;

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HiFJRhoProducer::HiFJRhoProducer(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("jetSource")),
  nExcl_(iConfig.getParameter<int>("nExcl")),
  etaMaxExcl_(iConfig.getParameter<double>("etaMaxExcl")),
  ptMinExcl_(iConfig.getParameter<double>("ptMinExcl")),
  nExcl2_(iConfig.getParameter<int>("nExcl2")),
  etaMaxExcl2_(iConfig.getParameter<double>("etaMaxExcl2")),
  ptMinExcl2_(iConfig.getParameter<double>("ptMinExcl2")),
  checkJetCand(true),
  usingPackedCand(false)
{
  jetsToken_ = consumes<edm::View<reco::Jet> >(src_);

  //register your products
  produces<std::vector<double > >("mapEtaEdges");
  produces<std::vector<double > >("mapToRho");
  produces<std::vector<double > >("mapToRhoM");
  produces<std::vector<double > >("ptJets");
  produces<std::vector<double > >("areaJets");
  produces<std::vector<double > >("etaJets");
  etaRanges = iConfig.getParameter<std::vector<double> >("etaRanges");

}


HiFJRhoProducer::~HiFJRhoProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to produce the data  ------------
void HiFJRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  // Get the vector of jets
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);

  int neta = (int)etaRanges.size();
  auto mapEtaRangesOut = std::make_unique<std::vector<double>>(neta,-999.);

  for(int ieta = 0; ieta < neta; ieta++){
   mapEtaRangesOut->at(ieta) = etaRanges[ieta];
  }
  auto mapToRhoOut = std::make_unique<std::vector<double>>(neta-1,1e-6);
  auto mapToRhoMOut = std::make_unique<std::vector<double>>(neta-1,1e-6);
  
  int njets = jets->size();
  
  auto ptJetsOut = std::make_unique<std::vector<double>>(njets,1e-6);
  auto areaJetsOut = std::make_unique<std::vector<double>>(njets,1e-6);
  auto etaJetsOut = std::make_unique<std::vector<double>>(njets,1e-6);
    
  double rhoVec[999];
  double rhomVec[999];
  double etaVec[999];

  // int neta = (int)mapEtaRangesOut->size();
  int nacc = 0;
  unsigned int njetsEx = 0;
  unsigned int njetsEx2 = 0;
  for(auto jet = jets->begin(); jet != jets->end(); ++jet) {
    if(njetsEx<nExcl_ && fabs(jet->eta())<etaMaxExcl_ && jet->pt()>ptMinExcl_) {
      njetsEx++;
      continue;
    }
    if(njetsEx2<nExcl2_ && fabs(jet->eta())<etaMaxExcl2_ && fabs(jet->eta())>etaMaxExcl_ && jet->pt()>ptMinExcl2_) {
      njetsEx2++;
      continue;
    }
    float pt = jet->pt();
    float area = jet->jetArea();
    float eta = jet->eta();
	
    if(eta<mapEtaRangesOut->at(0) || eta>mapEtaRangesOut->at(neta-1)) continue;
    if(area>0.) {
      rhoVec[nacc] = pt/area;
      rhomVec[nacc] = calcMd(&*jet)/area;
      etaVec[nacc] = eta;
      ptJetsOut->at(nacc) = pt;
      areaJetsOut->at(nacc) = area;
      etaJetsOut->at(nacc) = eta;
      ++nacc;
    }
  }

  ptJetsOut->resize(nacc);
  areaJetsOut->resize(nacc);
  etaJetsOut->resize(nacc);
  //calculate rho and rhom in eta ranges
  double radius = 0.2; //distance kt clusters needs to be from edge
  for(int ieta = 0; ieta<(neta-1); ieta++) {
    std::vector<double> rhoVecCur;
    std::vector<double> rhomVecCur;
    
    double etaMin = mapEtaRangesOut->at(ieta)+radius;
    double etaMax = mapEtaRangesOut->at(ieta+1)-radius;
     
    int    naccCur    = 0 ;
    double rhoCurSum  = 0.;
    double rhomCurSum = 0.;
    for(int i = 0; i<nacc; i++) {
      if(etaVec[i]>=etaMin && etaVec[i]<etaMax) {
        rhoVecCur.push_back(rhoVec[i]);
        rhomVecCur.push_back(rhomVec[i]);
        
        rhoCurSum += rhoVec[i];
        rhomCurSum += rhomVec[i];
        ++naccCur;
      }//eta selection
    }//accepted jet loop

    if(naccCur>0) {
      double rhoCur = calcMedian(rhoVecCur);
      double rhomCur = calcMedian(rhomVecCur);
      mapToRhoOut->at(ieta) = rhoCur;
      mapToRhoMOut->at(ieta) = rhomCur;
    }
  }//eta ranges
  
  iEvent.put(std::move(mapEtaRangesOut),"mapEtaEdges");
  iEvent.put(std::move(mapToRhoOut),"mapToRho");
  iEvent.put(std::move(mapToRhoMOut),"mapToRhoM");
  
  iEvent.put(std::move(ptJetsOut),"ptJets");
  iEvent.put(std::move(areaJetsOut),"areaJets");
  iEvent.put(std::move(etaJetsOut),"etaJets");

  return;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
HiFJRhoProducer::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
HiFJRhoProducer::endStream() {
}

double HiFJRhoProducer::calcMd(const reco::Jet *jet) {
  //
  //get md as defined in http://arxiv.org/pdf/1211.2811.pdf
  //

  //Loop over the jet constituents
  double sum = 0.;
  for(auto daughter : jet->getJetConstituentsQuick()){
    if(isPackedCandidate(daughter)){     //packed candidate situation
      auto part = static_cast<const pat::PackedCandidate*>(daughter);
      sum += sqrt(part->mass()*part->mass() + part->pt()*part->pt()) - part->pt();
    } else {
      auto part = static_cast<const reco::PFCandidate*>(daughter);
      sum += sqrt(part->mass()*part->mass() + part->pt()*part->pt()) - part->pt();
    }
  }

  return sum;
}

bool HiFJRhoProducer::isPackedCandidate(const reco::Candidate* candidate){
  if(checkJetCand) {
    if(typeid(pat::PackedCandidate)==typeid(*candidate)) usingPackedCand = true;
    else if(typeid(reco::PFCandidate)==typeid(*candidate)) usingPackedCand = false;
    else throw cms::Exception("WrongJetCollection", "Jet constituents are not particle flow candidates");
    checkJetCand = false;
  }
  return usingPackedCand;
}

void HiFJRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetSource",edm::InputTag("kt4PFJets"));
  desc.add<int>("nExcl", 2);
  desc.add<double>("etaMaxExcl",2.);
  desc.add<double>("ptMinExcl",20.);
  desc.add<int>("nExcl2", 2);
  desc.add<double>("etaMaxExcl2",2.);
  desc.add<double>("ptMinExcl2",20.);
  desc.add<std::vector<double> >("etaRanges",{});
  descriptions.add("hiFJRhoProducer",desc);
}


//--------- method to calculate median ------------------
double HiFJRhoProducer::calcMedian(std::vector<double> &v)
{
  //post-condition: After returning, the elements in v may be reordered and the resulting order is implementation defined.
  //works for even and odd collections
  if(v.empty()) {
    return 0.0;
  }
  auto n = v.size() / 2;
  std::nth_element(v.begin(), v.begin()+n, v.end());
  auto med = v[n];
  if(!(v.size() & 1)) { //If the set size is even
    auto max_it = std::max_element(v.begin(), v.begin()+n);
    med = (*max_it + med) / 2.0;
  }
  return med;    
}


//define this as a plug-in
DEFINE_FWK_MODULE(HiFJRhoProducer);
