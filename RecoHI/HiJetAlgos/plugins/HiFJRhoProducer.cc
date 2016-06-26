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

#include "TMath.h"

#include "RecoHI/HiJetAlgos/plugins/HiFJRhoProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

using namespace edm;
using namespace pat;

//using ival = boost::icl::interval<double>;

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
  nExcl_(iConfig.getUntrackedParameter<unsigned int>("nExcl",0)),
  etaMaxExcl_(iConfig.getUntrackedParameter<double>("etaMaxExcl",2.)),
  ptMinExcl_(iConfig.getUntrackedParameter<double>("ptMinExcl",20.)),
  nExcl2_(iConfig.getUntrackedParameter<unsigned int>("nExcl2",0)),
  etaMaxExcl2_(iConfig.getUntrackedParameter<double>("etaMaxExcl2",3.)),
  ptMinExcl2_(iConfig.getUntrackedParameter<double>("ptMinExcl2",20.)),
  checkJetCand(true),
  usingPackedCand(false)
{
  jetsToken_ = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>( "jetSource" ));

  //register your products
  produces<std::vector<double > >("mapEtaEdges");
  produces<std::vector<double > >("mapToRho");
  produces<std::vector<double > >("mapToRhoM");
  produces<std::vector<double > >("ptJets");
  produces<std::vector<double > >("areaJets");
  produces<std::vector<double > >("etaJets");
  etaRanges = iConfig.getUntrackedParameter<std::vector<double> >("etaRanges");

}


HiFJRhoProducer::~HiFJRhoProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

// ------------ method called to produce the data  ------------
void HiFJRhoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // Get the vector of jets
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);

  int neta = (int)etaRanges.size();
  std::auto_ptr<std::vector<double>> mapEtaRangesOut ( new std::vector<double>(neta,-999.));

  for(int ieta = 0; ieta < neta; ieta++){
   mapEtaRangesOut->at(ieta) = etaRanges[ieta];
  }
  std::auto_ptr<std::vector<double>> mapToRhoOut ( new std::vector<double>(neta-1,1e-6));
  std::auto_ptr<std::vector<double>> mapToRhoMOut ( new std::vector<double>(neta-1,1e-6));
  
  int njets = jets->size();
  
  std::auto_ptr<std::vector<double>> ptJetsOut ( new std::vector<double>(njets,1e-6));
  std::auto_ptr<std::vector<double>> areaJetsOut ( new std::vector<double>(njets,1e-6));
  std::auto_ptr<std::vector<double>> etaJetsOut ( new std::vector<double>(njets,1e-6));
    
  static double rhoVec[999];
  static double rhomVec[999];
  static double etaVec[999];

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
    static double rhoVecCur[999] = {0.};
    static double rhomVecCur[999]= {0.};

    double etaMin = mapEtaRangesOut->at(ieta)+radius;
    double etaMax = mapEtaRangesOut->at(ieta+1)-radius;
     
    int    naccCur    = 0 ;
    double rhoCurSum  = 0.;
    double rhomCurSum = 0.;
    for(int i = 0; i<nacc; i++) {
      if(etaVec[i]>=etaMin && etaVec[i]<etaMax) {
        rhoVecCur[naccCur] = rhoVec[i];
        rhomVecCur[naccCur] = rhomVec[i];
        rhoCurSum += rhoVec[i];
        rhomCurSum += rhomVec[i];
        ++naccCur;
      }//eta selection
    }//accepted jet loop

    if(naccCur>0) {
      double rhoCur = TMath::Median(naccCur, rhoVecCur);
      double rhomCur = TMath::Median(naccCur, rhomVecCur);
      mapToRhoOut->at(ieta) = rhoCur;
      mapToRhoMOut->at(ieta) = rhomCur;
    }
  }//eta ranges
  
  iEvent.put(mapEtaRangesOut,"mapEtaEdges");
  iEvent.put(mapToRhoOut,"mapToRho");
  iEvent.put(mapToRhoMOut,"mapToRhoM");
  
  iEvent.put(ptJetsOut,"ptJets");
  iEvent.put(areaJetsOut,"areaJets");
  iEvent.put(etaJetsOut,"etaJets");
  
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

/// Function to tell us if we are using packedCandidates, only test for first candidate
bool HiFJRhoProducer::isPackedCandidate(const reco::Candidate* candidate){
  if(checkJetCand) {
    if(typeid(pat::PackedCandidate)==typeid(*candidate)) usingPackedCand = true;
    else if(typeid(reco::PFCandidate)==typeid(*candidate)) usingPackedCand = false;
    else throw cms::Exception("WrongJetCollection", "Jet constituents are not particle flow candidates");
    checkJetCand = false;
  }
  return usingPackedCand;
}



// ------------ method called once each job just before starting event loop  ------------
void 
HiFJRhoProducer::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HiFJRhoProducer::endJob() {
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiFJRhoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(HiFJRhoProducer);
