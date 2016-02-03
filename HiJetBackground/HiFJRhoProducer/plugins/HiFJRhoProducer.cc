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

#include "TLorentzVector.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"

#include "HiJetBackground/HiFJRhoProducer/plugins/HiFJRhoProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "CommonTools/Utils/interface/PtComparator.h"

//#include <boost/icl/interval_map.hpp>

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
  checkJetCand(true),
  usingPackedCand(false)
{
  jetsToken_ = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>( "jetSource" ));

  //register your products
  // produces<double>("rho");  //pt-density
  // produces<double>("rhom"); //m-density

  //  std::string alias;
  //produces<boost::icl::interval_map<double, unsigned int> >("mapToIndex");//.setBranchAlias( alias );
  
  //produces<std::map<unsigned int, double > >("mapEtaEdges");//.setBranchAlias("mapEtaEdges");
  // produces<std::map<unsigned int, double > >("mapToRho").setBranchAlias( alias );
  // produces<std::map<unsigned int, double > >("mapToRhoM").setBranchAlias( alias );
  produces<std::vector<double > >("mapEtaEdges");//.setBranchAlias("mapEtaEdges");
  produces<std::vector<double > >("mapToRho");//.setBranchAlias( alias );
  produces<std::vector<double > >("mapToRhoM");//.setBranchAlias( alias );
  

  // //make settable from config later
  // mapEtaRanges_[1] = -5.;
  // mapEtaRanges_[2] = -3.;
  // mapEtaRanges_[3] = -2.1;
  // mapEtaRanges_[4] = -1.3;
  // mapEtaRanges_[5] =  1.3;
  // mapEtaRanges_[6] =  2.1;
  // mapEtaRanges_[7] =  3.;
  // mapEtaRanges_[8] =  5.;
  
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

  // std::auto_ptr<std::map<int,double>> mapEtaRangesOut ( new std::map<int,double>);
  std::auto_ptr<std::vector<double>> mapEtaRangesOut ( new std::vector<double>(8,-999.));
  //make settable from config later
  mapEtaRangesOut->at(0) = -5.;
  mapEtaRangesOut->at(1) = -3.;
  mapEtaRangesOut->at(2) = -2.1;
  mapEtaRangesOut->at(3) = -1.3;
  mapEtaRangesOut->at(4) =  1.3;
  mapEtaRangesOut->at(5) =  2.1;
  mapEtaRangesOut->at(6) =  3.;
  mapEtaRangesOut->at(7) =  5.;

  std::auto_ptr<std::vector<double>> mapToRhoOut ( new std::vector<double>(7,1e-6));
  std::auto_ptr<std::vector<double>> mapToRhoMOut ( new std::vector<double>(7,1e-6));
  
  //Printf("nExcl: %d",nExcl_);
  static double rhoVec[999];
  static double rhomVec[999];
  static double etaVec[999];
  //int neta = (int)mapEtaRanges_.size();
  int neta = (int)mapEtaRangesOut->size();
  int nacc = 0;
  for(auto jet = jets->begin() + nExcl_; jet != jets->end(); ++jet) {
    float pt = jet->pt();
    float area = jet->jetArea();
    float eta = jet->eta();
    //Printf("minEta: %f maxEta: %f",mapEtaRanges_.at(1), mapEtaRanges_.at(neta));
    //if(eta<mapEtaRanges_.at(1) || eta>mapEtaRanges_.at(neta)) continue;
    if(eta<mapEtaRangesOut->at(0) || eta>mapEtaRangesOut->at(neta-1)) continue;
    if(area>0.) {
      //Printf("pt: %f area: %f  eta: %f",pt,area,eta);
      rhoVec[nacc] = pt/area;
      rhomVec[nacc] = calcMd(&*jet)/area;
      etaVec[nacc] = eta;
      ++nacc;
    }
  }

  //calculate rho and rhom in eta ranges
  double radius = 0.2;
  //Printf("neta: %d",neta);
  for(int ieta = 0; ieta<(neta-1); ieta++) {
    static double rhoVecCur[999] = {0.};
    static double rhomVecCur[999]= {0.};

    //     double etaMin = mapEtaRanges_.at(ieta)+radius;
    // double etaMax = mapEtaRanges_.at(ieta+1)-radius;
    double etaMin = mapEtaRangesOut->at(ieta)+radius;
    double etaMax = mapEtaRangesOut->at(ieta+1)-radius;
    //Printf("ieta: %d minEta: %f maxEta: %f",ieta,etaMin,etaMax);
     
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
       // mapToRho_[ieta] = rhoCur;
       // mapToRhoM_[ieta] = rhomCur;
       mapToRhoOut->at(ieta) = rhoCur;
       mapToRhoMOut->at(ieta) = rhomCur;
       //Printf("HiFJRhoProducer ieta: %d  rho: %f  rhom: %f",ieta,rhoCur,rhomCur);
       //Printf("HiFJRhoProducer ieta: %d  rho: %f  rhom: %f",ieta,mapToRhoOut->at(ieta),mapToRhoMOut->at(ieta));
     }
  }//eta ranges
  
  iEvent.put(mapEtaRangesOut,"mapEtaEdges");
  iEvent.put(mapToRhoOut,"mapToRho");
  iEvent.put(mapToRhoMOut,"mapToRhoM");
}

double HiFJRhoProducer::calcMd(const reco::Jet *jet) {
  //
  //get md as defined in http://arxiv.org/pdf/1211.2811.pdf
  //

  //Loop over the jet constituents
  double sum = 0.;
  for(auto daughter : jet->getJetConstituentsQuick()){
    //double m = -1.; double pt = -1.; double eta = -999.;
    if(isPackedCandidate(daughter)){     //packed candidate situation
      auto part = static_cast<const pat::PackedCandidate*>(daughter);
      sum += TMath::Sqrt(part->mass()*part->mass() + part->pt()*part->pt()) - part->pt();
      //m = part->mass(); pt = part->pt(); eta = part->eta();
    } else {
      auto part = static_cast<const reco::PFCandidate*>(daughter);
      sum += TMath::Sqrt(part->mass()*part->mass() + part->pt()*part->pt()) - part->pt();
      //m = part->mass(); pt = part->pt(); eta = part->eta();
    }
    // Printf("constituent m = %f  pt = %f  eta: %f",m,pt,eta);
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
