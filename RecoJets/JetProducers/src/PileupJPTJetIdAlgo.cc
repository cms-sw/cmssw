#include<fstream>
#include<iomanip>
#include<iostream>

// DataFormat //

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"

#include "DataFormats/JetReco/interface/JetID.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"


// FWCore //

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// TrackingTools //
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

// Constants, Math //
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "CommonTools/Utils/interface/TMVAZipReader.h"

#include <vector>

///////////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/interface/PileupJPTJetIdAlgo.h"

////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
namespace cms
{
  
  PileupJPTJetIdAlgo::PileupJPTJetIdAlgo(const edm::ParameterSet& iConfig)
  {
    
    verbosity   = iConfig.getParameter<int>("Verbosity");
    tmvaWeights_         = edm::FileInPath(iConfig.getParameter<std::string>("tmvaWeightsCentral")).fullPath();
    tmvaWeightsF_         = edm::FileInPath(iConfig.getParameter<std::string>("tmvaWeightsForward")).fullPath();
    tmvaMethod_          = iConfig.getParameter<std::string>("tmvaMethod");

  }
  
  PileupJPTJetIdAlgo::~PileupJPTJetIdAlgo()
  {
  //  std::cout<<" JetPlusTrack destructor "<<std::endl;
  }
  
  void PileupJPTJetIdAlgo::bookMVAReader()
  {
    
// Read TMVA tree
//    std::string mvaw     = "RecoJets/JetProducers/data/TMVAClassification_BDTG.weights.xml";       
//    std::string mvawF     = "RecoJets/JetProducers/data/TMVAClassification_BDTG.weights_F.xml";
//    tmvaWeights_         = edm::FileInPath(mvaw).fullPath();
//    tmvaWeightsF_         = edm::FileInPath(mvawF).fullPath();
//    tmvaMethod_          = "BDTG method";

    if(verbosity>0) std::cout<<" TMVA method "<<tmvaMethod_.c_str()<<" "<<tmvaWeights_.c_str()<<std::endl;
    reader_ = new TMVA::Reader( "!Color:!Silent" );

    reader_->AddVariable( "Nvtx", &Nvtx );
    reader_->AddVariable( "PtJ", &PtJ );
    reader_->AddVariable( "EtaJ", &EtaJ );
    reader_->AddVariable( "Beta", &Beta );
    reader_->AddVariable( "MultCalo", &MultCalo );
    reader_->AddVariable( "dAxis1c", &dAxis1c );
    reader_->AddVariable( "dAxis2c", &dAxis2c );
    reader_->AddVariable( "MultTr", &MultTr );
    reader_->AddVariable( "dAxis1t", &dAxis1t );
    reader_->AddVariable( "dAxis2t", &dAxis2t );

    reco::details::loadTMVAWeights(reader_, tmvaMethod_.c_str(), tmvaWeights_.c_str());

    //std::cout<<" Method booked "<<std::endl;


    readerF_ = new TMVA::Reader( "!Color:!Silent" );

    readerF_->AddVariable( "Nvtx", &Nvtx );
    readerF_->AddVariable( "PtJ", &PtJ );
    readerF_->AddVariable( "EtaJ", &EtaJ );
    readerF_->AddVariable( "MultCalo", &MultCalo );
    readerF_->AddVariable( "dAxis1c", &dAxis1c );
    readerF_->AddVariable( "dAxis2c", &dAxis2c );

    reco::details::loadTMVAWeights(readerF_, tmvaMethod_.c_str(), tmvaWeightsF_.c_str());

   // std::cout<<" Method booked F "<<std::endl;
  }

float PileupJPTJetIdAlgo::fillJPTBlock(const reco::JPTJet* jet
                                      )
{

      if (verbosity > 0) { 
	std::cout<<"================================    JET LOOP  ====================  "   <<  std::endl;
	std::cout<<"================    jetPt   =  "   << (*jet).pt()  <<  std::endl;
	std::cout<<"================    jetEta   =  "   << (*jet).eta()  <<  std::endl;
      }      

    edm::RefToBase<reco::Jet> jptjetRef = jet->getCaloJetRef();
    reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);
 

      int ncalotowers=0.;
      double sumpt=0.;
      double dphi2=0.;
      double deta2=0.;
      double dphi1=0.;
      double dphideta=0.;     
      double deta1=0.;
      double ffrac01=0.;
      double ffrac02=0.;
      double ffrac03=0.;
      double ffrac04=0.;
      double ffrac05=0.;
      double EE=0.;
      double HE=0.;
      double EELong=0.;
      double EEShort=0.;
 
      std::vector <CaloTowerPtr> calotwrs = (*rawcalojet).getCaloConstituents();

      if (verbosity > 0) { std::cout<<"=======    CaloTowerPtr DONE   ==  "   <<  std::endl;}      
      
      for(std::vector <CaloTowerPtr>::const_iterator icalot = calotwrs.begin();
	                                             icalot!= calotwrs.end(); icalot++) {
	ncalotowers++;
	
	double  deta=(*jet).eta()-(*icalot)->eta();
	double  dphi=(*jet).phi()-(*icalot)->phi();

	if(dphi > M_PI ) dphi = dphi-2.*M_PI;
	if(dphi < -1.*M_PI ) dphi = dphi+2.*M_PI;

       if (verbosity > 0)  std::cout<<" CaloTower jet eta "<<(*jet).eta()<<" tower eta "<<(*icalot)->eta()<<" jet phi "<<(*jet).phi()<<" tower phi "<<(*icalot)->phi()<<" dphi "<<dphi<<" "<<(*icalot)->pt()<<" ieta "<<(*icalot)->ieta()<<" "<<abs((*icalot)->ieta())<<std::endl;

        double dr = sqrt(dphi*dphi+deta*deta);
        double enc = (*icalot)->emEnergy()+(*icalot)->hadEnergy();
        if(dr < 0.1) ffrac01 = ffrac01 + enc;
        if(dr < 0.2) ffrac02 = ffrac02 + enc;
        if(dr < 0.3) ffrac03 = ffrac03 + enc;
        if(dr < 0.4) ffrac04 = ffrac04 + enc;
        if(dr < 0.5) ffrac05 = ffrac05 + enc;
	
        if(abs((*icalot)->ieta())<30) EE = EE + (*icalot)->emEnergy();
        if(abs((*icalot)->ieta())<30) HE = HE + (*icalot)->hadEnergy();
        if(abs((*icalot)->ieta())>29) EELong = EELong + (*icalot)->emEnergy();
        if(abs((*icalot)->ieta())>29) EEShort = EEShort + (*icalot)->hadEnergy();

	sumpt = sumpt + (*icalot)->pt();
	dphi2 = dphi2 + dphi*dphi*(*icalot)->pt();
	deta2 = deta2 + deta*deta*(*icalot)->pt();	
	dphi1 = dphi1 + dphi*(*icalot)->pt();
        deta1 = deta1 + deta*(*icalot)->pt();	
        dphideta = dphideta + dphi*deta*(*icalot)->pt();	

      } // calojet constituents
      
            
      if( sumpt > 0.) {
		      deta1 = deta1/sumpt;
		      dphi1 = dphi1/sumpt;
		      deta2 = deta2/sumpt;
		      dphi2 = dphi2/sumpt;
		      dphideta = dphideta/sumpt;
      }
      
  // W.r.t. principal axis

      double detavar = deta2-deta1*deta1;
      double dphivar = dphi2-dphi1*dphi1;
      double dphidetacov = dphideta - deta1*dphi1;
      
      double det = (detavar-dphivar)*(detavar-dphivar)+4*dphidetacov*dphidetacov;
      det = sqrt(det);
      double x1 = (detavar+dphivar+det)/2.;
      double x2 = (detavar+dphivar-det)/2.;
      
      
  // Energy fraction in cone
 
      ffrac01 = ffrac01/(*jet).energy();
      ffrac02 = ffrac02/(*jet).energy();
      ffrac03 = ffrac03/(*jet).energy();
      ffrac04 = ffrac04/(*jet).energy();
      ffrac05 = ffrac05/(*jet).energy();
  
if (verbosity > 0)  
std::cout<<" ncalo "<<ncalotowers<<" deta2 "<<deta2<<" dphi2 "<<dphi2<<" deta1 "<<deta1<<" dphi1 "<<dphi1<<" detavar "<<detavar<<" dphivar "<<dphivar<<" dphidetacov "<<dphidetacov<<" sqrt(det) "<<sqrt(det)<<" x1 "<<x1<<" x2 "<<x2<<std::endl;

 
// For jets with |eta|<2 take also tracks shape
      int ntracks=0.;
      double sumpttr=0.;
      double dphitr2=0.;
      double detatr2=0.;
      double dphitr1=0.;
      double detatr1=0.;
      double dphidetatr=0.;     

      const reco::TrackRefVector pioninin = (*jet).getPionsInVertexInCalo();
      
      for(reco::TrackRefVector::const_iterator it = pioninin.begin(); it != pioninin.end(); it++) {      
            if ((*it)->pt() > 0.5 && ((*it)->ptError()/(*it)->pt()) < 0.05 )
              {              
                 ntracks++;
		 sumpttr = sumpttr + (*it)->pt();	
		 double  deta=(*jet).eta()-(*it)->eta();
	         double  dphi=(*jet).phi()-(*it)->phi();

	         if(dphi > M_PI ) dphi = dphi-2.*M_PI;
	         if(dphi < -1.*M_PI ) dphi = dphi+2.*M_PI;

	         dphitr2 = dphitr2 + dphi*dphi*(*it)->pt();
	         detatr2 = detatr2 + deta*deta*(*it)->pt();	
	         dphitr1 = dphitr1 + dphi*(*it)->pt();
                 detatr1 = detatr1 + deta*(*it)->pt();	
                 dphidetatr = dphidetatr + dphi*deta*(*it)->pt();
        if(verbosity>0) std::cout<<" Tracks-in-in "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<" in jet "<<(*jet).eta()<<" "<<
         (*jet).phi()<<" jet pt "<<(*jet).pt()<<std::endl;	
              }
      }// pioninin
      
      
       const reco::TrackRefVector pioninout = (*jet).getPionsInVertexOutCalo();
      
      for(reco::TrackRefVector::const_iterator it = pioninout.begin(); it != pioninout.end(); it++) {      
            if ((*it)->pt() > 0.5 && ((*it)->ptError()/(*it)->pt()) < 0.05 )
              {              
                 ntracks++;
		 sumpttr = sumpttr + (*it)->pt();	
		 double  deta=(*jet).eta()-(*it)->eta();
	         double  dphi=(*jet).phi()-(*it)->phi();

	         if(dphi > M_PI ) dphi = dphi-2.*M_PI;
	         if(dphi < -1.*M_PI ) dphi = dphi+2.*M_PI;

	         dphitr2 = dphitr2 + dphi*dphi*(*it)->pt();
	         detatr2 = detatr2 + deta*deta*(*it)->pt();	
	         dphitr1 = dphitr1 + dphi*(*it)->pt();
                 detatr1 = detatr1 + deta*(*it)->pt();	
                 dphidetatr = dphidetatr + dphi*deta*(*it)->pt();	
        if(verbosity>0) std::cout<<" Tracks-in-in "<<(*it)->pt()<<" "<<(*it)->eta()<<" "<<(*it)->phi()<<" in jet "<<(*jet).eta()<<" "<< 
         (*jet).phi()<<" jet pt "<<(*jet).pt()<<std::endl;
              }
      }// pioninout

      if(verbosity>0) std::cout<<" Number of tracks in-in and in-out "<<pioninin.size()<<" "<<pioninout.size()<<std::endl;

      	       
      if( sumpttr > 0.) {
		      detatr1 = detatr1/sumpttr;
		      dphitr1 = dphitr1/sumpttr;
		      detatr2 = detatr2/sumpttr;
		      dphitr2 = dphitr2/sumpttr;
		      dphidetatr = dphidetatr/sumpttr;
      }

  // W.r.t. principal axis

      double detavart = detatr2-detatr1*detatr1;
      double dphivart = dphitr2-dphitr1*dphitr1;
      double dphidetacovt = dphidetatr - detatr1*dphitr1;
      
      double dettr = (detavart-dphivart)*(detavart-dphivart)+4*dphidetacovt*dphidetacovt;
      dettr = sqrt(dettr);
      double x1tr = (detavart+dphivart+dettr)/2.;
      double x2tr = (detavart+dphivart-dettr)/2.;
     
        if (verbosity > 0) std::cout<<" ntracks "<<ntracks<<" detatr2 "<<detatr2<<" dphitr2 "<<dphitr2<<" detatr1 "<<detatr1<<" dphitr1 "<<dphitr1<<" detavart "<<detavart<<" dphivart "<<dphivart<<" dphidetacovt "<<dphidetacovt<<" sqrt(det) "<<sqrt(dettr)<<" x1tr "<<x1tr<<" x2tr "<<x2tr<<std::endl;
 
// Multivariate analisis
      PtJ = (*jet).pt();          
      EtaJ = (*jet).eta();
      Beta = (*jet).getSpecific().Zch;
      dAxis2c = x2;
      dAxis1c = x1;
      dAxis2t = x2tr;
      dAxis1t = x1tr;
      MultCalo = ncalotowers;
      MultTr = ntracks;

     float mva_ = 1.;
     if(fabs(EtaJ)<2.6) {
       mva_ = reader_->EvaluateMVA( "BDTG method" );
      // std::cout<<" MVA analysis "<<mva_<<std::endl;
      } else {
        mva_ = readerF_->EvaluateMVA( "BDTG method" );
      // std::cout<<" MVA analysis Forward "<<mva_<<std::endl;
      }
  if (verbosity > 0) std::cout<<"=======  Computed MVA =  " << 
                                           mva_  <<std::endl;   
  return mva_;
} // fillJPTBlock
} // namespace
