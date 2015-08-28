#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DQM/PhysicsHWW/interface/MVAJetIdMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;

MVAJetIdMaker::MVAJetIdMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  PFJetCollection_     = iCollector.consumes<reco::PFJetCollection> (iConfig.getParameter<edm::InputTag>("pfJetsInputTag"));
  thePVCollection_     = iCollector.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));
  jetCorrector_        = iConfig.getParameter<std::string>("jetCorrector");
  theRhoCollection_    = iCollector.consumes<double>(iConfig.getParameter<edm::InputTag>("rhoInputTag"));

  fPUJetIdAlgo = new PileupJetIdAlgo(iConfig.getParameter<edm::ParameterSet>("puJetIDParams"),true);

}

bool passPFLooseId(const reco::PFJet *iJet) {
	if(iJet->energy()== 0)                                  return false;
	if(iJet->neutralHadronEnergy()/iJet->energy() > 0.99)   return false;
	if(iJet->neutralEmEnergy()/iJet->energy()     > 0.99)   return false;
	if(iJet->nConstituents() <  2)                          return false;
	if(iJet->chargedHadronEnergy()/iJet->energy() <= 0 && fabs(iJet->eta()) < 2.4 ) return false;
	if(iJet->chargedEmEnergy()/iJet->energy() >  0.99  && fabs(iJet->eta()) < 2.4 ) return false;
	if(iJet->chargedMultiplicity()            < 1      && fabs(iJet->eta()) < 2.4 ) return false;
	return true;
}
          
void MVAJetIdMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup){

  using namespace std;
  using namespace edm;
  using namespace reco;

  hww.Load_pfjets_corr_p4();
  hww.Load_pfjets_mvavalue();
  hww.Load_pfjets_JEC();

  bool validToken;

  //Uncorrected Jets
  Handle<PFJetCollection>       lHUCJets;
  validToken = iEvent.getByToken(PFJetCollection_, lHUCJets);
  if(!validToken) return;
  PFJetCollection               lUCJets = *lHUCJets;

  // vertices    
  Handle<reco::VertexCollection> lHVertices;
  validToken = iEvent.getByToken(thePVCollection_, lHVertices); 
  if(!validToken) return;
  VertexCollection lVertices = *lHVertices;

  const JetCorrector* corrector=0;
  corrector = JetCorrector::getJetCorrector(jetCorrector_, iSetup); 

  Handle<double> lHrho;
  validToken = iEvent.getByToken(theRhoCollection_, lHrho); 
  if(!validToken) return;
  double lrho = *lHrho;

  std::vector<reco::PFJet> lCJets;
  for(reco::PFJetCollection::const_iterator jet=lUCJets.begin(); jet!=lUCJets.end(); ++jet){

    reco::PFJet tempJet = *jet; 
    tempJet.scaleEnergy(corrector ? corrector->correction(*jet, iEvent, iSetup) : 1.);
    lCJets.push_back(tempJet);

  }


  // select good vertices 
  // make new collection to put into computeIdVariables(...)
  VertexCollection lGoodVertices;
  for(int ivtx    = 0; ivtx < (int)lVertices.size(); ivtx++) {

	  const Vertex       *vtx = &(lVertices.at(ivtx));
	  if( vtx->isFake()               		)  continue;
	  if( vtx->ndof()<=4              		)  continue;
	  if( vtx->position().Rho()>2.0   		)  continue;
	  if( fabs(vtx->position().Z())>24.0  )  continue;
	  lGoodVertices.push_back(*vtx);
  }

  // loop over jets 
  for(int i0   = 0; i0 < (int) lUCJets.size(); i0++) {   // uncorrected jets collection
	  const PFJet       *pUCJet = &(lUCJets.at(i0));
	  for(int i1 = 0; i1 < (int) lCJets.size(); i1++) {   // corrected jets collection
		  const PFJet     *pCJet  = &(lCJets.at(i1));
		  if( pUCJet->jetArea() != pCJet->jetArea()                  	) continue;
		  if( fabs(pUCJet->eta() - pCJet->eta())         > 0.001         ) continue;
      	  if( pUCJet->pt()                               < 0.0    ) continue;
		  double lJec = pCJet ->pt()/pUCJet->pt();

		  // calculate mva value only when there are good vertices 
		  // otherwise store -999
		  if( lGoodVertices.size()>0 ) {
		  	PileupJetIdentifier lPUJetId =  fPUJetIdAlgo->computeIdVariables(pCJet,lJec,&lGoodVertices[0],lGoodVertices,lrho);
		   	hww.pfjets_mvavalue() .push_back( lPUJetId.mva()              );
        hww.pfjets_JEC() .push_back( lJec ); 
		  
        // print out MVA inputs 
        if(0) {

          LogDebug("MVAJetIdMaker")
            << "Debug Jet MVA : "
            << iEvent.id() 			<< " : "
            << lPUJetId.nvtx()      << " "
            << pCJet->pt()         	<< " "
            << lPUJetId.jetEta()    << " "
            << lPUJetId.jetPhi()    << " "
            << lPUJetId.d0()        << " "
            << lPUJetId.dZ()        << " "
            << lPUJetId.beta()      << " "
            << lPUJetId.betaStar()  << " "
            << lPUJetId.nCharged()  << " "
            << lPUJetId.nNeutrals() << " "
            << lPUJetId.dRMean()    << " "
            << lPUJetId.frac01()    << " "
            << lPUJetId.frac02()    << " "
            << lPUJetId.frac03()    << " "
            << lPUJetId.frac04()    << " "
            << lPUJetId.frac05()
            << " === : === "
            << lPUJetId.mva();
        }
		  }
		  else             

		  	hww.pfjets_mvavalue() .push_back( -999. );

		  break;

	  } // lCJets
  } // lUCJets
}
