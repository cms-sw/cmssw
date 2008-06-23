// File: TauMETAlgo.cc
// Description:  see TauMETAlgo.h
// Author: C.N.Nguyen
// Creation Date:  October 22, 2007 Initial version.
//
//--------------------------------------------

#include "JetMETCorrections/Type1MET/src/TauMETAlgo.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace reco;

  TauMETAlgo::TauMETAlgo() {}
  TauMETAlgo::~TauMETAlgo() {}

  void TauMETAlgo::run(edm::Event& iEvent,const edm::EventSetup& iSetup,
		       edm::Handle<PFJetCollection> pfjetHandle,edm::Handle<CaloJetCollection> calojetHandle, 
                       const JetCorrector& correctedjets,bool usecorrectedjets,
		       double jetMatchDeltaR,METCollection* corrMET) 
  {
    std::cerr << "TauMETAlgo::run -> Test.. " << std::endl;

    // initialise vertex pos.
    Point p3(0.,0.,0.);
    // Minimum correction with PFjets & CaloJets:
    int pfjetID = 0;
    for (PFJetCollection::const_iterator pfjetIter = pfjetHandle->begin();
	 pfjetIter != pfjetHandle->end();
	 ++pfjetIter) {

      bool matchFlag = false;
      for (CaloJetCollection::const_iterator calojetIter = calojetHandle->begin();
	   calojetIter != calojetHandle->end();
	   ++calojetIter) {
	
	//std::cerr << "DeltaR: " << deltaR(calojetIter->p4(),pfjetIter->p4()) << std::endl;
	if (deltaR(calojetIter->p4(),pfjetIter->p4())<jetMatchDeltaR) {
	  //std::cerr << "  PFJet: " << pfjetIter->eta() << " " << pfjetIter->phi() << " " << pfjetIter->et() << std::endl;
	  //std::cerr << "CaloJet: " << calojetIter->eta() << " " << calojetIter->phi() << " " << calojetIter->et() << std::endl;

          if(usecorrectedjets) {
            double correct = correctedjets.correction (*calojetIter);
            LorentzVector v4((correct * calojetIter->px())-pfjetIter->px(),(correct * calojetIter->py())-pfjetIter->py(),0.,0.);
	    MET m(v4,p3);
	    corrMET->push_back(m);
	    //corrMET[pfjetID] = m;
          } else {
            LorentzVector v4(calojetIter->px()-pfjetIter->px(),calojetIter->py()-pfjetIter->py(),0.,0.);
	    MET m(v4,p3);
	    corrMET->push_back(m);
	    //corrMET[pfjetID] = m;
          }

	  if (matchFlag) {
	    std::cerr << "### TauMETAlgo - ERROR:  Multiple jet matches!!!! " << std::endl;
	  }
	  matchFlag = true;
	}
      }

      if (!matchFlag) {
	//std::cerr << "  No match!!!! " << std::endl;
	LorentzVector v4(0.,0.,0.,0.);
	MET m(v4,p3);
	corrMET->push_back(m);
	//corrMET[pfjetID] = m;
      }
      
      
      pfjetID++;
    }

  }

