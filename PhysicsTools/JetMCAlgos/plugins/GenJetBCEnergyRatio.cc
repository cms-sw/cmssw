//
// Plugin to store B and C ratio for a GenJet in the event
// Author: Attilio
// Date: 05.10.2007
//

//=======================================================================

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;

class GenJetBCEnergyRatio : public edm::EDProducer
{
  public:
    GenJetBCEnergyRatio( const edm::ParameterSet & );
    ~GenJetBCEnergyRatio();

    typedef reco::JetFloatAssociation::Container JetBCEnergyRatioCollection;

  private:
    virtual void produce(edm::Event&, const edm::EventSetup& ) override;
    Handle< View <Jet> > genjets;
    edm::EDGetTokenT< View <Jet> > m_genjetsSrcToken;

};

//=========================================================================

GenJetBCEnergyRatio::GenJetBCEnergyRatio( const edm::ParameterSet& iConfig )
{
    produces<JetBCEnergyRatioCollection>("bRatioCollection");
    produces<JetBCEnergyRatioCollection>("cRatioCollection");
    m_genjetsSrcToken = consumes< View <Jet> >(iConfig.getParameter<edm::InputTag>("genJets"));
}

//=========================================================================

GenJetBCEnergyRatio::~GenJetBCEnergyRatio()
{
}

// ------------ method called to produce the data  ------------

void GenJetBCEnergyRatio::produce( Event& iEvent, const EventSetup& iEs )
{
  iEvent.getByToken(m_genjetsSrcToken, genjets);

  typedef edm::RefToBase<reco::Jet> JetRef;

  JetBCEnergyRatioCollection * jtc1;
  JetBCEnergyRatioCollection * jtc2;

  if (genjets.product()->size() > 0) {
    const JetRef jj = genjets->refAt(0);
    jtc1 = new JetBCEnergyRatioCollection(edm::makeRefToBaseProdFrom(jj, iEvent));
    jtc2 = new JetBCEnergyRatioCollection(edm::makeRefToBaseProdFrom(jj, iEvent));
  } else {
    jtc1 = new JetBCEnergyRatioCollection();
    jtc2 = new JetBCEnergyRatioCollection();
  }

  std::auto_ptr<JetBCEnergyRatioCollection> bRatioColl(jtc1);
  std::auto_ptr<JetBCEnergyRatioCollection> cRatioColl(jtc2);

  for( size_t j = 0; j != genjets->size(); ++j ) {

    float bRatio = EnergyRatioFromBHadrons( (*genjets)[j] );
    float cRatio = EnergyRatioFromCHadrons( (*genjets)[j] );

    const JetRef & aJet = genjets->refAt(j) ;

    JetFloatAssociation::setValue(*bRatioColl, aJet, bRatio);
    JetFloatAssociation::setValue(*cRatioColl, aJet, cRatio);

  }


  iEvent.put(bRatioColl, "bRatioCollection");
  iEvent.put(cRatioColl, "cRatioCollection");

}

//define this as a plug-in
DEFINE_FWK_MODULE(GenJetBCEnergyRatio);

