/** \class HLTOpenBJetAnalyser
 *
 *  $Date: 2008/04/23 11:57:53 $
 *  $Revision: 1.1 $
 *
 *  \author Andrea Bocci
 *
 */

#include <string>

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

class edm::Event;
class edm::EventSetup;
class edm::ParameterSet;

// class declaration
class HLTOpenBJetAnalyser : public edm::EDAnalyzer
{
public:
  explicit HLTOpenBJetAnalyser(const edm::ParameterSet & config);
  ~HLTOpenBJetAnalyser();

  virtual void produce(const edm::Event & event, const edm::EventSetup & setup);

private:
  edm::InputTag m_hltopen;
  int    m_jets;
  double m_cutL2;
  double m_cutL25;
  double m_cutL3;
  bool   m_useHTT;
};


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

// constructors and destructor
HLTOpenBJetAnalyser::HLTOpenBJetAnalyser(const edm::ParameterSet & config) :
  m_hltopen( config.getParameter<edm::InputTag> ("hltOpenBJets") ),
  m_jets(    config.getParameter<int>("requiredJets")),
  m_cutL2(   config.getParameter<double>("energyCut")),
  m_cutL25(  config.getParameter<double>("discriminantL25Cut")),
  m_cutL3(   config.getParameter<double>("discriminantL3Cut")),
  m_useHTT(  config.getParameter<bool>("cutOnHTT"))
{
}

HLTOpenBJetAnalyser::~HLTOpenBJetAnalyser()
{
}

// member functions

bool
HLTOpenBJetAnalyser::filter(edm::Event& event, const edm::EventSetup& setup)
{
  
  edm::Handle<edm::View<reco::Jet> > h_hltopen;
  event.getByLabel(m_hltopen, h_hltopen);
  const trigger::HLTOpenBJet & hltopen = * h_hltopen;

  if (m_useHTT) {
    // cut on total transverse hadronic energy
    return (hltopen.hadronicEnergy()     > m_cutL2)
       and (hltopen.jetDiscriminantL25() > m_cutL25)
       and (hltopen.jetDiscriminantL3()  > m_cutL3);
  } else {
    // cut on n-th jet energy
    return (hltopen.jetEnergy(m_jets-1)        > m_cutL2)
       and (hltopen.jetDiscriminantL25(m_jets) > m_cutL25)
       and (hltopen.jetDiscriminantL3(m_jets)  > m_cutL3);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTOpenBJetAnalyser);
