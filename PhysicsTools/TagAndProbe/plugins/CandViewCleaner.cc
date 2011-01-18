////////////////////////////////////////////////////////////////////////////////
//
// CandViewCleaner
// --------------
//
////////////////////////////////////////////////////////////////////////////////


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////

class CandViewCleaner : public edm::EDProducer
{
public:
  // construction/destruction
  CandViewCleaner(const edm::ParameterSet& iConfig);
  virtual ~CandViewCleaner();
  
  // member functions
  void produce(edm::Event& iEvent,const edm::EventSetup& iSetup);
  void endJob();

private:  
  // member data
  edm::InputTag              srcCands_;
  std::vector<edm::InputTag> srcObjects_;
  double                     deltaRMin_;
  
  std::string  moduleLabel_;
  unsigned int nCandidatesTot_;
  unsigned int nCandidatesClean_;
};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
CandViewCleaner::CandViewCleaner(const edm::ParameterSet& iConfig)
  : srcCands_    (iConfig.getParameter<edm::InputTag>         ("srcCands"))
  , srcObjects_ (iConfig.getParameter<vector<edm::InputTag> >("srcObjects"))
  , deltaRMin_  (iConfig.getParameter<double>                ("deltaRMin"))
  , moduleLabel_(iConfig.getParameter<string>                ("@module_label"))
  , nCandidatesTot_(0)
  , nCandidatesClean_(0)
{
  produces<edm::RefToBaseVector<reco::Candidate> >();
}


//______________________________________________________________________________
CandViewCleaner::~CandViewCleaner()
{
  
}



////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void CandViewCleaner::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  auto_ptr<edm::RefToBaseVector<reco::Candidate> >
    cleanCandidates(new edm::RefToBaseVector<reco::Candidate>());

  edm::Handle<reco::CandidateView> candidates;
  iEvent.getByLabel(srcCands_,candidates);
  
  bool* isClean = new bool[candidates->size()];
  for (unsigned int iCandidate=0;iCandidate<candidates->size();iCandidate++) isClean[iCandidate] = true;
  
  for (unsigned int iSrc=0;iSrc<srcObjects_.size();iSrc++) {
    edm::Handle<reco::CandidateView> objects;
    iEvent.getByLabel(srcObjects_[iSrc],objects);
    
    for (unsigned int iCandidate=0;iCandidate<candidates->size();iCandidate++) {
      const reco::Candidate& candidate = candidates->at(iCandidate);
      for (unsigned int iObj=0;iObj<objects->size();iObj++) {
	const reco::Candidate& obj = objects->at(iObj);
	double deltaR = reco::deltaR(candidate,obj);
	if (deltaR<deltaRMin_)  isClean[iCandidate] = false;
      }
    }
  }
  
  for (unsigned int iCandidate=0;iCandidate<candidates->size();iCandidate++)
    if (isClean[iCandidate]) cleanCandidates->push_back(candidates->refAt(iCandidate));
  
  nCandidatesTot_  +=candidates->size();
  nCandidatesClean_+=cleanCandidates->size();

  delete [] isClean;  
  iEvent.put(cleanCandidates);
}


//______________________________________________________________________________
void CandViewCleaner::endJob()
{
  stringstream ss;
  ss<<"nCandidatesTot="<<nCandidatesTot_<<" nCandidatesClean="<<nCandidatesClean_
    <<" fCandidatesClean="<<100.*(nCandidatesClean_/(double)nCandidatesTot_)<<"%\n";
  cout<<"++++++++++++++++++++++++++++++++++++++++++++++++++"
      <<"\n"<<moduleLabel_<<"(CandViewCleaner) SUMMARY:\n"<<ss.str()
      <<"++++++++++++++++++++++++++++++++++++++++++++++++++"
      <<endl;
}


////////////////////////////////////////////////////////////////////////////////
// plugin definition
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(CandViewCleaner);

