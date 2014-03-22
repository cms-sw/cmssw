/** \class HLTMuonPFIsoFilter
 *
 * See header file for documentation
 *
 *
 */   

#include "HLTrigger/Muon/interface/HLTMuonPFIsoFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <iostream>
//
// constructors and destructor
//
HLTMuonPFIsoFilter::HLTMuonPFIsoFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
   candTag_ 	      (iConfig.getParameter< edm::InputTag > ("CandTag") ),
   previousCandTag_   (iConfig.getParameter< edm::InputTag > ("PreviousCandTag")),
   depTag_  	      (iConfig.getParameter< std::vector< edm::InputTag > >("DepTag" ) ),
   depToken_(0),
   rhoTag_  	      (iConfig.getParameter< edm::InputTag >("RhoTag" ) ),
   maxIso_  	      (iConfig.getParameter<double>("MaxIso" ) ),
   min_N_   	      (iConfig.getParameter<int> ("MinN")),
   onlyCharged_	      (iConfig.getParameter<bool> ("onlyCharged")),
   doRho_	      (iConfig.getParameter<bool> ("applyRhoCorrection")),
   effArea_	      (iConfig.getParameter<double> ("EffectiveArea"))
{
  std::stringstream tags;
  for (unsigned int i=0;i!=depTag_.size();++i) {
    depToken_.push_back(consumes<edm::ValueMap<double> >(depTag_[i]));
    tags<<" PFIsoTag["<<i<<"] : "<<depTag_[i].encode()<<" \n";
  }

  candToken_            = consumes<reco::RecoChargedCandidateCollection>(candTag_);
  previousCandToken_    = consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_);
  if (doRho_) rhoToken_ = consumes<double>(rhoTag_);

  LogDebug("HLTMuonPFIsoFilter") << " candTag : " << candTag_.encode()
				<< "\n" << tags 
				<< "  MinN : " << min_N_;

  produces<edm::ValueMap<bool> >();
}

 
HLTMuonPFIsoFilter::~HLTMuonPFIsoFilter()
{
}

//
// member functions
//
void
HLTMuonPFIsoFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  std::vector<edm::InputTag> depTag(1,edm::InputTag("hltMuPFIsoValueCharged03"));
  desc.add<std::vector<edm::InputTag> >("DepTag",depTag);
  desc.add<edm::InputTag>("RhoTag",edm::InputTag("hltFixedGridRhoFastjetAllCaloForMuonsPF"));
  desc.add<double>("MaxIso",1.);
  desc.add<int>("MinN",1);
  desc.add<bool>("onlyCharged",false);
  desc.add<bool>("applyRhoCorrection",true);
  desc.add<double>("EffectiveArea",1.);
  descriptions.add("hltMuonPFIsoFilter", desc);
}

// ------------ method called to produce the data  ------------
 bool
 HLTMuonPFIsoFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
 {
    using namespace std;
    using namespace edm;
    using namespace trigger;
    using namespace reco;
 
    // All HLT filters must create and fill an HLT filter object,
    // recording any reconstructed physics objects satisfying (or not)
    // this HLT filter, and place it in the Event.
 
    //the decision map
    std::auto_ptr<edm::ValueMap<bool> > PFisoMap( new edm::ValueMap<bool> ());
 
    // get hold of trks
    Handle<RecoChargedCandidateCollection> mucands;
    if (saveTags()) filterproduct.addCollectionTag(candTag_);
    iEvent.getByToken (candToken_,mucands);
    Handle<TriggerFilterObjectWithRefs> previousLevelCands;
    iEvent.getByToken (previousCandToken_,previousLevelCands);
    vector<RecoChargedCandidateRef> vcands;
    previousLevelCands->getObjects(TriggerMuon,vcands);
    
    //get hold of energy deposition
    unsigned int nDep=depTag_.size();
    std::vector< Handle<edm::ValueMap<double> > > depMap(nDep);
    
    //get hold of rho of the event
    double Rho = 0;
    if (doRho_){
      Handle <double>  RhoCorr;
      iEvent.getByToken(rhoToken_, RhoCorr);
      Rho = *RhoCorr.product();
    }
 
    for (unsigned int i=0;i!=nDep;++i) iEvent.getByToken (depToken_[i],depMap[i]);

    // look at all mucands,  check cuts and add to filter object
    int nIsolatedMu = 0;
    unsigned int nMu=mucands->size();
    std::vector<bool> isos(nMu, false);

    unsigned int iMu=0;
    for (; iMu<nMu; iMu++) 
    {
      double MuonDeposits = 0;
      RecoChargedCandidateRef candref(mucands,iMu);
      LogDebug("HLTMuonPFIsoFilter") << "candref isNonnull " << candref.isNonnull(); 

      //did this candidate triggered at previous stage.
      if (!triggerdByPreviousLevel(candref,vcands)) continue;

      //reference to the track
      TrackRef tk = candref->get<TrackRef>();
      LogDebug("HLTMuonPFIsoFilter") << "tk isNonNull " << tk.isNonnull();

      //get the deposits and evaluate relIso if only the charged component is considered
      if (onlyCharged_){
	for(unsigned int iDep=0;iDep!=nDep;++iDep)
	  {
	    const edm::ValueMap<double> ::value_type & muonDeposit = (*(depMap[iDep]))[candref];
	    LogDebug("HLTMuonPFIsoFilter") << " Muon with q*pt= " << tk->charge()*tk->pt() 
					   << " (" << candref->charge()*candref->pt() << ") " 
					   << ", eta= " << tk->eta() << " (" << candref->eta() << ") " 
					   << "; has deposit["<<iDep<<"]: " << muonDeposit;
	    
	    std::size_t foundCharged = depTag_[iDep].label().find("Charged");
	    if (foundCharged!=std::string::npos)  MuonDeposits += muonDeposit; 
	  }
	MuonDeposits = MuonDeposits/tk->pt();
      }
      else {
	//get all the deposits 
	for(unsigned int iDep=0;iDep!=nDep;++iDep)
	  {
	    const edm::ValueMap<double> ::value_type & muonDeposit = (*(depMap[iDep]))[candref];
	    LogDebug("HLTMuonPFIsoFilter") << " Muon with q*pt= " << tk->charge()*tk->pt() 
					   << " (" << candref->charge()*candref->pt() << ") " 
					   << ", eta= " << tk->eta() << " (" << candref->eta() << ") " 
					   << "; has deposit["<<iDep<<"]: " << muonDeposit;
	    MuonDeposits += muonDeposit;
	  }
        //apply rho correction 
	if (doRho_) MuonDeposits -=  effArea_*Rho;
	MuonDeposits = MuonDeposits/tk->pt();
      }
      
      //get the selection
      if (MuonDeposits < maxIso_) isos[iMu] = true;

      LogDebug("HLTMuonPFIsoFilter") << " Muon with q*pt= " << tk->charge()*tk->pt() << ", eta= " << tk->eta() 
				     << "; "<<(isos[iMu]?"Is an isolated muon.":"Is NOT an isolated muon.");
       
      if (!isos[iMu]) continue;

      nIsolatedMu++;
      filterproduct.addObject(TriggerMuon,candref);
    }//for iMu

    // filter decision
    const bool accept (nIsolatedMu >= min_N_);

    //put the decision map
    if (nMu!=0)
    {
      edm::ValueMap<bool> ::Filler isoFiller(*PFisoMap);     
      isoFiller.insert(mucands, isos.begin(), isos.end());
      isoFiller.fill();
    }

    iEvent.put(PFisoMap);

    LogDebug("HLTMuonPFIsoFilter") << " >>>>> Result of HLTMuonPFIsoFilter is " << accept << ", number of muons passing isolation cuts= " << nIsolatedMu; 
    return accept;
 }
 

bool HLTMuonPFIsoFilter::triggerdByPreviousLevel(const reco::RecoChargedCandidateRef & candref, const std::vector<reco::RecoChargedCandidateRef>& vcands){
  unsigned int i=0;
  unsigned int i_max=vcands.size();
  for (;i!=i_max;++i){
    if (candref == vcands[i]) return true;
  }

  return false;
}
