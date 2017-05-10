#include "GeneratorInterface/GenFilters/interface/XtoFFbarFilter.h"
#include <unordered_set>
using namespace std;
using namespace reco;

XtoFFbarFilter::XtoFFbarFilter(const edm::ParameterSet& iConfig) :
  src_(iConfig.getParameter<edm::InputTag>("src")),
  idMotherX_(iConfig.getParameter<vector<int> >("idMotherX")),
  idDaughterF_(iConfig.getParameter<vector<int> >("idDaughterF")),
  idMotherY_(iConfig.getParameter<vector<int> >("idMotherY")),
  idDaughterG_(iConfig.getParameter<vector<int> >("idDaughterG")),
  idYequalsX_(iConfig.getParameter<bool>("idYequalsX")),
  xTotal_(0), xSumPt_(0.), xSumR_(0.), xSumCtau_(0.), totalEvents_(0), keptEvents_(0)
{ 
  if (idYequalsX_) idMotherY_ = idMotherX_;
  // Note if if not searching for Y --> g-gbar.
  requireY_ = (idMotherY_.size() > 0 && idDaughterG_.size() > 0);
}


bool XtoFFbarFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)  {

  bool keep = false;

  iEvent.getByLabel(src_, genParticles_);

  totalEvents_++;

  unsigned int numX = 0;
  unsigned int numY = 0;
  unsigned int numXorY = 0;

  unordered_set<int> xPresent;

  for (unsigned int j = 0; j < genParticles_->size(); j++) {
    GenParticleRef moth(genParticles_, j);

    // Is it X -> f fbar ?
    bool isXtoFFbar = this->foundXtoFFbar(moth, idMotherX_, idDaughterF_);
    if (isXtoFFbar) numX++; 

    if (!requireY_) {

      // Has X been found already ?
      if (numX >= 1) keep = true;

    } else {

      // Is it Y -> g gbar ?
      bool isYtoGGbar = this->foundXtoFFbar(moth, idMotherY_, idDaughterG_);
      if (isYtoGGbar) numY++;
      if (isXtoFFbar || isYtoGGbar) numXorY++;

      // Have X and Y been found already ?
      if (numX >= 1 && numY >= 1 && numXorY >= 2) keep = true;
    }

    // Note which species listed in idMotherX_ are present in event.
    if (idYequalsX_) {
      int pdgIdMoth = moth->pdgId();
      if (this->found(idMotherX_, pdgIdMoth)) xPresent.insert(pdgIdMoth);
    }
  }

  // If requested veto events where more than one species listed in idMotherX_ is present.
  if (idYequalsX_ && xPresent.size() > 1) keep = false;

  if (keep) {
    keptEvents_++;
    //    cout<<"XtoFFbarFilter: KEPT EVENT"<<endl;
    //  } else {
    //    cout<<"XtoFFbarFilter: REJECTED EVENT"<<endl;
  }
  return keep;
}


bool XtoFFbarFilter::foundXtoFFbar(const GenParticleRef& moth, 
				   const vector<int>& idMother, 
				   const vector<int>& idDaughter) {
  // Check if given particle "moth" is X-->f f'-bar + anything
  bool isXtoFFbar = false;
  int pdgIdMoth = moth->pdgId();
  double rho = -9.9e9;

  if (this->found(idMother, pdgIdMoth)) {
    bool foundF    = false;
    bool foundFbar = false;
    unsigned int nDau = moth->numberOfDaughters();

    for (unsigned int i = 0; i < nDau; i++) {
      GenParticleRef dau = moth->daughterRef(i);
      int pdgIdDau = dau->pdgId();
      if (this->found(idDaughter, -pdgIdDau))  foundFbar = true; 
      if (this->found(idDaughter,  pdgIdDau)) {foundF    = true; 

        // Just for statistics, get transverse decay length.
        // (To be really accurate, should do it w.r.t. P.V., but couldn't be bothered ...)
	// This is the normal case
        rho = dau->vertex().Rho();
        // Unfortunately, duplicate particles can appear in the event record. Handle this as follows:
        for (unsigned int j = 0; j < dau->numberOfDaughters(); j++) {
          GenParticleRef granddau = dau->daughterRef(j);        
          if (granddau->pdgId() == pdgIdDau) rho = granddau->vertex().Rho();
        }
      }
    }
    if (foundF && foundFbar) isXtoFFbar = true;
  }

  if (isXtoFFbar) {
    // Get statistics 
    xTotal_++;
    xSumPt_ += moth->pt();
    xSumR_  += rho;
    xSumCtau_ += rho*(moth->mass()/(moth->pt()+0.01)); // protection against unlikely case Pt = 0.
  }

  return isXtoFFbar;
}

void XtoFFbarFilter::endJob() {
  cout<<endl;
  cout<<"=== XtoFFbarFilter statistics of selected X -> f f'-bar + anything or Y -> g g'-bar + anything"<<endl;
  if (xTotal_ > 0) {
    cout<<"===   mean X & Y Pt = "<<xSumPt_/xTotal_<<" GeV and transverse decay length = "<<xSumR_/xTotal_<<" cm"<<endl;
    cout<<"===   mean c*tau = "<<xSumCtau_/xTotal_<<" cm"<<endl;
  } else {
    cout<<"===   WARNING: NONE FOUND !"<<endl;
  }
  cout<<"===   events kept = "<<keptEvents_<<"/"<<totalEvents_<<endl;                     
}
