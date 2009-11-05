#include "AnalysisDataFormats/EWK/interface/WMuNuCandidatePtr.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace edm;
using namespace std;
using namespace reco;

WMuNuCandidatePtr::WMuNuCandidatePtr(){}

WMuNuCandidatePtr::WMuNuCandidatePtr(const reco::CandidatePtr muon, const reco::CandidatePtr met): muon_(muon), neutrino_(met)  
{
      addDaughter(muon_);
      addDaughter(neutrino_);
      AddFourMomenta addP4;
      addP4.set(*this);
      cout<<"WCandidatePtr Created   Wpx="<<muon_->px()<<" + "<<neutrino_->px()<<" ?= "<<this->px()<<endl;
      //WARNING: W CandidatePtrs combine the information from a Muon with the (px,py) information of the MET as the Neutrino
      // --> There is no Pz information!!!!
      // Be very careful when using the default CandidatePtr functions (.mass, .mt, .et, etc). They may not be what you are looking for :-).
}

WMuNuCandidatePtr::~WMuNuCandidatePtr()
{
}

double WMuNuCandidatePtr::eT() const{
 double e_t=0; 
     e_t=muon_->pt()+neutrino_->pt();
 return e_t; 
}

double WMuNuCandidatePtr::massT() const{
      // CandidatePtrs have a mt() function which computes the tranverse mass from E & pz. 
      // As MET does not have pz information... WMuNuCandidatePtrs have an alternative function to compute the mt quantity
      // used in the WMuNu Inclusive analysis just from px, py
      double wpx=muon_->px()+neutrino_->px(); double wpy=muon_->py()+neutrino_->py();
      double mt = eT()*eT() - wpx*wpx - wpy*wpy;
      mt = (mt>0) ? sqrt(mt) : 0;
      return mt; 
}

double WMuNuCandidatePtr::acop() const{
      // Acoplanarity between the muon and the MET
       Geom::Phi<double> deltaphi(muon_->phi()-neutrino_->phi());
       double acop = deltaphi.value();
       if (acop<0) acop = - acop;
       acop = M_PI - acop;
       return acop;
}
