#include "AnalysisDataFormats/EWK/interface/WMuNuCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace edm;
using namespace std;
using namespace reco;

WMuNuCandidate::WMuNuCandidate(){}

WMuNuCandidate::WMuNuCandidate(edm::Ptr<reco::Muon> muon, edm::Ptr<reco::MET> met): muon_(muon), neutrino_(met)  
{
      addDaughter(*muon,"Muon");
      addDaughter(*met,"Met");
      AddFourMomenta addP4;
      addP4.set(*this);

      //WARNING: W Candidates combine the information from a Muon with the (px,py) information of the MET as the Neutrino
      // --> There is no Pz information!!!!
      // Be very careful when using the default Candidate functions (.mass, .mt, .et, etc). They may not be what you are looking for :-).
}

WMuNuCandidate::~WMuNuCandidate()
{
}

double WMuNuCandidate::eT() const{
 double e_t=0; 
     e_t=muon_->pt()+neutrino_->pt();
 return e_t; 
}

double WMuNuCandidate::massT() const{
      // Candidates have a mt() function which computes the tranverse mass from E & pz. 
      // As MET does not have pz information... WMuNuCandidates have an alternative function to compute the mt quantity
      // used in the WMuNu Inclusive analysis just from px, py
      double wpx=px(); double wpy=py();
      double mt = eT()*eT() - wpx*wpx - wpy*wpy;
      mt = (mt>0) ? sqrt(mt) : 0;
      return mt; 
}

double WMuNuCandidate::acop() const{
      // Acoplanarity between the muon and the MET
       Geom::Phi<double> deltaphi(daughter(0)->phi()-daughter(1)->phi());
       double acop = deltaphi.value();
       if (acop<0) acop = - acop;
       acop = M_PI - acop;
       return acop;
}
