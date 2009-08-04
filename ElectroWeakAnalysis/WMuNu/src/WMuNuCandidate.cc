#include "ElectroWeakAnalysis/WMuNu/interface/WMuNuCandidate.h"
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

double WMuNuCandidate::eT(bool useTrackPt) const{
 double e_t=0; 
 if(useTrackPt) {e_t=muon_->innerTrack()->pt()+neutrino_->pt();}  
 else {
     e_t=muon_->pt()+neutrino_->pt();}
 return e_t; 
}

double WMuNuCandidate::massT(bool useTrackPt) const{
      // Candidates have a mt() function which computes the tranverse mass from E & pz. 
      // As MET does not have pz information... WMuNuCandidates have an alternative function to compute the mt quantity
      // used in the WMuNu Inclusive analysis just from px, py
 
      double wpx=0, wpy=0;
      //"UseTrackPt" uses the tracker momentum instead of the global momentum.  
       if (useTrackPt){
          wpx=muon_->innerTrack()->px()+neutrino_->px(); 
          wpy=muon_->innerTrack()->py()+neutrino_->py();
       } else {wpx=px(); wpy=py();}

       double mt = eT(useTrackPt)*eT(useTrackPt) - wpx*wpx - wpy*wpy;
 
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
