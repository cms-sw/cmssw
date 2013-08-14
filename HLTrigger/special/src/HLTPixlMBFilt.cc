/** \class HLTPixlMBFilt
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 15:00:22 $
 *  $Revision: 1.4 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTrigger/special/interface/HLTPixlMBFilt.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
 
HLTPixlMBFilt::HLTPixlMBFilt(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    pixlTag_ (iConfig.getParameter<edm::InputTag>("pixlTag")),
    min_Pt_  (iConfig.getParameter<double>("MinPt")),
    min_trks_  (iConfig.getParameter<unsigned int>("MinTrks")),
    min_sep_  (iConfig.getParameter<double>("MinSep"))

{
  LogDebug("") << "MinPt cut " << min_Pt_   << "pixl: " << pixlTag_.encode();
  LogDebug("") << "Requesting : " << min_trks_ << " tracks from same vertex ";
  LogDebug("") << "Requesting tracks from same vertex eta-phi separation by " << min_sep_;
}

HLTPixlMBFilt::~HLTPixlMBFilt()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixlMBFilt::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.



   // Specific filter code

   // get hold of products from Event

   Handle<RecoChargedCandidateCollection> tracks;

   iEvent.getByLabel(pixlTag_,tracks);

   // pixel tracks
   int npixl_tot = 0;
   vector<double> etastore;
   vector<double> phistore;
   vector<int> itstore;
   bool accept;
   RecoChargedCandidateCollection::const_iterator apixl(tracks->begin());
   RecoChargedCandidateCollection::const_iterator epixl(tracks->end());
   RecoChargedCandidateCollection::const_iterator ipixl, jpixl;
   unsigned int nsame_vtx=0;
   int itrk = -1;
   if (tracks->size() >= min_trks_) {
     for (ipixl=apixl; ipixl!=epixl; ipixl++){ 
       itrk++;
       const double& ztrk1 = ipixl->vz();		    
       const double& etatrk1 = ipixl->momentum().eta();
       const double& phitrk1 = ipixl->momentum().phi();
       nsame_vtx=1;
       etastore.clear();
       phistore.clear();
       itstore.clear();
       etastore.push_back(etatrk1);
       phistore.push_back(phitrk1);
       itstore.push_back(itrk);
       if (fabs(ztrk1) < 15.0) {
         //  check this track against all others to see if others start from same point
	 int jtrk=-1;
         for (jpixl=apixl; jpixl!=epixl; jpixl++) {
	   jtrk++;
	   if (jpixl==ipixl) continue;
           const double& ztrk2 = jpixl->vz();		    
           const double& etatrk2 = jpixl->momentum().eta();
           const double& phitrk2 = jpixl->momentum().phi();
           double eta_dist=etatrk2-etatrk1;
           double phi_dist=phitrk2-phitrk1;
           double etaphi_dist=sqrt(eta_dist*eta_dist + phi_dist*phi_dist);
           if (fabs(ztrk2-ztrk1) < 1.0 && etaphi_dist > min_sep_) {
	      if (min_trks_ <= 2 || itstore.size() <= 1) {
	        etastore.push_back(etatrk2);
	        phistore.push_back(phitrk2);
		itstore.push_back(jtrk);
                nsame_vtx++;
              } else {
                // check also separation to already found 'second' tracks
		LogDebug("") << "HLTPixlMBFilt: with mintrks=2 we should not be here...";
		bool isok = true;
		for (unsigned int k=1; k < itstore.size(); k++) {
                  eta_dist=etatrk2-etastore.at(k);
                  phi_dist=phitrk2-phistore.at(k);
                  etaphi_dist=sqrt(eta_dist*eta_dist + phi_dist*phi_dist);
		  if (etaphi_dist < min_sep_) {
		    isok=false;
		    break;
                  }
		}
		if (isok) {
	          etastore.push_back(etatrk2);
	          phistore.push_back(phitrk2);
                  itstore.push_back(jtrk);
                  nsame_vtx++;
                }
	      }
	   }
           if (nsame_vtx >= min_trks_) break;
         }
       }
       npixl_tot++;

       if (nsame_vtx >= min_trks_) break;
     }

     //   final filter decision:
     //   request at least min_trks_ tracks compatible with vertex-region
     accept = (nsame_vtx >= min_trks_ ) ;

   } else {
     accept = false;
   }

   // At this point we have the indices of the accepted tracks stored in itstore
   // we now move them to the filterproduct

   if (accept) {
     for (unsigned int ipos=0; ipos < itstore.size(); ipos++) {
       int iaddr=itstore.at(ipos);
       filterproduct.addObject(TriggerTrack,RecoChargedCandidateRef(tracks,iaddr));
     }
   }

  LogDebug("") << "Number of pixel-track objects accepted:"
               << " " << npixl_tot;

  // return with final filter decision
  return accept;

}
