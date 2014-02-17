/** \class HLTPixlMBForAlignmentFilter
 *
 * See header file for documentation
 *
 *  $Date: 2012/01/21 15:00:22 $
 *  $Revision: 1.5 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTrigger/special/interface/HLTPixlMBForAlignmentFilter.h"

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
 
HLTPixlMBForAlignmentFilter::HLTPixlMBForAlignmentFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
    pixlTag_ (iConfig.getParameter<edm::InputTag>("pixlTag")),
    min_Pt_  (iConfig.getParameter<double>("MinPt")),
    min_trks_  (iConfig.getParameter<unsigned int>("MinTrks")),
    min_sep_  (iConfig.getParameter<double>("MinSep")),
    min_isol_  (iConfig.getParameter<double>("MinIsol"))

{
  LogDebug("") << "MinPt cut " << min_Pt_   << "pixl: " << pixlTag_.encode();
  LogDebug("") << "Requesting : " << min_trks_ << " tracks from same vertex ";
  LogDebug("") << "Requesting tracks from same vertex eta-phi separation by " << min_sep_;
  LogDebug("") << "Requesting track to be isolated within cone of " << min_isol_;
}

HLTPixlMBForAlignmentFilter::~HLTPixlMBForAlignmentFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixlMBForAlignmentFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
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
   vector<double> etastore;
   vector<double> phistore;
   vector<double> ptstore;
   vector<int> itstore;
   bool accept = false;
   RecoChargedCandidateCollection::const_iterator apixl(tracks->begin());
   RecoChargedCandidateCollection::const_iterator epixl(tracks->end());
   RecoChargedCandidateCollection::const_iterator ipixl, jpixl;
   int itrk = 0;
   double zvtxfit = 0.0;
   double zvtxfit2 = 0.0;
   if (tracks->size() >= min_trks_) {
     etastore.clear();
     phistore.clear();
     itstore.clear();
     for (ipixl=apixl; ipixl!=epixl; ipixl++){ 
       const double& ztrk1 = ipixl->vz();                    
       const double& etatrk1 = ipixl->momentum().eta();
       const double& phitrk1 = ipixl->momentum().phi();
       const double& pttrk1 = ipixl->pt();
       zvtxfit  = zvtxfit  + ztrk1;
       zvtxfit2 = zvtxfit2 + ztrk1 * ztrk1;
       if (pttrk1 > min_Pt_) {
//       the *store-vectors store the tracks above pt-cut
//       itstore is the position in the original collection
         etastore.push_back(etatrk1);
         phistore.push_back(phitrk1);
         ptstore.push_back(pttrk1);
         itstore.push_back(itrk);
       }
       itrk++;
     }
     if (itrk > 0) {
//     implement proper vertex fit here ?
       zvtxfit  = zvtxfit  / itrk;
       zvtxfit2 = zvtxfit2 / itrk;
       zvtxfit2 = sqrt(zvtxfit2 - zvtxfit*zvtxfit);
     }
//   locisol is the position in the *store vectors
     vector<int> locisol;
     if (itstore.size() > 1) {
       // now check that tracks are isolated
       locisol.clear();
       for (unsigned int i=0; i<itstore.size(); i++) {
         int nincone=0;
//       check isolation wrt ALL tracks, not only those above ptcut
         for (ipixl=apixl; ipixl!=epixl; ipixl++){ 
           double phidist=std::abs( phistore.at(i) - ipixl->momentum().phi() );
           double etadist=std::abs( etastore.at(i) - ipixl->momentum().eta() );
           double trkdist = sqrt(phidist*phidist + etadist*etadist);
           if (trkdist < min_isol_) nincone++;
         }
//       the check above always find the track itself, so nincone never should be 0
         if (nincone < 2) locisol.push_back(i);
       }
     }
//   now check that the selected tracks have enough mutual separation
     vector<int> itsep;
     for (unsigned int i=0; i<locisol.size(); i++) {
//     check for each so far selected track...
       itsep.clear();
       itsep.push_back(locisol.at(i));
       for (unsigned int j=i+1; j<locisol.size(); j++) {
//       ...if it is sufficiently separated from other selectad tracks...
         double phidist = phistore.at(locisol.at(i))-phistore.at(locisol.at(j));
         double etadist = etastore.at(locisol.at(i))-etastore.at(locisol.at(j));
         double dist = sqrt(phidist*phidist + etadist*etadist);
         if (dist > min_sep_) {
           if (itsep.size() == 1) {
             itsep.push_back(locisol.at(j));
           } else {
             bool is_separated = true;
             for (unsigned int k=0; k<itsep.size(); k++){
//             ...and the other ones, that are on the 'final acceptance' list already, if min_trks_ > 2
               double phisep = phistore.at(itsep.at(k))-phistore.at(locisol.at(j));
               double etasep = etastore.at(itsep.at(k))-etastore.at(locisol.at(j));
               double sep = sqrt(phisep*phisep + etasep*etasep);
               if (sep < min_sep_) {
//               this one was no good, too close to some other already accepted
                 is_separated = false;
                 break;
               }
             }
             if (is_separated) itsep.push_back(locisol.at(j));
           }
         }
         if (itsep.size() >= min_trks_) { 
           accept = true;
           break;
         }
       }
       if (accept) {
         break;
       }
     }
     // At this point we have the indices of the accepted tracks stored in itstore
     // we now move them to the filterproduct

     if (accept) {
       for (unsigned int ipos=0; ipos < itsep.size(); ipos++) {
         int iaddr=itstore.at(itsep.at(ipos));
         filterproduct.addObject(TriggerTrack,RecoChargedCandidateRef(tracks,iaddr));
       }
       // std::cout << "Accept this event " << std::endl;
     }
   }


//  LogDebug("") << "Number of pixel-track objects accepted:"
//               << " " << npixl_tot;

   // return with final filter decision
   return accept;

}
