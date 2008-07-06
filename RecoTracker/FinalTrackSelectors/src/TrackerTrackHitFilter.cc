#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"


#include <boost/regex.hpp>

/**
 * Configurables:
 *
 *   Generic: 
 *     tracks                  = InputTag of a collection of tracks to read from
 *     minimumHits             = Minimum hits that the output TrackCandidate must have to be saved
 *     replaceWithInactiveHits = instead of discarding hits, replace them with a invalid "inactive" hits, 
 *                               so multiple scattering is accounted for correctly.
 *     stripFrontInvalidHits   = strip invalid hits at the beginning of the track
 *     stripBackInvalidHits    = strip invalid hits at the end of the track
 *     stripAllInvalidHits     = remove ALL invald hits (might be a problem for multiple scattering, use with care!)
 *
 *   Per structure: 
 *      commands = list of commands, to be applied in order as they are written
 *      commands can be:
 *          "keep XYZ"  , "drop XYZ"    (XYZ = PXB, PXE, TIB, TID, TOB, TEC)
 *          "keep XYZ l", "drop XYZ n"  (XYZ as above, n = layer, wheel or disk = 1 .. 6 ; positive and negative are the same )
 *
 *   Individual modules:
 *     detsToIgnore        = individual list of detids on which hits must be discarded
 */
namespace reco { namespace modules {
class TrackerTrackHitFilter : public edm::EDProducer {
    public:
       TrackerTrackHitFilter(const edm::ParameterSet &iConfig) ; 
       virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

    private:
       class Rule {
            public:
                // parse a rule from a string
                Rule(const std::string &str) ;
                // check this DetId, update the verdict if the rule has anything to say about it
                void apply(DetId detid, bool & verdict) const {
                    // check detector
                    if (detid.subdetId() == subdet_) {
                        // check layer
                        if ( (layer_ == 0) || (layer_ == layer(detid)) ) {
                            // override verdict
                            verdict = keep_;
                        }
                    }
                }
            private:
                int  subdet_;
                int  layer_;
                bool keep_;
                int layer(DetId detid) const ;
       };
    
       edm::InputTag tracks_;
    
       size_t minimumHits_;

       bool replaceWithInactiveHits_;
       bool stripFrontInvalidHits_;
       bool stripBackInvalidHits_;
       bool stripAllInvalidHits_;

       std::vector<uint32_t> detsToIgnore_;
       std::vector<Rule> rules_;

       edm::ESHandle<TrackerGeometry> theGeometry;
       edm::ESHandle<MagneticField>   theMagField;

       TrackCandidate makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd) ;
       
}; // class



TrackerTrackHitFilter::Rule::Rule(const std::string &str) {
    static boost::regex rule("(keep|drop)\\s+([A-Z]+)(\\s+(\\d+))?");
    boost::cmatch match;
    // match and check it works
    if (!regex_match(str.c_str(), match, rule)) {
        throw cms::Exception("Configuration") << "Rule '" << str << "' not understood.\n";
    }
    // Set up fields:
    //  rule type
    keep_  = (match[1].first == "keep");
    //  subdet
    subdet_ = -1;
    if      (strncmp(match[2].first, "PXB", 3) == 0) subdet_ = PixelSubdetector::PixelBarrel;
    else if (strncmp(match[2].first, "PXE", 3) == 0) subdet_ = PixelSubdetector::PixelEndcap;
    else if (strncmp(match[2].first, "TIB", 3) == 0) subdet_ = StripSubdetector::TIB;
    else if (strncmp(match[2].first, "TID", 3) == 0) subdet_ = StripSubdetector::TID;
    else if (strncmp(match[2].first, "TOB", 3) == 0) subdet_ = StripSubdetector::TOB;
    else if (strncmp(match[2].first, "TEC", 3) == 0) subdet_ = StripSubdetector::TEC;
    if (subdet_ == -1) {
        throw cms::Exception("Configuration") << "Detector '" << match[2].first << "' not understood. Should be PXB, PXE, TIB, TID, TOB, TEC.\n";
    }
    //   layer (if present)
    if (match[4].first != match[4].second) {
        layer_ = atoi(match[4].first);        
    } else {
        layer_ = 0;
    }
}

int TrackerTrackHitFilter::Rule::layer(DetId detid) const {
    switch (detid.subdetId()) {
        case PixelSubdetector::PixelBarrel: return PXBDetId(detid).layer();
        case PixelSubdetector::PixelEndcap: return PXFDetId(detid).disk();
        case StripSubdetector::TIB:         return TIBDetId(detid).layer();
        case StripSubdetector::TID:         return TIDDetId(detid).wheel();
        case StripSubdetector::TOB:         return TOBDetId(detid).layer();
        case StripSubdetector::TEC:         return TECDetId(detid).wheel();
    }
    return -1; // never match
}


TrackerTrackHitFilter::TrackerTrackHitFilter(const edm::ParameterSet &iConfig) :
    tracks_(iConfig.getParameter<edm::InputTag>("tracks")),
    minimumHits_(iConfig.getParameter<uint32_t>("minimumHits")),
    replaceWithInactiveHits_(iConfig.getParameter<bool>("replaceWithInactiveHits")),
    stripFrontInvalidHits_(iConfig.getParameter<bool>("stripFrontInvalidHits")),
    stripBackInvalidHits_( iConfig.getParameter<bool>("stripBackInvalidHits") ),
    stripAllInvalidHits_(  iConfig.getParameter<bool>("stripAllInvalidHits")  ),
    detsToIgnore_( iConfig.getParameter<std::vector<uint32_t> >("detsToIgnore") )
{
    // sanity check 
    if (stripAllInvalidHits_ && replaceWithInactiveHits_) {
        throw cms::Exception("Configuration") << "Inconsistent Configuration: you can't set both 'stripAllInvalidHits' and 'replaceWithInactiveHits' to true\n";
    }

    // read and parse commands
    std::vector<std::string> str_rules = iConfig.getParameter<std::vector<std::string> >("commands");
    rules_.reserve(str_rules.size());
    for (std::vector<std::string>::const_iterator it = str_rules.begin(), ed = str_rules.end(); it != ed; ++it) {
        rules_.push_back(Rule(*it));
    }
    // sort detids to ignore
    std::sort(detsToIgnore_.begin(), detsToIgnore_.end());

    // issue the produce<>
    produces<TrackCandidateCollection>();
}

void
TrackerTrackHitFilter::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) 
{
    // read with View, so we can read also a TrackRefVector
    edm::Handle<std::vector<reco::Track> > tracks;
    iEvent.getByLabel(tracks_, tracks);

    // read from EventSetup
    iSetup.get<TrackerDigiGeometryRecord>().get(theGeometry);
    iSetup.get<IdealMagneticFieldRecord>().get(theMagField);

    // prepare output collection
    std::auto_ptr<TrackCandidateCollection> output(new TrackCandidateCollection());
    output->reserve(tracks->size());
    
    // working area and tools
    std::vector<TrackingRecHit *> hits;

    //std::cout << "NewTrackHitFilter: loop on tracks" << std::endl;
    // loop on tracks
    for (std::vector<reco::Track>::const_iterator itt = tracks->begin(), edt = tracks->end(); itt != edt; ++itt) {
        hits.clear(); // extra safety
        //std::cout << "   loop on hits of track #" << (itt - tracks->begin()) << std::endl;
        for (trackingRecHit_iterator ith = itt->recHitsBegin(), edh = itt->recHitsEnd(); ith != edh; ++ith) {
            const TrackingRecHit * hit = ith->get(); // ith is an iterator on edm::Ref to rechit
            //std::cout << "         hit number " << (ith - itt->recHitsBegin()) << std::endl;
            // let's look at valid hits
            if (hit->isValid()) { 
                //std::cout << "            valid, detid = " << hit->geographicalId().rawId() << std::endl;
                DetId detid = hit->geographicalId();
                if (detid.det() == DetId::Tracker) {  // check for tracker hits
                    //std::cout << "            valid, tracker " << std::endl;
                    bool  verdict = true;
                    // first check at structure level
                    for (std::vector<Rule>::const_iterator itr = rules_.begin(), edr = rules_.end(); itr != edr; ++itr) {
                        itr->apply(detid, verdict);
                    }
                    //std::cout << "            valid, verdict after rules is: " << (verdict ? "ok" : "no") << std::endl;
                    // if the hit is good, check again at module level
                    if ( verdict  && std::binary_search(detsToIgnore_.begin(), detsToIgnore_.end(), detid.rawId())) {
                        verdict = false;
                    }
                    
                    //std::cout << "                   verdict after module list: " << (verdict ? "ok" : "no") << std::endl;
                    if (verdict == true) {
                        // just copy the hit
                         hits.push_back(hit->clone());
                    } else {
                        // still, if replaceWithInactiveHits is true we have to put a new hit
                        if (replaceWithInactiveHits_) {
                            hits.push_back(new InvalidTrackingRecHit(detid, TrackingRecHit::inactive));
                        } 
                    }
                } else { // just copy non tracker hits
                    hits.push_back(hit->clone());
                }
            } else {
                if (!stripAllInvalidHits_) {
                    hits.push_back(hit->clone());
                } 
            } // is valid hit
            //std::cout << "         end of hit " << (ith - itt->recHitsBegin()) << std::endl;
        } // loop on hits
        //std::cout << "   end of loop on hits of track #" << (itt - tracks->begin()) << std::endl;

        std::vector<TrackingRecHit *>::iterator begin = hits.begin(), end = hits.end();

        //std::cout << "   selected " << hits.size() << " hits " << std::endl;

        // strip invalid hits at the beginning
        if (stripFrontInvalidHits_) {
            while ( (begin != end) && ( (*begin)->isValid() == false ) ) ++begin;
        }

        //std::cout << "   after front stripping we have " << (end - begin) << " hits " << std::endl;

        // strip invalid hits at the end
        if (stripBackInvalidHits_ && (begin != end)) {
            --end;
            while ( (begin != end) && ( (*end)->isValid() == false ) ) --end;
            ++end;
        }

        //std::cout << "   after back stripping we have " << (end - begin) << " hits " << std::endl;

        // if we still have some hits
        if ((end - begin) >= int(minimumHits_)) {
            output->push_back( makeCandidate ( *itt, begin, end ) );
        } 
        // now delete the hits not used by the candidate
        for (begin = hits.begin(), end = hits.end(); begin != end; ++begin) {
            if (*begin) delete *begin;
        } 
        hits.clear();
    } // loop on tracks

    iEvent.put(output);
}

TrackCandidate
TrackerTrackHitFilter::makeCandidate(const reco::Track &tk, std::vector<TrackingRecHit *>::iterator hitsBegin, std::vector<TrackingRecHit *>::iterator hitsEnd) {
    TrajectoryStateTransform transform;
    PropagationDirection   pdir = tk.seedDirection();
    PTrajectoryStateOnDet *state;
    if ( pdir == anyDirection ) throw cms::Exception("UnimplementedFeature") << "Cannot work with tracks that have 'anyDirecton' \n";
    if ( (pdir == alongMomentum) == ( tk.p() >= tk.outerP() ) ) {
        // use inner state
        TrajectoryStateOnSurface originalTsosIn(transform.innerStateOnSurface(tk, *theGeometry, &*theMagField));
        state = transform.persistentState( originalTsosIn, DetId(tk.innerDetId()) );
    } else { 
        // use outer state
        TrajectoryStateOnSurface originalTsosOut(transform.outerStateOnSurface(tk, *theGeometry, &*theMagField));
        state = transform.persistentState( originalTsosOut, DetId(tk.outerDetId()) );
    }

    TrajectorySeed seed(*state, TrackCandidate::RecHitContainer(), pdir);
 
    TrackCandidate::RecHitContainer ownHits;
    ownHits.reserve(hitsEnd - hitsBegin);
    for ( ; hitsBegin != hitsEnd; ++hitsBegin) { ownHits.push_back( *hitsBegin ); }

    TrackCandidate cand(ownHits, seed, *state, tk.seedRef());
    delete state;

    //std::cout << "   dumping the hits now: " << std::endl;
    //for (TrackCandidate::range hitR = cand.recHits(); hitR.first != hitR.second; ++hitR.first) {
    //    std::cout << "     hit detid = " << hitR.first->geographicalId().rawId() <<
    //        ", type  = " << typeid(*hitR.first).name() << std::endl;
    //}

    return cand;
}

}} //namespaces


// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using reco::modules::TrackerTrackHitFilter;
DEFINE_FWK_MODULE(TrackerTrackHitFilter);
