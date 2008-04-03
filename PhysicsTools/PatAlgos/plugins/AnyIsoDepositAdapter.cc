#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageService/interface/MessageLogger.h"

namespace pat { namespace helper {
class AnyIsoDepositAdaptor {
    public:
        typedef reco::IsoDeposit          value_type;
        typedef edm::View<reco::Candidate> Collection;

        AnyIsoDepositAdaptor(const edm::InputTag &in, const edm::ParameterSet & iConfig) :
            type_(Uninitialized), tkType_(TkUninitialized),
            in_(in), label_(in.label() + in.instance()) { }

        const std::string & label() { return label_; }
        
        bool run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
            switch (type_) {
                case Uninitialized: 
                     if (run_<  edm::ValueMap<value_type>  >(iEvent, coll, ret)) { type_ = ValueMapC; return true; }
                     if (run_<  AssoVec<value_type>::type  >(iEvent, coll, ret)) { type_ = AssoVecC;  return true; }
                     if (runTk_<AssoMapTk<value_type>::type>(iEvent, coll, ret)) { type_ = AssoMapT;  return true; }
                     type_ = Nothing; return false;
                     break;
                case ValueMapC: return run_<  edm::ValueMap<value_type>  >(iEvent, coll, ret);
                case AssoVecC : return run_<  AssoVec<value_type>::type  >(iEvent, coll, ret);
                case AssoMapT : return runTk_<AssoMapTk<value_type>::type>(iEvent, coll, ret);
                case Nothing  : return false;
            }
            return false;
        } 

    private:
        enum Type { Uninitialized  = 0,
                    ValueMapC, AssoVecC, 
                    //AssoMapC, // not yet used
                    AssoMapT,
                    Nothing };
        enum TkKeyType { 
            TkUninitialized,
            TkGlobal, TkTrack, TkStandalone
        };
        template<typename T> struct AssoVec { typedef typename edm::AssociationVector<reco::CandidateBaseRefProd, typename std::vector<T> > type; };
        template<typename T> struct AssoMapTk { 
            typedef typename edm::AssociationMap<edm::OneToValue<reco::TrackCollection,T> > type;
        };
        template<typename T> bool   run_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
        template<typename T> bool   runTk_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
        inline const value_type *  tryTkRef_(const reco::TrackRef &ref, const AssoMapTk<value_type>::type &map) ;
        Type type_;
        TkKeyType tkType_;
        edm::InputTag in_;
        std::string label_;

};
    
template<typename T>
bool AnyIsoDepositAdaptor::run_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
    edm::Handle<T> handle;
    try { // 1_6_X: need try/catch
        iEvent.getByLabel(in_, handle);
        if (handle.failedToGet()) return false;
    } catch (cms::Exception &ex) { return false; }

    for (size_t i = 0, n = coll.size(); i < n; ++i) {
        reco::CandidateBaseRef ref = coll.refAt(i);
        ret.push_back( (*handle)[ref] );
    }
    return true;    
}

inline const AnyIsoDepositAdaptor::value_type *
AnyIsoDepositAdaptor::tryTkRef_(const reco::TrackRef &ref, 
                                const AnyIsoDepositAdaptor::AssoMapTk<AnyIsoDepositAdaptor::value_type>::type &map) 
{
    //std::cout << "Ref is " << (ref.isNull() ? "NULL" : "not null.") << std::endl;
    if (ref.isNull()) return 0;
    AssoMapTk<value_type>::type::const_iterator match = map.find(ref);
    //std::cout << "Key was " << ((match == map.end()) ? "NOT FOUND" : "FOUND") << std::endl;
    if (match == map.end()) return 0;
    return & match->val;
}

template<typename T>
bool AnyIsoDepositAdaptor::runTk_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
    edm::Handle<T> handle;
    try { // 1_6_X: need try/catch
        iEvent.getByLabel(in_, handle);
        if (handle.failedToGet()) return false;
    } catch (cms::Exception &ex) { return false; }
    if (handle->empty()) return true; // can't datamine track type if the map is empty!
    //std::cout << "MAP " << in_.encode() << " SIZE IS " << handle->size() << std::endl;
    for (size_t i = 0, n = coll.size(); i < n; ++i) {
        const reco::RecoCandidate *rc = dynamic_cast<const reco::RecoCandidate *>(&coll[i]);
        if (rc == 0) {
            //std::cout << "Reading AssoMap<Track,IsoDeposit> '" << in_.encode() << "' but item is not a RecoCandidate." << std::endl;
            typename edm::LogWarning("AnyIsoDepositAdaptor") << 
                        "Reading AssoMap<Track,IsoDeposit> '" << in_.encode() << "' but item is not a RecoCandidate.";
            return false;
        }
        const value_type *val = 0;
        switch (tkType_) {
            case TkUninitialized:
                //std::cout << "Trying to autodetect the ref key" << std::endl;
                if ((val = tryTkRef_(rc->combinedMuon(), *handle)) != 0) {
                    //std::cout << "Global" << std::endl;
                    tkType_ = TkGlobal; 
                } else if ((val = tryTkRef_(rc->track(), *handle)) != 0) {
                    //std::cout << "Tracker" << std::endl;
                    tkType_ = TkTrack; 
                } else if ((val = tryTkRef_(rc->standAloneMuon(), *handle)) != 0) {
                    //std::cout << "Muon" << std::endl;
                    tkType_ = TkStandalone; 
                } 
                //if (val == 0) std::cout << "NOTHING??????" << std::endl;
                break;
            case TkGlobal:     val = tryTkRef_(rc->combinedMuon(),   *handle); break;
            case TkTrack:      val = tryTkRef_(rc->track(),          *handle); break;
            case TkStandalone: val = tryTkRef_(rc->standAloneMuon(), *handle); break;
        } 
        if (val != 0) {
            ret.push_back( *val );
        } else {
            //std::cout << "Reading AssoMap<Track,IsoDeposit> '" << in_.encode() << "'" << 
            //       " but I can't get a valid track ref on which this map knows anything about." << std::endl; 
            typename edm::LogWarning("AnyIsoDepositAdaptor") <<
                "Reading AssoMap<Track,IsoDeposit> '" << in_.encode() << "'" << 
                " but I can't get a valid track ref on which this map knows anything about.";
            return false;
        }
    }
    return true;    
}


typedef ManyThingsToValueMaps<AnyIsoDepositAdaptor> MultipleIsoDepositsToValueMaps;

}} // namespaces


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat::helper;
DEFINE_FWK_MODULE(MultipleIsoDepositsToValueMaps);
