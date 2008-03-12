#include "PhysicsTools/PatAlgos/plugins/AnythingToValueMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat { namespace helper {
class AnyNumberAssociationAdaptor {
    public:
        typedef float                      value_type;
        typedef edm::View<reco::Candidate> Collection;

        AnyNumberAssociationAdaptor(const edm::InputTag &in, const edm::ParameterSet & iConfig) :
            type_(Uninitialized), in_(in), label_(in.label() + in.instance()) { }

        const std::string & label() { return label_; }
        
        bool run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
            switch (type_) {
                case Uninitialized: 
                     if (run_<edm::ValueMap<double> >(iEvent, coll, ret)) { type_ = ValueMapDouble; return true; }
                     if (run_<edm::ValueMap<float>  >(iEvent, coll, ret)) { type_ = ValueMapFloat;  return true; }
                     if (run_<edm::ValueMap<int>    >(iEvent, coll, ret)) { type_ = ValueMapInt;    return true; }
                     if (run_<edm::ValueMap<bool>   >(iEvent, coll, ret)) { type_ = ValueMapBool;   return true; }
                     if (run_<AssoVec<double>::type >(iEvent, coll, ret)) { type_ = AssoVecDouble;  return true; }
                     if (run_<AssoVec<float>::type  >(iEvent, coll, ret)) { type_ = AssoVecFloat;   return true; }
                     if (run_<AssoVec<int>::type    >(iEvent, coll, ret)) { type_ = AssoVecInt;     return true; }
                     if (run_<AssoVec<bool>::type   >(iEvent, coll, ret)) { type_ = AssoVecBool;    return true; }
                     type_ = Nothing; return false;
                     break;
                case ValueMapDouble : return run_<edm::ValueMap<double> >(iEvent, coll, ret);
                case ValueMapFloat  : return run_<edm::ValueMap<float>  >(iEvent, coll, ret);
                case ValueMapInt    : return run_<edm::ValueMap<int>    >(iEvent, coll, ret);
                case ValueMapBool   : return run_<edm::ValueMap<bool>   >(iEvent, coll, ret);
                case AssoVecDouble  : return run_<AssoVec<double>::type >(iEvent, coll, ret);
                case AssoVecFloat   : return run_<AssoVec<float>::type  >(iEvent, coll, ret);
                case AssoVecInt     : return run_<AssoVec<int>::type    >(iEvent, coll, ret);
                case AssoVecBool    : return run_<AssoVec<bool>::type   >(iEvent, coll, ret);
                case Nothing        : return false;
            }
            return false;
        } 
    private:
        enum Type { Uninitialized  = 0,
                    ValueMapDouble, ValueMapFloat, ValueMapInt, ValueMapBool,
                    AssoVecDouble ,  AssoVecFloat,  AssoVecInt,  AssoVecBool,
                    //AssoMapDouble ,  AssoMapFloat,  AssoMapInt,  AssoMapBool // TODO: obsolete, I hope we don't need it
                    Nothing };
        template<typename T> struct AssoVec { typedef typename edm::AssociationVector<reco::CandidateBaseRefProd, typename std::vector<T> > type; };
        template<typename T> bool   run_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
        Type type_;
        edm::InputTag in_;
        std::string label_;

};
    
template<typename T>
bool AnyNumberAssociationAdaptor::run_(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
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

typedef ManyThingsToValueMaps<AnyNumberAssociationAdaptor> MultipleNumbersToValueMaps;

}} // namespaces


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat::helper;
DEFINE_FWK_MODULE(MultipleNumbersToValueMaps);
