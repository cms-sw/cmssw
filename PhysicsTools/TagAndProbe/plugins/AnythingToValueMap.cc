#include "PhysicsTools/TagAndProbe/plugins/AnythingToValueMap.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace pat { namespace helper {
class AnyNumberAssociationAdaptor {
    public:
        typedef float                      value_type;
        typedef edm::View<reco::Candidate> Collection;
        template<typename T> struct AssoVec { typedef typename edm::AssociationVector<reco::CandidateBaseRefProd, typename std::vector<T> > type; };

        AnyNumberAssociationAdaptor(const edm::InputTag &in, const edm::ParameterSet & iConfig, edm::ConsumesCollector && iC) :
            type_(Uninitialized), in_(in), label_(in.label() + in.instance()),
            tokenVMd_(iC.consumes<edm::ValueMap<double> >(in)),
            tokenVMf_(iC.consumes<edm::ValueMap<float>  >(in)),
            tokenVMi_(iC.consumes<edm::ValueMap<int>    >(in)),
            tokenVMb_(iC.consumes<edm::ValueMap<bool>   >(in)),
            tokenAVd_(iC.consumes<AssoVec<double>::type >(in)),
            tokenAVf_(iC.consumes<AssoVec<float>::type  >(in)),
            tokenAVi_(iC.consumes<AssoVec<int>::type    >(in)),
            tokenAVb_(iC.consumes<AssoVec<bool>::type   >(in))            
            { }

        const std::string & label() { return label_; }
        
        bool run(const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
            switch (type_) {
                case Uninitialized: 
                     if (run_<edm::ValueMap<double> >(tokenVMd_, iEvent, coll, ret)) { type_ = ValueMapDouble; return true; }
                     if (run_<edm::ValueMap<float>  >(tokenVMf_, iEvent, coll, ret)) { type_ = ValueMapFloat;  return true; }
                     if (run_<edm::ValueMap<int>    >(tokenVMi_, iEvent, coll, ret)) { type_ = ValueMapInt;    return true; }
                     if (run_<edm::ValueMap<bool>   >(tokenVMb_, iEvent, coll, ret)) { type_ = ValueMapBool;   return true; }
                     if (run_<AssoVec<double>::type >(tokenAVd_, iEvent, coll, ret)) { type_ = AssoVecDouble;  return true; }
                     if (run_<AssoVec<float>::type  >(tokenAVf_, iEvent, coll, ret)) { type_ = AssoVecFloat;   return true; }
                     if (run_<AssoVec<int>::type    >(tokenAVi_, iEvent, coll, ret)) { type_ = AssoVecInt;     return true; }
                     if (run_<AssoVec<bool>::type   >(tokenAVb_, iEvent, coll, ret)) { type_ = AssoVecBool;    return true; }
                     type_ = Nothing; return false;
                     break;
                case ValueMapDouble : return run_<edm::ValueMap<double> >(tokenVMd_, iEvent, coll, ret);
                case ValueMapFloat  : return run_<edm::ValueMap<float>  >(tokenVMf_, iEvent, coll, ret);
                case ValueMapInt    : return run_<edm::ValueMap<int>    >(tokenVMi_, iEvent, coll, ret);
                case ValueMapBool   : return run_<edm::ValueMap<bool>   >(tokenVMb_, iEvent, coll, ret);
                case AssoVecDouble  : return run_<AssoVec<double>::type >(tokenAVd_, iEvent, coll, ret);
                case AssoVecFloat   : return run_<AssoVec<float>::type  >(tokenAVf_, iEvent, coll, ret);
                case AssoVecInt     : return run_<AssoVec<int>::type    >(tokenAVi_, iEvent, coll, ret);
                case AssoVecBool    : return run_<AssoVec<bool>::type   >(tokenAVb_, iEvent, coll, ret);
                case Nothing        : return false;
            }
            return false;
        } 
    private:
        enum Type { Uninitialized  = 0,
                    ValueMapDouble, ValueMapFloat, ValueMapInt, ValueMapBool,
                    AssoVecDouble ,  AssoVecFloat,  AssoVecInt,  AssoVecBool,
                    Nothing };
        template<typename T> bool   run_(const edm::EDGetTokenT<T> & token, const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) ;
        Type type_;
        edm::InputTag in_;
        std::string label_;
        edm::EDGetTokenT<edm::ValueMap<double> > tokenVMd_;
        edm::EDGetTokenT<edm::ValueMap<float>  > tokenVMf_;
        edm::EDGetTokenT<edm::ValueMap<int>    > tokenVMi_;
        edm::EDGetTokenT<edm::ValueMap<bool>   > tokenVMb_;
        edm::EDGetTokenT<AssoVec<double>::type > tokenAVd_;
        edm::EDGetTokenT<AssoVec<float>::type  > tokenAVf_;
        edm::EDGetTokenT<AssoVec<int>::type    > tokenAVi_;
        edm::EDGetTokenT<AssoVec<bool>::type   > tokenAVb_;

};
    
template<typename T>
bool AnyNumberAssociationAdaptor::run_(const edm::EDGetTokenT<T> & token, const edm::Event &iEvent, const Collection &coll, std::vector<value_type> &ret) {
    edm::Handle<T> handle;
    iEvent.getByToken(token, handle);
    if (handle.failedToGet()) return false;

    for (size_t i = 0, n = coll.size(); i < n; ++i) {
        reco::CandidateBaseRef ref = coll.refAt(i);
        ret.push_back( (*handle)[ref] );
    }
    return true;    
}

typedef ManyThingsToValueMaps<AnyNumberAssociationAdaptor> AnyNumbersToValueMaps;

}} // namespaces


#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat::helper;
DEFINE_FWK_MODULE(AnyNumbersToValueMaps);
