#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include <vector>

template<typename TIn, typename TCol, FlatTable::ColumnType CT>
class NativeArrayTableProducer : public edm::stream::EDProducer<> {
    public:

        NativeArrayTableProducer( edm::ParameterSet const & params ) :
            name_( params.getParameter<std::string>("name") ),
            doc_(params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""),
            src_(consumes<TIn>( params.getParameter<edm::InputTag>("src") )) 
        {
            produces<FlatTable>();
        }

        ~NativeArrayTableProducer() override {}

        void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
            edm::Handle<TIn> src;
            iEvent.getByToken(src_, src);

            const auto & in = *src;
            auto out = std::make_unique<FlatTable>(in.size(), name_, false, false);
            out->setDoc(doc_);
            (*out).template addColumn<TCol>(this->name_, in, this->doc_, CT);
            iEvent.put(std::move(out));
        }

    protected:
        const std::string name_; 
        const std::string doc_;
        const edm::EDGetTokenT<TIn> src_;
};

typedef NativeArrayTableProducer<std::vector<float>,float,FlatTable::FloatColumn> FloatArrayTableProducer;
typedef NativeArrayTableProducer<std::vector<double>,float,FlatTable::FloatColumn> DoubleArrayTableProducer;
typedef NativeArrayTableProducer<std::vector<int>,int,FlatTable::IntColumn> IntArrayTableProducer;
typedef NativeArrayTableProducer<std::vector<bool>,uint8_t,FlatTable::UInt8Column> BoolArrayTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FloatArrayTableProducer);
DEFINE_FWK_MODULE(DoubleArrayTableProducer);
DEFINE_FWK_MODULE(IntArrayTableProducer);
DEFINE_FWK_MODULE(BoolArrayTableProducer);

