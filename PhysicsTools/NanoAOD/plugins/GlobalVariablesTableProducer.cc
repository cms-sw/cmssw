#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <utility>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>

class GlobalVariablesTableProducer : public edm::stream::EDProducer<> {
    public:

        GlobalVariablesTableProducer( edm::ParameterSet const & params )
        {
            edm::ParameterSet const & varsPSet = params.getParameter<edm::ParameterSet>("variables");
            for (const std::string & vname : varsPSet.getParameterNamesForType<edm::ParameterSet>()) {
                const auto & varPSet = varsPSet.getParameter<edm::ParameterSet>(vname);
                const std::string & type = varPSet.getParameter<std::string>("type");
                if (type == "int") vars_.push_back(new IntVar(vname, nanoaod::FlatTable::IntColumn, varPSet, consumesCollector()));
                else if (type == "float") vars_.push_back(new FloatVar(vname, nanoaod::FlatTable::FloatColumn, varPSet, consumesCollector()));
                else if (type == "double") vars_.push_back(new DoubleVar(vname, nanoaod::FlatTable::FloatColumn, varPSet, consumesCollector()));
                else if (type == "bool") vars_.push_back(new BoolVar(vname, nanoaod::FlatTable::UInt8Column, varPSet, consumesCollector()));
                else if (type == "candidatescalarsum") vars_.push_back(new CandidateScalarSumVar(vname, nanoaod::FlatTable::FloatColumn, varPSet, consumesCollector()));
                else if (type == "candidatesize") vars_.push_back(new CandidateSizeVar(vname, nanoaod::FlatTable::IntColumn, varPSet, consumesCollector()));
		else if (type == "candidatesummass") vars_.push_back(new CandidateSumMassVar(vname, nanoaod::FlatTable::FloatColumn, varPSet, consumesCollector()));
                else throw cms::Exception("Configuration", "unsupported type "+type+" for variable "+vname);
            }

            produces<nanoaod::FlatTable>();
        }

        ~GlobalVariablesTableProducer() override {}

        void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
            auto out = std::make_unique<nanoaod::FlatTable>(1, "", true);

            for (const auto & var : vars_) var.fill(iEvent, *out);

            iEvent.put(std::move(out));
        }

    protected:
        class Variable {
            public:
                Variable(const std::string & aname, nanoaod::FlatTable::ColumnType atype, const edm::ParameterSet & cfg) : 
                    name_(aname), doc_(cfg.getParameter<std::string>("doc")), type_(atype) {}
                virtual void fill(const edm::Event &iEvent, nanoaod::FlatTable & out) const = 0;
                virtual ~Variable() {}
                const std::string & name() const { return name_; }
                const nanoaod::FlatTable::ColumnType & type() const { return type_; }
            protected:
                std::string name_, doc_;
                nanoaod::FlatTable::ColumnType type_;
        };
	template <typename ValType>
	class Identity {
		public:
			static ValType convert(ValType x){return x;}
			
	};
	template <typename ValType>
	class Size {
		public:
			static int convert(ValType x){return x.size();}
			
	};

	template <typename ColType,typename ValType>
	class Max {
		public:
			static ColType convert(ValType x){
				ColType v=std::numeric_limits<ColType>::min();
				for(const auto & i : x) if(i>v) v=i;
				return v;
			}
        };
	template <typename ColType,typename ValType>
        class Min {
                public:
                        static ColType convert(ValType x){
                                ColType v=std::numeric_limits<ColType>::max(); 
                                for(const auto & i : x) if(i<v) v=i;
                                return v;
                        }
        };
	template <typename ColType,typename ValType>
        class ScalarPtSum {
                public:
                        static ColType convert(ValType x){
                                ColType v=0;
                                for(const auto & i : x) v+=i.pt();
                                return v;
                        }
        };
        template <typename ColType,typename ValType>
        class MassSum {
                public:
                        static ColType convert(ValType x){
			        if(x.empty()) return 0;
 			        auto v=x[0].p4();
                                for(const auto & i : x) v+=i.p4();
                                return v.mass();
                        }
        };
	template <typename ColType,typename ValType>
        class PtVectorSum {
                public:
                        static ColType convert(ValType x){
				if(x.empty()) return 0;
                                auto v=x[0].p4();
				v-=x[0].p4();
                                for(const auto & i : x) v+=i.p4();
                                return v.pt();
                        }
        };



        template<typename ValType, typename ColType=ValType,  typename Converter=Identity<ValType> >
            class VariableT : public Variable {
                public:
                    VariableT(const std::string & aname, nanoaod::FlatTable::ColumnType atype, const edm::ParameterSet & cfg, edm::ConsumesCollector && cc) :
                        Variable(aname, atype, cfg), src_(cc.consumes<ValType>(cfg.getParameter<edm::InputTag>("src"))) {}
                    ~VariableT() override {}
                    void fill(const edm::Event &iEvent, nanoaod::FlatTable & out) const override {
                        edm::Handle<ValType> handle;
                        iEvent.getByToken(src_, handle);
                        out.template addColumnValue<ColType>(this->name_, Converter::convert(*handle), this->doc_, this->type_);
                    }
                protected:
                    edm::EDGetTokenT<ValType> src_;
            };
        typedef VariableT<int> IntVar;
        typedef VariableT<float> FloatVar;
        typedef VariableT<double,float> DoubleVar;
        typedef VariableT<bool,uint8_t> BoolVar;
        typedef VariableT<edm::View<reco::Candidate>,float,ScalarPtSum<float,edm::View<reco::Candidate>>> CandidateScalarSumVar;
        typedef VariableT<edm::View<reco::Candidate>,float,MassSum<float,edm::View<reco::Candidate>>> CandidateSumMassVar;
        typedef VariableT<edm::View<reco::Candidate>,int,Size<edm::View<reco::Candidate>>> CandidateSizeVar;
        boost::ptr_vector<Variable> vars_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GlobalVariablesTableProducer);

