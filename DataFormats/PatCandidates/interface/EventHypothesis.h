#ifndef DataFormats_PatCandidates_interface_EventHypothesis_h
#define DataFormats_PatCandidates_interface_EventHypothesis_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat {
   class EventHypothesis {
        public:
            typedef std::pair<reco::CandidateBaseRef, int32_t> value_type;
            typedef std::vector<value_type>                    vector_type;
            typedef vector_type::const_iterator                const_iterator;

            void add(const reco::CandidateBaseRef &ref, int32_t role) ;

            const_iterator begin() const { return particles_.begin(); }
            const_iterator end()   const { return particles_.end();   }

            const reco::CandidateBaseRef & operator[](int32_t role) const ;

            class ByRole {
                public:
                    ByRole(int32_t role) : role_(role) {}
                    bool operator()(const value_type &p) { return p.second == role_; }
                private:
                    int32_t role_;
            };
        private:
            std::vector<value_type> particles_;
   } ;
}

#endif
