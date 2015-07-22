#ifndef DataFormats_EventHypothesis_interface_EventHypothesis_h
#define DataFormats_EventHypothesis_interface_EventHypothesis_h

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include <typeinfo>
#include <iostream>
#include <type_traits>
#include <cxxabi.h>

namespace pat {
   // forward declaration
   namespace eventhypothesis { template<typename T> class Looper; }
   // real declarations
   namespace eventhypothesis { 
        // typedef for the Ref
        typedef reco::CandidatePtr CandRefType; // Ptr one day in the future
        // filter
        struct ParticleFilter {
            virtual ~ParticleFilter() {}
            bool operator()(const std::pair<std::string, CandRefType> &p) const { return operator()(p.second, p.first); }
            virtual bool operator()(const CandRefType &cand, const std::string &role) const = 0;
        };
        // smart pointer to the filter
        typedef boost::shared_ptr<const ParticleFilter> ParticleFilterPtr;
   }
   // the class
   class EventHypothesis {
        public:
            typedef eventhypothesis::CandRefType              CandRefType;
            typedef std::pair<std::string, CandRefType>       value_type;
            typedef std::vector<value_type>                   vector_type;
            typedef vector_type::const_iterator               const_iterator;
            typedef vector_type::const_reverse_iterator       const_reverse_iterator;
            typedef eventhypothesis::Looper<reco::Candidate>  CandLooper;

            void add(const CandRefType &ref, const std::string &role) ;

            const_iterator begin() const { return particles_.begin(); }
            const_iterator end()   const { return particles_.end();   }
            const_reverse_iterator rbegin() const { return particles_.rbegin(); }
            const_reverse_iterator rend()   const { return particles_.rend();   }

            class ByRole {
                public:
                    ByRole(const std::string &role) : role_(role) {}
                    bool operator()(const value_type &p) const { return p.first == role_; }
                private:
                    const std::string &role_;
            };

            typedef eventhypothesis::ParticleFilter ParticleFilter;
            typedef eventhypothesis::ParticleFilterPtr ParticleFilterPtr;
    
            const CandRefType & get(const std::string &role, int index=0) const ;
            const CandRefType & get(const ParticleFilter &filter, int index=0) const ;
            template<typename T> const T * getAs(const std::string &role, int index=0) const ;
            template<typename T> const T * getAs(const ParticleFilter &filter, int index=0) const ;
            const CandRefType & operator[](const std::string &role) const { return get(role,0); }
            const CandRefType & operator[](const ParticleFilter &filter) const { return get(filter,0); }

            /// Return EDM references to all particles which have certaint roles. 
            std::vector<CandRefType> all(const std::string &roleRegexp) const;
            /// Return EDM references to all particles which satisfy some condition. 
            std::vector<CandRefType> all(const ParticleFilter &filter) const;

            size_t count() const { return particles_.size(); }
            /// Counts particles which have certaint roles. 
            size_t count(const std::string &roleRegexp) const;
            /// Counts particles which satisfy some condition. 
            size_t count(const ParticleFilter &role) const;
            
            /// Loops over all particles
            CandLooper loop() const ;
            /// Loops over particles which have certaint roles. 
            CandLooper loop(const std::string &roleRegexp) const;
            /// Loops over particles which satisfy some condition. 
            /// The caller code owns the filter, and must take care it is not deleted while the looper is still being used
            CandLooper loop(const ParticleFilter &filter) const;
            /// Loops over particles which satisfy some condition. 
            /// The looper owns the filter, which will be deleted when the looper is deleted.
            /// That is, you can call eventHypothesis.loop(new WhateverFilterYouLike(...))
            CandLooper loop(const ParticleFilter *filter) const;
            /// Loops over particles which satisfy some condition. 
            CandLooper loop(const ParticleFilterPtr &filter) const;



            /// Loops over particles which have certaint roles. 
            template<typename T> eventhypothesis::Looper<T> loopAs(const std::string &roleRegexp) const ;
            /// Loops over particles which satisfy some condition. 
            /// The caller code owns the filter, and must take care it is not deleted while the looper is still being used
            template<typename T> eventhypothesis::Looper<T> loopAs(const ParticleFilter &filter) const;
            /// Loops over particles which satisfy some condition. 
            /// The looper owns the filter, which will be deleted when the looper is deleted.
            /// That is, you can call eventHypothesis.loopAs<...>(new WhateverFilterYouLike(...))
            template<typename T> eventhypothesis::Looper<T> loopAs(const ParticleFilter *filter) const;
            /// Loops over particles which satisfy some condition. 
            template<typename T> eventhypothesis::Looper<T> loopAs(const ParticleFilterPtr &filter) const;
        private:
            template<typename Iterator, typename Predicate>
            Iterator realGet(const Iterator &realBegin, const Iterator &realEnd, const Predicate &p, size_t idx) const ;
	    char * getDemangledSymbol(const char* mangledSymbol) const;
	    template<typename T> std::string createExceptionMessage(const CandRefType &ref) const;

            std::vector<value_type> particles_;

   } ;

   namespace eventhypothesis {
        struct AcceptAllFilter : public ParticleFilter {
            AcceptAllFilter(){}
            static const AcceptAllFilter & get() { return s_dummyFilter; }
            virtual bool operator()(const CandRefType &cand, const std::string &role) const { return true; }
            private:
               static const AcceptAllFilter s_dummyFilter;
        };
        class RoleRegexpFilter : public ParticleFilter {
            public:
                explicit RoleRegexpFilter(const std::string &roleRegexp) : re_(roleRegexp) {}
                virtual bool operator()(const CandRefType &cand, const std::string &role) const {
                    return boost::regex_match(role, re_);
                }
            private:
                boost::regex re_;
        };
   }
 
   template<typename Iterator, typename Predicate>
   Iterator EventHypothesis::realGet(const Iterator &realBegin, const Iterator &realEnd, const Predicate &pred, size_t idx) const  
   {
        Iterator it = realBegin;
        while (it != realEnd) {
            if (pred(*it)) {
                if (idx == 0) return it;
                idx--;
            }
            ++it;
        }
        return it;
   }

   template<typename T>
   std::string
   EventHypothesis::createExceptionMessage(const CandRefType &ref) const {
     std::stringstream message;
     char *currentType = getDemangledSymbol(typeid(std::remove_reference<decltype(ref)>::type::value_type).name());
     char *targetType = getDemangledSymbol(typeid(T).name());
     if (currentType != nullptr && targetType != nullptr) {
       message << "You can't convert a '" << currentType << "' to a '" << targetType << "'" << std::endl;
       free(currentType);
       free(targetType);
     } else {
       message << "You can't convert a '" << typeid(ref).name() << "' to a '" << typeid(T).name() << "'" << std::endl;
       message << "Note: you can use 'c++filt -t' command to convert the above in human readable types." << std::endl;
     }
     return message.str();
   }

   template<typename T> 
   const T * 
   EventHypothesis::getAs(const std::string &role, int index) const 
   {
       CandRefType ref = get(role, index);
       const T* ret = dynamic_cast<const T*>(ref.get());
       if ((ret == 0) && (ref.get() != 0)) throw cms::Exception("Type Checking") << createExceptionMessage<T>(ref);
       return ret;
   }
   template<typename T> 
   const T * 
   EventHypothesis::getAs(const ParticleFilter &filter, int index) const 
   {
       CandRefType ref = get(filter, index);
       const T* ret = dynamic_cast<const T*>(ref.get());
       if ((ret == 0) && (ref.get() != 0)) throw cms::Exception("Type Checking") << createExceptionMessage<T>(ref);
       return ret;
   }
   template<typename T>
   eventhypothesis::Looper<T>
   EventHypothesis::loopAs(const std::string &roleRegexp) const 
   {
       return loopAs<T>(new pat::eventhypothesis::RoleRegexpFilter(roleRegexp));
   }

   template<typename T>
   eventhypothesis::Looper<T>
   EventHypothesis::loopAs(const ParticleFilter &role) const
   {
       return pat::eventhypothesis::Looper<T>(*this, role);
   }

   template<typename T>
   eventhypothesis::Looper<T>
   EventHypothesis::loopAs(const ParticleFilter *role) const
   {
       return pat::eventhypothesis::Looper<T>(*this, role);
   }

   template<typename T>
   eventhypothesis::Looper<T>
   EventHypothesis::loopAs(const ParticleFilterPtr &role) const
   {
       return pat::eventhypothesis::Looper<T>(*this, role);
   }
}

#endif
