#ifndef PhysicsTools_Utilities_AnyMethodArgument_h
#define PhysicsTools_Utilities_AnyMethodArgument_h

#include "Reflex/Object.h"
#include "Reflex/Member.h"
#include <algorithm>

#include <string>
#include <boost/variant.hpp>

namespace reco {
  namespace parser {

    typedef boost::variant<
                int8_t, uint8_t,
                int16_t, uint16_t,
                int32_t, uint32_t,
                int64_t, uint64_t,
                double,float,
                std::string> AnyMethodArgument;

    class AnyMethodArgumentFixup : public boost::static_visitor<std::pair<AnyMethodArgument, int> > {
        private:
            const std::type_info & type_;
            template<typename From, typename To>
            std::pair<AnyMethodArgument, int> retOk_(const From &f, int cast) const {
                return std::pair<AnyMethodArgument,int>(AnyMethodArgument(static_cast<To>(f)), cast);
            }

            // correct return for each int output type
            std::pair<AnyMethodArgument,int> doInt(int t) const {
                if (type_ == typeid(int8_t))   { return retOk_<int,int8_t>  (t,0); }
                if (type_ == typeid(uint8_t))  { return retOk_<int,uint8_t> (t,0); }
                if (type_ == typeid(int16_t))  { return retOk_<int,int16_t> (t,0); }
                if (type_ == typeid(uint16_t)) { return retOk_<int,uint16_t>(t,0); }
                if (type_ == typeid(int32_t))  { return retOk_<int,int32_t> (t,0); }
                if (type_ == typeid(uint32_t)) { return retOk_<int,uint32_t>(t,0); }
                if (type_ == typeid(int64_t))  { return retOk_<int,int64_t> (t,0); }
                if (type_ == typeid(uint64_t)) { return retOk_<int,uint64_t>(t,0); }
                if (type_ == typeid(double))   { return retOk_<int,double>  (t,1); }
                if (type_ == typeid(float))    { return retOk_<int,float>   (t,1); }
                return std::pair<AnyMethodArgument,int>(t,-1);
            }
        public:
            AnyMethodArgumentFixup(const ROOT::Reflex::Type & type) : 
                type_(type.Name() == "string" ? typeid(std::string) : type.TypeInfo()) // Otherwise Reflex does this wrong :-(
            {
                //std::cerr << "\nAnyMethodArgumentFixup: Conversion [" << type.Name() << "] => [" << type_.name() << "]" << std::endl;
            }

            // we handle all integer types through 'int', as that's the way they are parsed by boost::spirit
            std::pair<AnyMethodArgument,int> operator()(const  int8_t  &t)  const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const uint8_t  &t)  const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const  int16_t &t) const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const uint16_t &t) const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const  int32_t &t) const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const uint32_t &t) const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const  int64_t &t) const { return doInt(t); }
            std::pair<AnyMethodArgument,int> operator()(const uint64_t &t) const { return doInt(t); }

            std::pair<AnyMethodArgument,int> operator()(const float &t) const { 
                if (type_ == typeid(double)) { return retOk_<float,double>(t,0); }
                if (type_ == typeid(float))  { return retOk_<float,float> (t,0); }
                return std::pair<AnyMethodArgument,int>(t,-1);
            }
            std::pair<AnyMethodArgument,int> operator()(const double &t) const { 
                if (type_ == typeid(double)) { return retOk_<double,double>(t,0); }
                if (type_ == typeid(float))  { return retOk_<double,float> (t,0); }
                return std::pair<AnyMethodArgument,int>(t,-1);
            }
            std::pair<AnyMethodArgument,int> operator()(const std::string &t) const { 
                if (type_ == typeid(std::string)) { return std::pair<AnyMethodArgument,int>(t,0); }
                return std::pair<AnyMethodArgument,int>(t,-1);
            }

    };
    
    class AnyMethodArgument2VoidPtr : public boost::static_visitor<void *> {
        public:
            template<typename T>
            void * operator()(const T &t) const { return const_cast<void*>(static_cast<const void *>(&t)); }
    };
  }
}

#endif
