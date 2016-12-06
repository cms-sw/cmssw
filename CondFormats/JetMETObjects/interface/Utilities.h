#ifndef CondFormats_JetMETObjects_Utilities_h
#define CondFormats_JetMETObjects_Utilities_h

#ifdef STANDALONE
#include <stdexcept>
#else
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <utility>

namespace std
{
  //print a tuple
  template<typename Type, unsigned N, unsigned Last>
  struct tuple_printer {
  
      static void print(std::ostream& out, const Type& value) {
          out << std::get<N>(value) << ", ";
          tuple_printer<Type, N + 1, Last>::print(out, value);
      }
  };
  template<typename Type, unsigned N>
  struct tuple_printer<Type, N, N> {
  
      static void print(std::ostream& out, const Type& value) {
          out << std::get<N>(value);
      }
  
  };
  template<typename... Types>
  std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& value) {
      out << "(";
      tuple_printer<std::tuple<Types...>, 0, sizeof...(Types) - 1>::print(out, value);
      out << ")";
      return out;
  }

  //----------------------------------------------------------------------
  //list of type indices
  template <size_t... n>
  struct ct_integers_list {
      template <size_t m>
      struct push_back
      {
          typedef ct_integers_list<n..., m> type;
      };
  };
  template <size_t max>
  struct ct_iota_1
  {
      typedef typename ct_iota_1<max-1>::type::template push_back<max>::type type;
  };
  template <>
  struct ct_iota_1<0>
  {
      typedef ct_integers_list<> type;
  };

  //----------------------------------------------------------------------
  //return a subset of the tuple
  template <size_t... indices, typename Tuple>
  auto tuple_subset(const Tuple& tpl, ct_integers_list<indices...>)
      -> decltype(std::make_tuple(std::get<indices>(tpl)...))
  {
      return std::make_tuple(std::get<indices>(tpl)...);
      // this means:
      //   make_tuple(get<indices[0]>(tpl), get<indices[1]>(tpl), ...)
  }
  template <typename Head, typename... Tail>
  std::tuple<Tail...> tuple_tail(const std::tuple<Head, Tail...>& tpl)
  {
      return tuple_subset(tpl, typename ct_iota_1<sizeof...(Tail)>::type());
      // this means:
      //   tuple_subset<1, 2, 3, ..., sizeof...(Tail)-1>(tpl, ..)
  }

  //----------------------------------------------------------------------
  // Recursive hashing function for tuples
  template<typename Head, typename... ndims> struct hash_specialization
  {
    typedef std::tuple<Head,ndims...> argument_type;
    typedef std::size_t result_type;
    result_type operator()(const argument_type& t) const
    {
      const uint32_t& b = reinterpret_cast<const uint32_t&>(std::get<0>(t));
      //const uint32_t& more = (*this)(tuple_tail(t));
      const uint32_t& more = hash_specialization<ndims...>()(tuple_tail(t));
      return b^more;
    }
  };
  // Base case
  template<> struct hash_specialization<float>
  {
    typedef std::tuple<float> argument_type;
    typedef std::size_t result_type;
    result_type operator()(const argument_type& t) const
    {
      const uint32_t& b = reinterpret_cast<const uint32_t&>(std::get<0>(t));
      const result_type& result = reinterpret_cast<const result_type&>(b);
      return result;
    } 
  };
  // Overloaded verions of std::hash for tuples
  template<typename Head, typename... ndims> struct hash<std::tuple<Head, ndims...> >
  {
    typedef std::tuple<Head,ndims...> argument_type;
    typedef std::size_t result_type;
    result_type operator()(const argument_type& t) const
    {
      return hash_specialization<Head,ndims...>()(t);
    }
  };
}

namespace
{
  void handleError(const std::string& fClass, const std::string& fMessage)
  {
#ifdef STANDALONE
    std::stringstream sserr;
    sserr<<fClass<<" ERROR: "<<fMessage;
    throw std::runtime_error(sserr.str());
#else
    edm::LogError(fClass) << fMessage;
#endif
  }
  //----------------------------------------------------------------------
  inline float getFloat(const std::string& token)
  {
    char* endptr;
    float result = strtod (token.c_str(), &endptr);
    if (endptr == token.c_str())
      {
        std::stringstream sserr;
        sserr<<"can't convert token "<<token<<" to float value";
	handleError("getFloat",sserr.str());
      }
    return result;
  }
  //----------------------------------------------------------------------
  inline unsigned getUnsigned(const std::string& token)
  {
    char* endptr;
    unsigned result = strtoul (token.c_str(), &endptr, 0);
    if (endptr == token.c_str())
      {
        std::stringstream sserr;
        sserr<<"can't convert token "<<token<<" to unsigned value";
	handleError("getUnsigned",sserr.str());
      }
    return result;
  }
  inline long int getSigned(const std::string& token)
  {
    char* endptr;
    unsigned result = strtol (token.c_str(), &endptr, 0);
    if (endptr == token.c_str())
      {
        std::stringstream sserr;
        sserr<<"can't convert token "<<token<<" to signed value";
	handleError("getSigned",sserr.str());
      }
    return result;
  }
  //----------------------------------------------------------------------
  inline std::string getSection(const std::string& token)
  {
    size_t iFirst = token.find ('[');
    size_t iLast = token.find (']');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1);
    return "";
  }
  //----------------------------------------------------------------------
  inline std::vector<std::string> getTokens(const std::string& fLine)
  {
    std::vector<std::string> tokens;
    std::string currentToken;
    for (unsigned ipos = 0; ipos < fLine.length (); ++ipos)
      {
        char c = fLine[ipos];
        if (c == '#') break; // ignore comments
        else if (c == ' ')
          { // flush current token if any
            if (!currentToken.empty())
              {
	        tokens.push_back(currentToken);
	        currentToken.clear();
              }
          }
        else
          currentToken += c;
      }
    if (!currentToken.empty()) tokens.push_back(currentToken); // flush end
    return tokens;
  }
  //----------------------------------------------------------------------
  inline std::string getDefinitions(const std::string& token)
  {
    size_t iFirst = token.find ('{');
    size_t iLast = token.find ('}');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1);
    return "";
  }
  //------------------------------------------------------------------------
  inline float quadraticInterpolation(float fZ, const float fX[3], const float fY[3])
  {
    // Quadratic interpolation through the points (x[i],y[i]). First find the parabola that
    // is defined by the points and then calculate the y(z).
    float D[4],a[3];
    D[0] = fX[0]*fX[1]*(fX[0]-fX[1])+fX[1]*fX[2]*(fX[1]-fX[2])+fX[2]*fX[0]*(fX[2]-fX[0]);
    D[3] = fY[0]*(fX[1]-fX[2])+fY[1]*(fX[2]-fX[0])+fY[2]*(fX[0]-fX[1]);
    D[2] = fY[0]*(pow(fX[2],2)-pow(fX[1],2))+fY[1]*(pow(fX[0],2)-pow(fX[2],2))+fY[2]*(pow(fX[1],2)-pow(fX[0],2));
    D[1] = fY[0]*fX[1]*fX[2]*(fX[1]-fX[2])+fY[1]*fX[0]*fX[2]*(fX[2]-fX[0])+fY[2]*fX[0]*fX[1]*(fX[0]-fX[1]);
    if (D[0] != 0)
      {
        a[0] = D[1]/D[0];
        a[1] = D[2]/D[0];
        a[2] = D[3]/D[0];
      }
    else
      {
        a[0] = 0.0;
        a[1] = 0.0;
        a[2] = 0.0;
      }
    float r = a[0]+fZ*(a[1]+fZ*a[2]);
    return r;
  }
}
#endif
