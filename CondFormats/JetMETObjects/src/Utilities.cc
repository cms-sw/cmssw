#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


#ifdef STANDALONE
#include <stdexcept>
#else
#include "FWCore/Utilities/interface/Exception.h"
#endif

namespace 
{
  void handleError(const std::string& fClass, const std::string& fMessage);
  //----------------------------------------------------------------------
  float getFloat(const std::string& token) 
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
  unsigned getUnsigned(const std::string& token) 
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
  //----------------------------------------------------------------------
  std::string getSection(const std::string& token) 
  {
    size_t iFirst = token.find ('[');
    size_t iLast = token.find (']');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1); 
    return "";
  }
  //----------------------------------------------------------------------
  std::vector<std::string> getTokens(const std::string& fLine)
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
  std::string getDefinitions(const std::string& token) 
  {
    size_t iFirst = token.find ('{');
    size_t iLast = token.find ('}');
    if (iFirst != std::string::npos && iLast != std::string::npos && iFirst < iLast)
      return std::string (token, iFirst+1, iLast-iFirst-1); 
    return "";
  }
  //------------------------------------------------------------------------ 
  void handleError(const std::string& fClass, const std::string& fMessage)
  {
#ifdef STANDALONE 
    std::stringstream sserr;
    sserr<<fClass<<" ERROR: "<<fMessage;
    throw std::runtime_error(sserr.str());
#else
    throw cms::Exception(fClass)<<fMessage;
#endif
  }
  //------------------------------------------------------------------------ 
  float quadraticInterpolation(float fZ, const float fX[3], const float fY[3])
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
