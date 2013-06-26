/*  =====================================================================================
 *
 *       Filename:  CSCDQM_Utility.cc
 *
 *    Description:  Histogram Utility code
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 *  =====================================================================================
 */

#ifdef CSC_RENDER_PLUGIN
#include "CSCDQM_Utility.h"
#else
#include "CSCDQM_Utility.h"
#endif

#include <stdint.h>

namespace cscdqm {

  /**
    * @brief  Get CSC y-axis position from chamber string
    * @param  cstr Chamber string
    * @return chamber y-axis position
    */
  int Utility::getCSCTypeBin(const std::string& cstr) {
    if (cstr.compare("ME-4/2") == 0) return 0;
    if (cstr.compare("ME-4/1") == 0) return 1;
    if (cstr.compare("ME-3/2") == 0) return 2;
    if (cstr.compare("ME-3/1") == 0) return 3;
    if (cstr.compare("ME-2/2") == 0) return 4;
    if (cstr.compare("ME-2/1") == 0) return 5;
    if (cstr.compare("ME-1/3") == 0) return 6;
    if (cstr.compare("ME-1/2") == 0) return 7;
    if (cstr.compare("ME-1/1") == 0) return 8;
    if (cstr.compare("ME+1/1") == 0) return 9;
    if (cstr.compare("ME+1/2") == 0) return 10;
    if (cstr.compare("ME+1/3") == 0) return 11;
    if (cstr.compare("ME+2/1") == 0) return 12;
    if (cstr.compare("ME+2/2") == 0) return 13;
    if (cstr.compare("ME+3/1") == 0) return 14;
    if (cstr.compare("ME+3/2") == 0) return 15;
    if (cstr.compare("ME+4/1") == 0) return 16;
    if (cstr.compare("ME+4/2") == 0) return 17;
    return 0;
  }
  
  /**
   * @brief  Get CSC label from CSC parameters
   * @param  endcap Endcap number
   * @param  station Station number
   * @param  ring Ring number
   * @return chamber label
   */
  std::string Utility::getCSCTypeLabel(int endcap, int station, int ring ) {
    std::string label = "Unknown";
    std::ostringstream st;
    if ((endcap > 0) && (station > 0) && (ring > 0)) {
      if (endcap == 1) {
        st << "ME+" << station << "/" << ring;
        label = st.str();
      } else if (endcap==2) {
        st << "ME-" << station << "/" << ring;
        label = st.str();
      } else {
        label = "Unknown";
      }
    }
    return label;
  }
  
  
  /**
   * @brief  Break string into tokens
   * @param  str source string to break
   * @param  tokens pointer to result vector
   * @param  delimiters delimiter string, default " "
   * @return 
   */
  int Utility::tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters) {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos = str.find_first_of(delimiters, lastPos);
    }
    return tokens.size();
  }
  
  /**
   * @brief  Split string according to delimiter
   * @param  str String to split
   * @param  delim Delimiter
   * @param  results Vector to write results to
   * @return 
   */
  void Utility::splitString(const std::string& str, const std::string& delim, std::vector<std::string>& results) {
    std::string::size_type lastPos = str.find_first_not_of(delim, 0);
    std::string::size_type pos     = str.find_first_of(delim, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      results.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delim, pos);
      pos = str.find_first_of(delim, lastPos);
    }
  }
  
  /**
   * @brief  Trim string
   * @param  str string to trim
   */
  void Utility::trimString(std::string& str) {
    std::string::size_type pos = str.find_last_not_of(' ');
    if(pos != std::string::npos) {
      str.erase(pos + 1);
      pos = str.find_first_not_of(' ');
      if(pos != std::string::npos) {
        str.erase(0, pos);
      }
    } else {
      str.erase(str.begin(), str.end());
    }
  }
  
  
  /**
   * @brief  Match RegExp expression against string message and return result
   * @param  re_expression RegExp expression to match
   * @param  message value to check
   * @return true if message matches RegExp expression
   */
  bool Utility::regexMatch(const TPRegexp& re_expression, const std::string& message) {
    TPRegexp *re = const_cast<TPRegexp*>(&re_expression);
    return re->MatchB(message);
  }

  /**
   * @brief  Match RegExp expression string against string message and return result
   * @param  expression RegExp expression in string to match
   * @param  message value to check
   * @return true if message matches RegExp expression
   */
  bool Utility::regexMatch(const std::string& expression, const std::string& message) {
    return regexMatch(TPRegexp(expression), message);
  }

  /**
   * @brief  Replace string part that matches RegExp expression with some
   * string
   * @param  expression RegExp expression in string to match
   * @param  message value to check
   * @param  replace string to replace matched part 
   */
  void Utility::regexReplace(const std::string& expression, std::string& message, const std::string replace) {
    Utility::regexReplace(TPRegexp(expression), message, replace);
  }

  /**
   * @brief  Replace string part that matches RegExp expression with some
   * string
   * @param  re_expression RegExp expression to match
   * @param  message value to check
   * @param  replace string to replace matched part 
   */
  void Utility::regexReplace(const TPRegexp& re_expression, std::string& message, const std::string replace) {
    TString s(message); 
    TPRegexp *re = const_cast<TPRegexp*>(&re_expression);
    re->Substitute(s, replace);
    message = s;
  }

  /**
   * @brief  Replace string part that matches RegExp expression with some
   * string
   * @param  expression RegExp expression in string to match
   * @param  message value to check
   * @param  replace string to replace matched part 
   * @return modified string
   */
  std::string Utility::regexReplaceStr(const std::string& expression, const std::string& message, const std::string replace) {
    return regexReplaceStr(TPRegexp(expression), message, replace);
  }

  /**
   * @brief  Replace string part that matches RegExp expression with some
   * string
   * @param  re_expression RegExp expression to match
   * @param  message value to check
   * @param  replace string to replace matched part 
   * @return modified string
   */
  std::string Utility::regexReplaceStr(const TPRegexp& re_expression, const std::string& message, const std::string replace) {
    TString s(message); 
    TPRegexp *re = const_cast<TPRegexp*>(&re_expression);
    re->Substitute(s, replace);
    return s.Data();
  }

#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8) + (uint32_t)(((const uint8_t *)(d))[0]) )
#endif

  /**
  * @brief  Calculate super fast hash (from http://www.azillionmonkeys.com/qed/hash.html)
  * @param  data Source Data 
  * @param  length of data
  * @return hash result
  */
  uint32_t Utility::fastHash(const char * data, int len) {
    uint32_t hash = len, tmp;
    int rem;
  
    if (len <= 0 || data == NULL) return 0;
    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
      hash  += get16bits (data);
      tmp    = (get16bits (data+2) << 11) ^ hash;
      hash   = (hash << 16) ^ tmp;
      data  += 2*sizeof (uint16_t);
      hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
      case 3: hash += get16bits (data);
              hash ^= hash << 16;
              hash ^= data[sizeof (uint16_t)] << 18;
              hash += hash >> 11;
              break;
      case 2: hash += get16bits (data);
              hash ^= hash << 11;
              hash += hash >> 17;
              break;
      case 1: hash += *data;
              hash ^= hash << 10;
              hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
  }

  
  /**
   * @brief  Check the hypothesis that observed events (n) value is too low or too high comparing with the expected N 
   * @param  N Expected number of events
   * @param  n Actual (observed) number of events
   * @param  low_threshold Rate of lower boundary of tolerance (< 1)
   * @param  high_threshold Rate of higher boundary of tolerance (> 1)
   * @param  low_sigfail Significance threshold for low value
   * @param  high_sigfail Significance threshold for high value
   * @return check outcome: 1 - observed number of events too high wrt expected
   * (HOT), -1 - observed number of events too low wrt expected (COLD), 0 -
   * observed number of events is fine wrt expected
   */
  short Utility::checkOccupancy(const unsigned int N, const unsigned int n, const double low_threshold, const double high_threshold, const double low_sigfail, const double high_sigfail) {
    if (N > 0) {
      double eps_meas = (1.0 * n) / (1.0 * N);
      if (eps_meas < low_threshold) {
        double S = Utility::SignificanceLevelLow(N, n, low_threshold);
        if (S > low_sigfail) return -1;
      } else 
      if (eps_meas > high_threshold) {
        double S = Utility::SignificanceLevelHigh(N, n);
        if (S > high_sigfail) return 1;
      }
    }
    return 0;
  }

  /**
   * @brief  Check the hypothesis that error events (n) value above threshold comparing with the expected 0 and statistics is enough
   * @param  N Number of total events
   * @param  n Actual (observed) number of events errors
   * @param  threshold Rate of tolerance (<1)
   * @param  sigfail Significance threshold for low value
   * @return check result: true - error is significant, false - otherwise
   */
  bool Utility::checkError(const unsigned int N, const unsigned int n, const double threshold, const double sigfail) {
    if (N > 0) {
      const double eps_meas = (1.0 * n) / (1.0 * N);
      if (eps_meas > threshold) {
        if (Utility::SignificanceLevelLow(N, n, threshold) > sigfail) {
          return true;
        }
      } 
    }
    return false;
  }

  /**
   * @brief  Calculate error significance alpha for the given number of errors 
   * based on reference number of errors for "cold" elements: actual number of 
   * events have to be less then the reference.
   * @param  N Number of events
   * @param  n Number of errors
   * @param  eps Rate of tolerance
   * @return Significance level
   */
  double Utility::SignificanceLevelLow(const unsigned int N, const unsigned int n, const double eps) {
  
    /** std::cout << "N = " << N << ", n = " << n << ", eps = " << eps << "\n"; */
  
    double l_eps = eps;
    if (l_eps <= 0.0) l_eps = 0.000001;
    if (l_eps >= 1.0) l_eps = 0.999999;
  
    double eps_meas = (1.0 * n) / (1.0 * N);
    double a = 1.0, b = 1.0;
  
    if (n > 0) {
      for (unsigned int r = 0; r < n; r++) a = a * (eps_meas / l_eps);
    }
  
    if (n < N) {
      for (unsigned int r = 0; r < (N - n); r++) b = b * (1 - eps_meas) / (1 - l_eps);
    }
  
    return sqrt(2.0 * log(a * b));
  
  }
  
  /**
   * @brief  Calculate error significance alpha for the given number of events
   * based on reference number of errors for "hot" elements: actual number of
   * events have to be larger then the reference.
   * @param  N number of reference events
   * @param  n number of actual events
   * @return error significance
   */
  double Utility::SignificanceLevelHigh(const unsigned int N, const unsigned int n) {
    if (N > n) return 0.0;
    /**  no - n observed, ne - n expected */
    double no = 1.0 * n, ne = 1.0 * N;
    return sqrt(2.0 * (no * (log(no / ne) - 1) + ne));
  }

}
