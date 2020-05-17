// Date   : 30/05/2005
// Author : N.Almeida (LIP)

#ifndef ECALTBPARSEREXCEPTION_H
#define ECALTBPARSEREXCEPTION_H

#include <iostream>
#include <string>
#include <utility>

class ECALTBParserException {
public:
  /**
			 * Constructor
			 */
  ECALTBParserException(std::string exceptionInfo_) { info_ = std::move(exceptionInfo_); }

  /**
			 * Exception's discription
			 */
  const char* what() const throw() { return info_.c_str(); }

protected:
  std::string info_;
};

#endif
