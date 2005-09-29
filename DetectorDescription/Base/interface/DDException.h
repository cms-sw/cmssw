#ifndef DDException_h
#define DDException_h

/* #include <string> */
#include<iostream>
//#include "Utilities/GenUtil/interface/CMSexception.h"
#include "SealBase/Error.h"

//! An exception for DDD errors
/** @class DDException DDException.h
 *
 *  @author:  Martin Listd::endl               Initial Version
 *  @version: 0.0
 *  @date:    
 * 
 *  Description:
 *  
 *  Provides an exception for DDD errors.
 *
 *  Modifications:
 *  MEC:   8 June 2005 Michael Case: changed to inherit fromseal::Error
 */

class DDException : public seal::Error //: public Genexception
{
public: 
  //! constructor takes simply an error message via a std::string
  explicit DDException(const std::string & s);
  DDException(const DDException& dde);

  virtual ~DDException();

  //! seal::Error required implementations.			
  virtual std::string	explainSelf() const;
			
  virtual void	rethrow (void);

  virtual DDException* clone() const;

  //! other methods just for this to work with DDException legacy...
  std::string message() const;

  const char* what() const;

 private:
  std::string m_message;
			
};    

//! stream the exception message
std::ostream & operator<<(std::ostream & os, const DDException & ex);
#endif
