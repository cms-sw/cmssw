#ifndef Mag_MagExceptions_H
#define Mag_MagExceptions_H

#include <exception>

class MagException : public std::exception {
public:
  MagException() throw() {}
  MagException( const std::string& message) : theMessage(message) {}
  virtual ~MagException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

class MagGeometryError : public MagException {
public:
  MagGeometryError() throw() {}
  MagGeometryError( const std::string& message) : MagException(message) {}
  virtual ~MagGeometryError() throw() {}
};

class MagLogicError : public MagException {
public:
  MagLogicError() throw() {}
  MagLogicError( const std::string& message) : MagException(message) {}
  virtual ~MagLogicError() throw() {}
};
#endif
