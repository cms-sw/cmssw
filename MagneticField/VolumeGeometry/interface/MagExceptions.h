#ifndef Mag_MagExceptions_H
#define Mag_MagExceptions_H

#include <exception>
#include <string>

class MagException : public std::exception {
public:
  MagException() throw() {}
  MagException( const char *message);
  virtual ~MagException() throw() {}
  virtual const char* what() const throw() { return theMessage.c_str();}
private:
  std::string theMessage;
};

class MagGeometryError : public MagException {
public:
  MagGeometryError() throw() {}
  MagGeometryError(const char *message) : MagException(message) {}
  virtual ~MagGeometryError() throw() {}
};

class MagLogicError : public MagException {
public:
  MagLogicError() throw() {}
  MagLogicError(const char *message) : MagException(message) {}
  virtual ~MagLogicError() throw() {}
};

class GridInterpolator3DException : public std::exception {
public:

  GridInterpolator3DException( double a1, double b1, double c1,
				     double a2, double b2, double c2)  throw() :
    a1_(a1), b1_(b1), c1_(c1), a2_(a2), b2_(b2), c2_(c2) {}

  virtual ~GridInterpolator3DException() throw() {}

  virtual const char* what() const throw() { return "LinearGridInterpolator3D: field requested outside of grid validity";}

  double a1_, b1_, c1_, a2_, b2_, c2_;
};



#endif
