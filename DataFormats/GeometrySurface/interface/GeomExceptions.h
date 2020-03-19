#ifndef Geom_GeomExceptions_H
#define Geom_GeomExceptions_H

#include <exception>
#include <string>

class BaseGeomException : public std::exception {
public:
  BaseGeomException() throw() {}
  BaseGeomException(const std::string& message) : theMessage(message) {}
  ~BaseGeomException() throw() override {}
  const char* what() const throw() override { return theMessage.c_str(); }

private:
  std::string theMessage;
};

class GeometryError : public BaseGeomException {
public:
  GeometryError() throw() {}
  GeometryError(const std::string& message) : BaseGeomException(message) {}
  ~GeometryError() throw() override {}
};

#endif
