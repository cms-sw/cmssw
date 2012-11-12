#include "DetectorDescription/Base/interface/DDException.h"

// "regular" constructor
DDException::DDException(const std::string & s)
  : cms::Exception("DetectorDescriptionFault", s)
{ }

// default
DDException::DDException()
  : cms::Exception("DetectorDescriptionFault")
{ }

// copy constructor
DDException::DDException(const DDException& e)
  : cms::Exception(e)
{ }

void
DDException::swap(DDException& other) throw()
{
  cms::Exception::swap(other);
}

DDException&
DDException::operator=(DDException const& other)
{
  DDException temp(other);
  this->swap(temp);
  return *this;
}

DDException::~DDException() throw()
{ }

