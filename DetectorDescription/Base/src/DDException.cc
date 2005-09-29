
#include "DetectorDescription/Base/interface/DDException.h"
#include "SealBase/Error.h"

// "regular" constructor
DDException::DDException(const std::string & s) : seal::Error() {}

// copy constructor
DDException::DDException(const DDException& e)
  : Error(e)
  , m_message(e.message())
{ }

DDException::~DDException() { }

std::string DDException::explainSelf() const {
  return m_message;
}
			
void DDException::rethrow (void) {
  throw (*this);
}

DDException* DDException::clone() const {
  return new DDException(*this);
}

std::string DDException::message() const {
  return explainSelf();
}

const char* DDException::what() const {
  return explainSelf().c_str();
}

std::ostream & operator<<(std::ostream & os, const DDException & ex)
{
  os << ex.what();
  return os;
}
