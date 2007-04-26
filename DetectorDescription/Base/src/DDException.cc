#include "DetectorDescription/Base/interface/DDException.h"

// "regular" constructor
DDException::DDException(const std::string & s) : cms::Exception("DetectorDescriptionFault", s) { }

// default
DDException::DDException() : cms::Exception("DetectorDescriptionFault") { }

// copy constructor
DDException::DDException(const DDException& e)  : cms::Exception(e) { }

DDException::~DDException() throw() { }

