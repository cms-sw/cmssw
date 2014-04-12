#include "FWCore/Utilities/src/Guid.h"

#include <string>
#include <cassert>
#include <iostream>

int main() {
	edm::Guid guid;
	std::string guidString = guid.toString();
	edm::Guid guid2;
	guid2.fromString(guidString);
	edm::Guid guid3(guid2);
	assert(guid == guid2);
	assert(guid == guid3);
	std::string guidString2 = guid2.toString();
	std::string guidString3 = guid3.toString();
	assert(guidString2 == guidString);
	assert(guidString3 == guidString);
}
