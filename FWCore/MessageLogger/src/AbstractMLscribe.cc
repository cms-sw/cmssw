#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

namespace edm  {
namespace service { 

AbstractMLscribe::AbstractMLscribe() {}
AbstractMLscribe::~AbstractMLscribe() {}
void AbstractMLscribe::runCommand(MessageLoggerQ::OpCode, void *) {}

}   // end of namespace service
}  // namespace edm
