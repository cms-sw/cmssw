#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

L1TriggerKey L1TriggerKey::fromRun (const std::string & tag, const unsigned long long run)
{
    std::stringstream ss;
    ss << "L1Tag:" << tag << "_run:" << run;
    return L1TriggerKey (ss.str ());
}
