#ifndef L1TObjects_L1TriggerKey_h
#define L1TObjects_L1TriggerKey_h

#include <string>
#include <sstream>

/**
 * Description: L1 key used to load all other configuration data from offline db
 */
class L1TriggerKey
{
protected:

    // this object stores only one data field.
    // this type is used for this field.
    std::string _key;
public:
    // Constructors
    L1TriggerKey () {}
    L1TriggerKey (std::string key) : _key (key) {}

    /* Helper method that will create key from provided tag and run number.
     * This key will have value created from these two parameters.
     */
    static L1TriggerKey fromRun (const std::string & tag, const unsigned long long run);

    // getters/setters
    void setKey (const std::string key) { _key = key; }
    std::string getKey () const { return _key; }
};

#endif

