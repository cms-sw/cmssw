#ifndef PhysicsTools_NanoAOD_UniqueString_h
#define PhysicsTools_NanoAOD_UniqueString_h

#include <string>

namespace nanoaod {

class UniqueString {
    public:
        UniqueString() {}
        UniqueString(const std::string & str) : str_(str) {}
        const std::string & str() const { return str_; }
        bool operator==(const std::string & other) const { return str_ == other; }
        bool operator==(const UniqueString & other) const { return str_ == other.str_; }
        bool isProductEqual(const UniqueString & other) const { return (*this) == other; }
    private:
        std::string str_;
};

} // namespace nanoaod

#endif
