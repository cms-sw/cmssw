#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cctype>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

#define NLOCAL 1024

namespace gs {
    void ClassId::setVersion(const unsigned newVersion)
    {
        if (version_ != newVersion)
        {
            version_ = newVersion;

            // Need to update the id string
            const std::size_t lastOpen = id_.find_last_of('(');
            assert(lastOpen != std::string::npos);

            std::ostringstream os;
            os << id_.substr(0, lastOpen) << '(' << version_ << ')';
            if (isPtr_)
                os << '*';
            id_ = os.str();
        }
    }

    void ClassId::ensureSameId(const ClassId& id) const
    {
        if (name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameId: reference id is not valid");
        if (id.name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameId: argument id is not valid");
        if (id_ != id.id_)
        {
            std::ostringstream os;
            os << "In gs::ClassId::ensureSameId: expected \""
               << id_ << "\", got \"" << id.id_ << "\"";
            throw gs::IOInvalidArgument(os.str());
        }
    }

    void ClassId::ensureSameName(const ClassId& id) const
    {
        if (name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameName: reference id is not valid");
        if (id.name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameName: argument id is not valid");
        if (name_ != id.name_)
        {
            std::ostringstream os;
            os << "In gs::ClassId::ensureSameName: expected class name \""
               << name_ << "\", got \"" << id.name_ << "\"";
            throw gs::IOInvalidArgument(os.str());
        }
    }

    void ClassId::ensureSameVersion(const ClassId& id) const
    {
        if (name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameVersion: reference id is not valid");
        if (id.name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureSameVersion: argument id is not valid");
        if (version_ != id.version_)
        {
            std::ostringstream os;
            os << "In gs::ClassId::ensureSameVersion: expected version "
               << version_ << " for class " << id.name_
               << ", got " << id.version_;
            throw gs::IOInvalidArgument(os.str());
        }
    }

    void ClassId::ensureVersionInRange(const unsigned vmin,
                                       const unsigned vmax) const
    {
        if (name_.empty())
            throw gs::IOInvalidArgument(
                "In gs::ClassId::ensureVersionInRange: id is not valid");
        if (version_ < vmin || version_ > vmax)
        {
            std::ostringstream os;
            os << "In gs::ClassId::ensureVersionInRange: expected version"
               << " number for class " << name_ << " to be in range ["
               <<  vmin << ", " << vmax << "], got " << version_;
            throw gs::IOInvalidArgument(os.str());
        }
    }

    bool ClassId::validatePrefix(const char* prefix)
    {
        // Prefix can not be an empty string
        if (prefix == NULL)
            return false;
        const unsigned len = strlen(prefix);
        if (len == 0)
            return false;

        // Characters '(' and ')' are special and can not be used
        // as parts of class names unless they enclose a version
        // number. Version number is an unsigned integer.
        bool inVersion = false;
        unsigned vstart = 0;
        for (unsigned i=0; i<len; ++i)
        {
            if (prefix[i] == '(')
            {
                // Can't have stacked parentheses.
                // Can't have '(' as the very first character.
                if (inVersion || i == 0)
                    return false;
                inVersion = true;
                vstart = i + 1;
            }
            else if (prefix[i] == ')')
            {
                // Can't have closing parentheses withoup opening ones
                if (!inVersion)
                    return false;
                inVersion = false;
                if (vstart >= i)
                    return false;
                char *endptr;
                // Compiler can complain about unused result of "strtoul"
                unsigned long dummy = strtoul(prefix+vstart, &endptr, 10);
                ++dummy;
                if (endptr != prefix+i)
                    return false;
            }
        }
        // Can't have missing closing parentheses
        if (inVersion)
            return false;

        return true;
    }

    void ClassId::initialize(const char* prefix, const unsigned version,
                             const bool isPtr)
    {
        std::ostringstream os;
        if (!validatePrefix(prefix))
        {
            if (prefix)
                os << "In gs::ClassId::initialize: bad class name prefix \""
                   << prefix << "\". Check for problematic parentheses.";
            else
                os << "In gs::ClassId::initialize: NULL class name prefix.";
            throw gs::IOInvalidArgument(os.str());
        }
        os << prefix << '(' << version << ')';
        if (isPtr)
            os << '*';
        id_ = os.str();
        version_ = version;
        isPtr_ = isPtr;
        makeName();
    }

    // Remove all version numbers from the class id
    bool ClassId::makeName()
    {
        char localbuf[NLOCAL];
        char* buf = localbuf;
        const unsigned idLen = id_.size();
        if (idLen+1U > NLOCAL)
            buf = new char[idLen+1U];
        const char* from = id_.data();
        bool inVersion = false;
        unsigned ito=0;
        for (unsigned ifrom=0; ifrom < idLen; ++ifrom)
        {
            if (from[ifrom] == '(')
            {
                if (inVersion) 
                {
                    if (buf != localbuf) delete [] buf;
                    return false;
                }
                inVersion = true;
            }
            else if (from[ifrom] == ')')
            {
                if (!inVersion)
                {
                    if (buf != localbuf) delete [] buf;
                    return false;
                }
                inVersion = false;
            }
            else if (!inVersion)
                buf[ito++] = from[ifrom];
        }
        if (inVersion)
        {
            if (buf != localbuf) delete [] buf;
            return false;
        }
        buf[ito] = '\0';
        name_ = buf;
        if (buf != localbuf) delete [] buf;
        return true;
    }

    // Parse the version number and pointer switch
    bool ClassId::makeVersion()
    {
        bool correct = false;
        const unsigned ns = id_.size();
        const char* const buf = id_.data();
        const char* sep = buf + (ns - 1U);
        if (*sep == '*')
        {
            isPtr_ = true;
            --sep;
        }
        else
            isPtr_ = false;
        if (*sep == ')')
        {
            const char* closingBrace = sep;
            for (; sep != buf; --sep)
                if (*sep == '(')
                    break;
            if (sep != buf)
            {
                char* endptr;
                version_ = strtoul(sep + 1, &endptr, 10);
                if (endptr > sep + 1 && endptr == closingBrace)
                    correct = true;
            }
        }
        return correct;
    }

    ClassId::ClassId(const std::string& id)
        : id_(id)
    {
        if (!(!id_.empty() && makeName() && makeVersion()))
        {
            std::ostringstream os;
            os << "In gs::ClassId::ClassId(const std::string&): "
               << "invalid input id string \"" << id_ << "\"";
            throw gs::IOInvalidArgument(os.str());
        }
    }

    ClassId::ClassId(std::istream& in, int)
    {
        read_pod(in, &id_);
        if (in.fail()) throw IOReadFailure(
            "In gs::ClassId::ClassId(std::istream&, int): "
            "input stream failure");

        if (!(!id_.empty() && makeName() && makeVersion()))
        {
            std::ostringstream os;
            os << "In gs::ClassId::ClassId(std::istream&, int): "
               << "read invalid id string \"" << id_ << "\"";
            throw IOInvalidData(os.str());
        }
    }

    bool ClassId::write(std::ostream& os) const
    {
        write_pod(os, id_);
        return !os.fail();
    }

    ClassId::ClassId()
        : name_(""), id_("(0)"), version_(0U), isPtr_(false)
    {
    }

    ClassId ClassId::invalidId()
    {
        ClassId dummy;
        return dummy;
    }

    bool ClassId::isTemplate() const
    {
        const std::size_t leftBrak = id_.find('<');
        const std::size_t rightBrak = id_.rfind('>');
        return leftBrak != std::string::npos && 
               rightBrak != std::string::npos && 
               leftBrak < rightBrak;
    }

    void ClassId::templateParameters(
        std::vector<std::vector<ClassId> >* params) const
    {
        assert(params);
        params->clear();
        const std::size_t leftBrak = id_.find('<');
        const std::size_t rightBrak = id_.rfind('>');
        if (leftBrak != std::string::npos && 
            rightBrak != std::string::npos && 
            leftBrak < rightBrak)
        {
            // Count commas and angle brackets
            unsigned ncommas = 0;
            int nbrackets = 0;
            for (std::size_t pos = leftBrak+1; pos<rightBrak; ++pos)
            {
                const char c = id_[pos];
                if (c == '<') ++nbrackets;
                else if (c == '>') --nbrackets;
                else if (c == ',' && nbrackets == 0)
                    ++ncommas;
            }

            // Must be a well-formed name
            if (nbrackets)
            {
                std::ostringstream os;
                os << "In gs::ClassId::templateParameters: "
                   << "unbalanced angle brackets in the "
                   << "template id \"" << id_ << "\"";
                throw gs::IOInvalidArgument(os.str());
            }

            // Reserve a proper size vector
            params->resize(ncommas + 1);
            for (unsigned i=0; i<=ncommas; ++i)
                (*params)[i].reserve(1);

            // Cycle over commas again and fill the ids
            ncommas = 0;
            nbrackets = 0;
            std::size_t begin = leftBrak + 1;
            for (std::size_t pos = begin; pos<rightBrak; ++pos)
            {
                const char c = id_[pos];
                if (c == '<') ++nbrackets;
                else if (c == '>') --nbrackets;
                else if (c == ',' && nbrackets == 0)
                {
                    while (isspace(id_[begin])) ++begin;
                    std::size_t end = pos - 1;
                    while (isspace(id_[end])) --end;
                    ++end;
                    (*params)[ncommas].push_back(
                        ClassId(id_.substr(begin, end - begin)));
                    begin = pos + 1;
                    ++ncommas;
                }
            }
            while (isspace(id_[begin])) ++begin;
            std::size_t end = rightBrak - 1;
            while (isspace(id_[end])) --end;
            ++end;
            (*params)[ncommas].push_back(
                ClassId(id_.substr(begin, end - begin)));
        }
    }
}
