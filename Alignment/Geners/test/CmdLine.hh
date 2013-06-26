//=========================================================================
// CmdLine.hh
//
// Simple command line parser for the C++ "main" program. It provides
// functionality of "getopt" and "getopt_long" with a convenient interface.
// Typical usage is as follows:
//
// #include <necessary standard headers>
//
// #include "CmdLine.hh"
//
// int main(int argc, char *argv[])
// {
//     CmdLine cmdline(argc, argv);
//
//     int i = 0;
//     double d = 0;
//     bool b = false;
//     std::string requiredOption;
//     std::vector<std::string> positionalArgs;
//
//     try {
//         /* Arguments are short and long versions of the option name. */
//         /* Long version can be omitted.                              */
//         cmdline.option("-i", "--integer") >> i;
//         cmdline.option("-d") >> d;
//
//         /* Options that must be present on the command line  */
//         cmdline.require("-r", "--required") >> requiredOption;
//
//         /* Switches that do not require additional arguments */
//         b = cmdline.has("-b");
//
//         /* Declare the end of option processing. Unconsumed options  */
//         /* will cause an exception to be thrown.                     */
//         cmdline.optend();
//
//         /* Process all remaining arguments */ 
//         while (cmdline)
//         {
//             std::string s;
//             cmdline >> s;
//             positionalArgs.push_back(s);
//         }
//     }
//     catch (const CmdLineError& e) {
//         std::cerr << "Error in " << cmdline.progname() << ": "
//                   << e.str() << std::endl;
//         return 1;
//     }
//
//     /* ..... Do main processing here ..... */
//
//     return 0;
// }    
//
// Note that each of the parsing methods of the CmdLine class ("option",
// "has", and "require") is greedy. These methods will consume all
// corresponding options or switches and will set the result to the last
// option value seen. It is therefore impossible to provide a collection
// of values by using an option more than once. This is done in order to
// avoid difficulties with deciding what to do when multiple option values
// were consumed only partially.
//
// After the "optend()" call, the "argc()" method of the CmdLine 
// class can be used to determine the number of remaining program
// arguments. If the expected number of arguments is known in advance,
// the simplest way to get the arguments out is like this (should be
// inside the "try" block):
//
// if (cmdline.argc() != n_expected)
//     throw CmdLineError("wrong number of command line arguments");
// cmdline >> arg0 >> arg1 >> ...;
//
//
// I. Volobouev
// July 2011
//=========================================================================

#ifndef CMDLINE_HH_
#define CMDLINE_HH_

#include <list>
#include <sstream>
#include <cstring>
#include <utility>
#include <cstdio>

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"

// Subsequent classes will throw exceptions of the following class
class CmdLineError
{
public:
    inline CmdLineError(const char* msg = 0)
        : os_(new std::ostringstream()) {if (msg) *os_ << msg;}

    template<typename T>
    inline CmdLineError& operator<<(const T& obj)
    {
        *os_ << obj;
        return *this;
    }

    inline std::string str() const {return os_->str();}

private:
    CPP11_shared_ptr<std::ostringstream> os_;
};

template<typename T>
inline void OneShotExtract(std::istringstream& is, T& obj)
{is >> obj;}

template<>
inline void OneShotExtract<std::string>(std::istringstream& is,
                                        std::string& obj)
{obj = is.str(); is.seekg(0, std::ios_base::end);}

class OneShotIStream
{
public:
    inline OneShotIStream() : valid_(false), readout_(false) {}

    inline OneShotIStream(const std::string& s)
        : str_(s), valid_(true), readout_(false) {}

    inline operator void*() const
    {
        return valid_ && !readout_ ? (void*)this : (void*)0;
    }

    template<typename T>
    inline bool operator>>(T& obj)
    {
        if (readout_)
            throw CmdLineError()
                << "can't reuse command line argument \"" << str_ << '"';
        readout_ = true;
        if (valid_)
        {
            std::istringstream is(str_);
            OneShotExtract(is, obj);
            if (is.bad() || is.fail())
                throw CmdLineError()
                    << "failed to parse command line argument \""
                    << str_ << '"';
            if (is.peek() != EOF)
                throw CmdLineError()
                    << "extra characters in command line argument \""
                    << str_ << '"';
        }
        return valid_;
    }

    inline bool isValid() const {return valid_;}

private:
    std::string str_;
    bool valid_;
    bool readout_;
};

class CmdLine
{
    // Argument codes (second member of the pair):
    //   0 -- possible option value (or program argument, not yet known which)
    //   1 -- short option switch
    //   2 -- long option switch
    //   3 -- program argument
    typedef std::pair<std::string,int> Pair;
    typedef std::list<Pair> Optlist;

    inline Optlist::iterator find(const char* shortOpt, const char* longOpt)
    {
        Optlist::iterator iend = args_.end();
        for (Optlist::iterator it = args_.begin(); it != iend; ++it)
        {
            if (shortOpt && it->second == 1 && it->first == shortOpt)
                return it;
            if (longOpt && it->second == 2 && it->first == longOpt)
                return it;
        }
        return iend;
    }

public:
    inline CmdLine(const unsigned argc, const char* const argv[])
        : nprogargs_(0)
    {
        // Parse the program name
        const char *progname = std::strrchr(argv[0], '/');
        if (progname)
            ++progname;
        else
            progname = argv[0];

        // Take into account program name mangling by GNU autotools
        if (strncmp(progname, "lt-", 3) == 0)
            progname += 3;
        progname_ = progname;

        // Make a list of arguments noting on the way if this is
        // a short option, long option, or possible option argument
        bool previousIsOpt = false;
        for (unsigned i=1; i<argc; ++i)
        {
            if (strcmp(argv[i], "-") == 0)
            {
                args_.push_back(Pair(argv[i], previousIsOpt ? 0 : 3));
                previousIsOpt = false;
                ++nprogargs_;
            }
            else if (strcmp(argv[i], "--") == 0)
            {
                // End of options
                for (unsigned k=i+1; k<argc; ++k)
                {
                    args_.push_back(Pair(argv[k], 3));
                    ++nprogargs_;
                }
                return;
            }
            else if (strncmp(argv[i], "--", 2) == 0)
            {
                args_.push_back(Pair(argv[i], 2));
                previousIsOpt = true;
            }
            else if (argv[i][0] == '-')
            {
                const unsigned len = strlen(argv[i]);
                for (unsigned k=1; k<len; ++k)
                {
                    std::string dummy("-");
                    dummy += argv[i][k];
                    args_.push_back(Pair(dummy, 1));
                    previousIsOpt = true;
                }
            }
            else
            {
                args_.push_back(Pair(argv[i], previousIsOpt ? 0 : 3));
                previousIsOpt = false;
                ++nprogargs_;
            }
        }
    }

    inline const char* progname() const {return progname_.c_str();}

    inline bool has(const char* shortOpt, const char* longOpt=0)
    {
        bool found = false;
        for (Optlist::iterator it = find(shortOpt, longOpt);
             it != args_.end(); it = find(shortOpt, longOpt))
        {
            found = true;
            Optlist::iterator it0(it);
            if (++it != args_.end())
                if (it->second == 0)
                    it->second = 3;
            args_.erase(it0);
        }
        return found;
    }

    inline OneShotIStream option(const char* shortOpt, const char* longOpt=0)
    {
        OneShotIStream result;
        for (Optlist::iterator it = find(shortOpt, longOpt);
             it != args_.end(); it = find(shortOpt, longOpt))
        {
            Optlist::iterator it0(it);
            if (++it != args_.end())
                if (it->second == 0)
                {
                    result = OneShotIStream(it->first);
                    args_.erase(it0, ++it);
                    --nprogargs_;
                    continue;
                }
            throw CmdLineError()
                << "missing command line argument for option \""
                << it0->first << '"';
        }
        return result;
    }

    inline OneShotIStream require(const char* shortOpt, const char* longOpt=0)
    {
        const OneShotIStream& is(option(shortOpt, longOpt));
        if (!is.isValid())
        {
            const char empty[] = "";
            const char* s = shortOpt ? shortOpt : (longOpt ? longOpt : empty);
            throw CmdLineError()
                << "required command line option \"" << s << "\" is missing";
        }
        return is;
    }

    inline void optend() const
    {
        for (Optlist::const_iterator it = args_.begin();
             it != args_.end(); ++it)
            if (it->second == 1 || it->second == 2)
                throw CmdLineError("invalid command line option \"")
                    << it->first << '"';
    }

    inline operator void*() const
    {
        return (void*)(static_cast<unsigned long>(nprogargs_));
    }

    inline unsigned argc() const
    {
        return nprogargs_;
    }

    template<typename T>
    inline CmdLine& operator>>(T& obj)
    {
        if (!nprogargs_)
            throw CmdLineError("no more input available on the command line");
        Optlist::iterator it = args_.begin();
        for (; it != args_.end(); ++it)
            if (it->second == 0 || it->second == 3)
                break;
        OneShotIStream is(it->first);
        args_.erase(it);
        --nprogargs_;
        is >> obj;
        return *this;
    }

private:
    CmdLine();

    std::string progname_;
    Optlist args_;
    unsigned nprogargs_;
};

#endif // CMDLINE_HH_
