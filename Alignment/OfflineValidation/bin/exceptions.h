#ifndef __EXCEPTIONS__
#define __EXCEPTIONS__

#include <cstdlib>
#include <iostream>
#include <exception>
#include <string>

#include <boost/exception/exception.hpp>
#include <boost/exception/diagnostic_information.hpp> 

#include <boost/property_tree/exceptions.hpp>
#include <boost/filesystem.hpp>
//#include <boost/program_options/errors.hpp>

// TODO: utilise the exceptions provided by boost as much as possible

using std::cerr;

namespace AllInOneConfig {

class ConfigError : public std::exception {
    const char * msg;
    virtual const char * what () const throw()
    {
        return msg;
    }
public:
    ConfigError (const char * m) : msg(m) {}
};

inline std::string colorify (std::string s)
{
    const char * red = "\x1B[31m"/*, * green = "\x1B[32m"*/ /*, * black = "\x1B[30m"*/;
    return red + s /*+ black*/; // TODO: clarify colours...
}

template<int FUNC(int, char **)> int exceptions (int argc, char * argv[])
{
    try { return FUNC(argc, argv); }
    //catch (const AllInOneConfig::ConfigError & e) { cerr << "Config error: " << e.what() << '\n'; }
    catch (const boost::exception & e) { cerr << colorify("Boost exception: ") << boost::diagnostic_information(e); throw; }
    //catch (const boost::program_options::required_option &e) {
    //    if (e.get_option_name() == "--config") cerr << colorify("Missing config") << '\n';
    //    else cerr << colorify("Program Options Required Option: ") << e.what() << '\n';
    //}
    //catch (const boost::program_options::invalid_syntax & e) { cerr << colorify("Program Options Invalid Syntax: ") << e.what() << '\n'; }
    //catch (const boost::program_options::error          & e) { cerr << colorify("Program Options Error: ") << e.what() << '\n'; }
    catch (const boost::property_tree::ptree_bad_data & e) { cerr << colorify("Property Tree Bad Data Error: ") << e.data<std::string>() << '\n'; }
    catch (const boost::property_tree::ptree_bad_path & e) { cerr << colorify("Property Tree Bad Path Error: ") << e.path<std::string>() << '\n'; }
    catch (const boost::property_tree::ptree_error    & e) { cerr << colorify("Property Tree Error: ") << e.what() << '\n'; }
    catch (const boost::filesystem::filesystem_error & e) { cerr << colorify("Filesystem Error:" ) << e.what() << '\n'; }
    catch (const std::logic_error & e) { cerr << colorify("Logic Error: ")    << e.what() << '\n'; }
    catch (const std::exception   & e) { cerr << colorify("Standard Error: ") << e.what() << '\n'; }
    catch (...) { cerr << colorify("Unkown failure\n"); }
    return EXIT_FAILURE;
}

}
#endif
