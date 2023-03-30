#ifndef __EXCEPTIONS__
#define __EXCEPTIONS__

#include <cstdlib>
#include <iostream>
#include <exception>
#include <string>

#include "boost/exception/exception.hpp"
#include "boost/exception/diagnostic_information.hpp"

#include "boost/property_tree/exceptions.hpp"
#include "boost/filesystem.hpp"
//#include "boost/program_options/errors.hpp"

// TODO: utilise the exceptions provided by boost as much as possible

namespace AllInOneConfig {

  class ConfigError : public std::exception {
    const char *msg;
    const char *what() const throw() override { return msg; }

  public:
    ConfigError(const char *m) : msg(m) {}
  };

  inline std::string colorify(std::string s) {
    const char *red = "\x1B[31m" /*, * green = "\x1B[32m"*/ /*, * black = "\x1B[30m"*/;
    return red + s /*+ black*/;  // TODO: clarify colours...
  }

  template <int FUNC(int, char **)>
  int exceptions(int argc, char *argv[]) {
    try {
      return FUNC(argc, argv);
    } catch (const boost::exception &e) {
      std::cerr << colorify("Boost exception: ") << boost::diagnostic_information(e);
      throw;
    } catch (const boost::property_tree::ptree_bad_data &e) {
      std::cerr << colorify("Property Tree Bad Data Error: ") << e.data<std::string>() << '\n';
    } catch (const boost::property_tree::ptree_bad_path &e) {
      std::cerr << colorify("Property Tree Bad Path Error: ") << e.path<std::string>() << '\n';
    } catch (const boost::property_tree::ptree_error &e) {
      std::cerr << colorify("Property Tree Error: ") << e.what() << '\n';
    } catch (const boost::filesystem::filesystem_error &e) {
      std::cerr << colorify("Filesystem Error:") << e.what() << '\n';
    } catch (const std::logic_error &e) {
      std::cerr << colorify("Logic Error: ") << e.what() << '\n';
    } catch (const std::exception &e) {
      std::cerr << colorify("Standard Error: ") << e.what() << '\n';
    }
    return EXIT_FAILURE;
  }

}  // namespace AllInOneConfig
#endif
