#ifndef CondCore_Utilities_CommonOptions_h
#define CondCore_Utilities_CommonOptions_h

#include <boost/program_options.hpp>

namespace cond{
  class CommonOptions{
  public:
    explicit CommonOptions( const std::string& commandname);
    ~CommonOptions();
    void addAuthentication(const bool withEnvironmentAuth=true);
    void addConnect();
    void addLogDB();
    void addDictionary();
    void addFileConfig();
    void addBlobStreamer();
    
    boost::program_options::options_description& description();
    boost::program_options::options_description& visibles();

  private:
    std::string m_name;
    boost::program_options::options_description* m_description;
    boost::program_options::options_description* m_visible;
  };
}
#endif
