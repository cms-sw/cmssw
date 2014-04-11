#ifndef CommonAnalyzer_h
#define CommonAnalyzer_h

#include <string>
#include <vector>

class TFile;
class TObject;
class TNamed;
class TH1F;

class CommonAnalyzer {

 public:
  CommonAnalyzer(TFile* file, const char* run, const char* mod, const char* path ="", const char* prefix="");
  CommonAnalyzer(const CommonAnalyzer& dtca);

  CommonAnalyzer& operator=(const CommonAnalyzer& dtca);

  void setRunNumber(const char* run);
  void setFile(TFile* file);
  void setModule(const char* mod);
  void setPath(const char* path);
  void setPrefix(const char* prefix);

  const std::string& getRunNumber() const;
  const std::string& getModule() const;
  const std::string& getPath() const;
  const std::string& getPrefix() const;

  const std::vector<unsigned int> getRunList() const;

  TObject* getObject(const char* name) const;
  TNamed* getObjectWithSuffix(const char* name, const char* suffix="") const;

  TH1F* getBinomialRatio(const CommonAnalyzer& denom, const char* name, const int rebin=-1) const;

 private:


  TFile* _file;
  std::string _runnumber;
  std::string _module;
  std::string _path;
  std::string _prefix;

};



#endif // CommonAnalyzer_h
