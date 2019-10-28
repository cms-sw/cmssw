#ifndef COMMONTOOLS_RECOALGOS_MVACOMPUTER
#define COMMONTOOLS_RECOALGOS_MVACOMPUTER
#include <memory>
#include <string>
#include <vector>
#include <tuple>

#include <TMVA/Reader.h>

class MVAComputer {
public:
  typedef std::vector<std::tuple<std::string, float> > mva_variables;

  //--ctros---
  MVAComputer(){};
  MVAComputer(mva_variables* vars, std::string weights_file);

  //---dtor---
  ~MVAComputer(){};

  //---getters---
  float operator()();
  MVAComputer& operator=(MVAComputer&& other);

private:
  std::unique_ptr<TMVA::Reader> reader_;
  mva_variables* vars_;
};
#endif
