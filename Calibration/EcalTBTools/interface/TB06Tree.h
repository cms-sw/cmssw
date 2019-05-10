#ifndef TB06Tree_h
#define TB06Tree_h

// includes
#include <string>

#include "TClonesArray.h"

class TFile;
class TTree;

class G3EventProxy;

class TB06Tree {
public:
  //! ctor
  TB06Tree(const std::string &fileName = "TB06Tree.root", const std::string &treeName = "Analysis");
  //! dtor
  ~TB06Tree();

  void store(const int &tableIsMoving,
             const int &run,
             const int &event,
             const int &S6adc,
             const double &xhodo,
             const double &yhodo,
             const double &xslope,
             const double &yslope,
             const double &xquality,
             const double &yquality,
             const int &icMax,
             const int &ietaMax,
             const int &iphiMax,
             const double &beamEnergy,
             const double ampl[49]);

  void reset(float crystal[11][21]);

  void check();

private:
  TFile *m_file;
  TTree *m_tree;

  TClonesArray *m_data;
  int m_dataSize;
};

#endif
