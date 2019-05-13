#ifndef TB06TreeH2_h
#define TB06TreeH2_h

// includes
#include <string>

#include "TClonesArray.h"

class TFile;
class TTree;

class G3EventProxy;

class TB06TreeH2 {
public:
  //! ctor
  TB06TreeH2(const std::string &fileName = "TB06Tree.root", const std::string &treeName = "Analysis");
  //! dtor
  ~TB06TreeH2();

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
             const double ampl[49],
             const int &wcAXo,
             const int &wcAYo,
             const int &wcBXo,
             const int &wcBYo,
             const int &wcCXo,
             const int &wcCYo,
             const double &xwA,
             const double &ywA,
             const double &xwB,
             const double &ywB,
             const double &xwC,
             const double &ywC,
             const float &S1adc,
             const float &S2adc,
             const float &S3adc,
             const float &S4adc,
             const float &VM1,
             const float &VM2,
             const float &VM3,
             const float &VM4,
             const float &VM5,
             const float &VM6,
             const float &VM7,
             const float &VM8,
             const float &VMF,
             const float &VMB,
             const float &CK1,
             const float &CK2,
             const float &CK3,
             const float &BH1,
             const float &BH2,
             const float &BH3,
             const float &BH4,
             const float &TOF1S,
             const float &TOF2S,
             const float &TOF1J,
             const float &TOF2J);

  void reset(float crystal[11][21]);

  void check();

private:
  TFile *m_file;
  TTree *m_tree;

  TClonesArray *m_data;
  int m_dataSize;
};

#endif
