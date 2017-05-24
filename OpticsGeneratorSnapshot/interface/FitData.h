/*********************************************************
* $Id: RPXMLConfig.h,v 1.3 2006/08/17 15:54:21 lgrzanka Exp $
* $Revision: 1.3 $
* $Date: 2006/08/17 15:54:21 $
**********************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TTree.h"
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>


class FitData{

public:

enum VarName {
      kX,
      kPX,
      kY,
      kPY,
      kPT
};

struct lost_particle
{
  int part_id;
  Double_t x1, px1, y1, py1, pt1, s;
  std::string element;
};


struct inter_plane_data
{
  TMatrixD *data;
  std::string x_lab;
  std::string theta_x_lab;
  std::string y_lab;
  std::string theta_y_lab;
  std::string ksi_lab;
  std::string s_lab;
  std::string valid_lab;
  int cur_index;

  double x, theta_x, y, theta_y, ksi, s, valid;
};


FitData();
~FitData();

void readIn(std::string name);  //reads generated input samples
void writeIn(std::string name);
void readLost(std::string name);  //reads lost particles file
int readOut(std::string name, std::string section);  //reads transported particles
void readAdditionalScoringPlane(std::string name, const std::string &section);  //reads additional scoring plane
void readAdditionalScoringPlanes(std::string name, const std::vector<std::string> &sections);  //reads additional scoring plane

void writeOut(std::string name);
int AppendRootFile(TTree *inp_tree, std::string data_prefix = std::string("def"));
int AppendLostParticlesRootFile(TTree *lost_particles_tree);
int AppendAcceleratorAcceptanceRootFile(TTree *acceptance_tree);

TVectorD getSampleIn(int n);
Double_t getOutVar(VarName id, int nr);
TVectorD getOutVector(VarName id);
TVectorD getInVector(VarName id);
Int_t getInSize();
Int_t getOutSize();

private:

TMatrixD dataIn;
TMatrixD dataOut;
std::vector<lost_particle> dataLost;
std::map<std::string, TMatrixD> additional_scoring_planes;

Int_t inSize;
Int_t outSize;
Int_t lostSize;
};
