#ifndef GEMCode_GEMValidation_BarrelTriggerPtAssignmentHelper_h
#define GEMCode_GEMValidation_BarrelTriggerPtAssignmentHelper_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>
#include <map>
#include <string>

namespace BarrelTriggerPtAssignmentHelper {

enum DT_Types_2stubs {DT1_DT2, DT1_DT3, DT1_DT4, DT2_DT3, DT2_DT4, DT3_DT4};
enum DT_Types_3or4stubs {DT1_DT2__DT1_DT3, DT1_DT2__DT1_DT4, DT1_DT2__DT2_DT3, DT1_DT2__DT2_DT4,
                         DT1_DT3__DT1_DT4, DT1_DT3__DT2_DT3, DT1_DT3__DT3_DT4,
                         DT1_DT4__DT2_DT4, DT1_DT4__DT3_DT4,
                         DT2_DT3__DT2_DT4, DT2_DT3__DT3_DT4,
                         DT3_DT4__DT3_DT4,
                         DT1_DT2__DT3_DT4, DT1_DT3__DT2_DT4, DT1_DT4__DT2_DT3};

const std::vector<std::string> DT_Types_2stubs_string = {
  "DT1_DT2", "DT1_DT3", "DT1_DT4", "DT2_DT3", "DT2_DT4", "DT3_DT4"};

const std::vector<std::string> DT_Types_3or4stubs_string = {
  "DT1_DT2__DT1_DT3", "DT1_DT2__DT1_DT4", "DT1_DT2__DT2_DT3", "DT1_DT2__DT2_DT4",
  "DT1_DT3__DT1_DT4", "DT1_DT3__DT2_DT3", "DT1_DT3__DT3_DT4",
  "DT1_DT4__DT2_DT4", "DT1_DT4__DT3_DT4",
  "DT2_DT3__DT2_DT4", "DT2_DT3__DT3_DT4",
  "DT3_DT4__DT3_DT4",
  "DT1_DT2__DT3_DT4", "DT1_DT3__DT2_DT4", "DT1_DT4__DT2_DT3"};

float getDirectionPt2Stubs(float DPhi, int DT_type);
float getDirectionPt2Stubs(float DPhi, std::string DT_type);

float getDirectionPt3or4Stubs(float DPhi1, float DPhi2, int DT_type);
float getDirectionPt3or4Stubs(float DPhi1, float DPhi2, std::string DT_type);

float getEllipse(float x, float y, float a, float b, float alpha, float x0=0, float y0=0);
bool passEllipse(float x, float y, float a, float b, float alpha, float x0=0, float y0=0);
bool failEllipse(float x, float y, float a, float b, float alpha, float x0=0, float y0=0);

std::string getBestDPhiPair(int station1, int station2, int station3);
std::string getBestDPhiPair(int station1, int station2, int station3, int station4);

}

#endif

