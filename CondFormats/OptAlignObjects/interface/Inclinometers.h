#ifndef INCLINOMETERS_H
#define INCLINOMETERS_H
#include<vector>
#include<string> 
class Inclinometers {
public:
  struct Item {
    public: 
    std::string Sensor_type;
    int Sensor_number;
    std::string ME_layer;
    std::string Logical_Alignment_Name;
    std::string CERN_Designator;
    std::string CERN_Barcode;
    std::string  Inclination_Direction;
    float Abs_Slope;
    float Abs_Slope_Error;
    float Norm_Slope;
    float Norm_Slope_Error;
    float Abs_Intercept;
    float Abs_Intercept_Error;
    float Norm_Intercept;
    float Norm_Intercept_Error;
    float Shifts_due_to_shims_etc;
  };
  Inclinometers();
  virtual ~Inclinometers();
  std::vector<Item>  m_inclinometers;
};
#endif
