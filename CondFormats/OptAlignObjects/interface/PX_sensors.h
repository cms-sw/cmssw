#ifndef PX_SENSORS_H
#define PX_SENSORS_H
#include<vector>
#include<string>
class PX_sensors {
public:
  struct Item{
    std::string Sensor_type;
    int Sensor_number;
    std::string ME_layer;
    std::string Logical_Alignment_Name;
    std::string CERN_Designator;
    std::string CERN_Barcode;
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
  PX_sensors();
  virtual ~PX_sensors();
  std::vector<Item>  m_PX_sensors;
};
#endif
