#include "L1Trigger/DTTriggerPhase2/interface/GlobalLutObtainer.h"

using namespace edm;

// ============================================================================
// Constructors and destructor
// ============================================================================
GlobalLutObtainer::GlobalLutObtainer(const ParameterSet& pset) {
  global_coords_filename_ = pset.getParameter<edm::FileInPath>("global_coords_filename");
  std::ifstream ifin3(global_coords_filename_.fullPath());
  
  if (ifin3.fail()) {
   throw cms::Exception("Missing Input File")
        << "GlobalLutObtainer::GlobalLutObtainer() -  Cannot find " << global_coords_filename_.fullPath();
  }

  int wh, st, se, sl;
  double perp, x_phi0;
  std::string line;
  
  global_constant_per_sl sl1_constants; 
  global_constant_per_sl sl3_constants; 
  
  while (ifin3.good()) {
    ifin3 >> wh >> st >> se >> sl >> perp >> x_phi0;
    
    if (sl == 1) {
      sl1_constants.perp = perp;
      sl1_constants.x_phi0 = x_phi0;
    }
    else {
      sl3_constants.perp = perp;
      sl3_constants.x_phi0 = x_phi0;
     
      DTChamberId ChId(wh, st, se);
      global_constants.push_back({ChId.rawId(), sl1_constants, sl3_constants});
    }
  }
}

GlobalLutObtainer::~GlobalLutObtainer() {}

void GlobalLutObtainer::generate_luts() {
  for (auto &global_constant: global_constants){
    
    int sgn; 
    DTChamberId ChId(global_constant.chid);
    // typical hasPosRF function
    if (ChId.wheel() > 0 || (ChId.wheel() == 0 && ChId.sector() % 4 > 1)) {
      sgn = 1;
    } else sgn = -1;
    
    auto phi1 = calc_atan_lut(12, 6, (1. / 16) / (global_constant.sl1.perp * 10),
      global_constant.sl1.x_phi0/global_constant.sl1.perp, 1./std::pow(2, 17), 10, 3, 12, 20, sgn);
      
    auto phi3 = calc_atan_lut(12, 6, (1. / 16) / (global_constant.sl3.perp * 10),
      global_constant.sl3.x_phi0/global_constant.sl3.perp, 1./std::pow(2, 17), 10, 3, 12, 20, sgn);

    double max_x_phi0 = global_constant.sl1.x_phi0;
    if (global_constant.sl3.x_phi0 > max_x_phi0) {
      max_x_phi0 = global_constant.sl3.x_phi0 ;
    }
    
    auto phic = calc_atan_lut(12, 6, (1. / 16) / ((global_constant.sl1.perp + global_constant.sl3.perp)/.2),
      max_x_phi0 / ((global_constant.sl1.perp + global_constant.sl3.perp)/2), 1. / std::pow(2, 17), 10, 3, 12, 20, sgn);


    auto phib = calc_atan_lut(9, 6, 1./4096, 0., 4./pow(2, 13), 10, 3, 10, 16, sgn);
    
    luts[global_constant.chid] = {phi1, phi3, phic, phib};
  }
}

std::map <int, lut_value> GlobalLutObtainer::calc_atan_lut(int msb_num, int lsb_num, double in_res, double abscissa_0,
    double out_res, int a_extra_bits, int b_extra_bits, int a_size, int b_size, int sgn) {
  
  // Calculates the coefficients needed to calculate the arc-tan function in fw
  // by doing a piece-wise linear approximation.
  // In fw, input (x) and output (y) are integers, these conversions are needed
  //   t = x*in_res - abscissa_0
  //   phi = arctan(t)
  //   y = phi/out_res
  //   => y = arctan(x*in_res - abcissa_0)/out_res
  // The linear function is approximated as
  //   y = a*x_lsb + b
  // Where a, b = func(x_msb) are the coefficients stored in the lut
  
  // a is stored as unsigned, b as signed, with their respective sizes a_size, b_size,
  // previously shifted left by a_extra_bits and b_extra_bits, respectively
  
  
  long int a_min = - std::pow(2, (a_size - 1));
  long int a_max =   std::pow(2, (a_size - 1)) - 1;
  long int b_min = - std::pow(2, (b_size - 1));
  long int b_max =   std::pow(2, (b_size - 1)) - 1;
  
  std::map <int, lut_value> lut;
  
  for (int x_msb = - std::pow(2, msb_num - 1); x_msb < std::pow(2, msb_num - 1); x_msb++) {
    int x1 = ((x_msb    ) << lsb_num);
    int x2 = ((x_msb + 1) << lsb_num) - 1;

    double t1 = x1 * in_res - abscissa_0;
    double t2 = x2 * in_res - abscissa_0;

    double phi1 = sgn * atan(t1);
    double phi2 = sgn * atan(t2);
    
    double y1 = phi1 / out_res;
    double y2 = phi2 / out_res;

    // we want to find a, b so that the error in the extremes is the same as the error in the center
    // so the error in the extremes will be the same, so the "a" is determined by those points
    double a = (y2 - y1) / (x2 - x1);

    // by derivating the error function and equaling to 0, you get this is the point
    // towards the interval center with the highest error
    // err_f = y - (a*x+b) = sgn*arctan(x*in_res - abcissa_0)/out_res - (a*x+b)
    // d(err_f)/dx = sgn*(1/(1+(x*in_res - abcissa_0)^2))*in_res/out_res - a
    // d(err_f)/dx = 0 => x_max_err = (sqrt(in_res/out_res/a-1) + abscissa_0)/in_res
    // There is sign ambiguity in the sqrt operation. The sqrt has units of t (adimensional).
    // It is resolved by setting the sqrt to have the same sign as (t1+t2)/2
  
    double t_max_err = sqrt(sgn * in_res/out_res/a - 1);
    if ((t1 + t2) < 0) {
      t_max_err *= -1;
    }
    
    double x_max_err = (t_max_err + abscissa_0) / in_res;
    double phi_max_err = sgn * atan(t_max_err);
    double y_max_err = phi_max_err / out_res;
    
    // once you have the point of max error, the "b" parameter is chosen as the average between
    // those two numbers, which makes the error at the center be equal in absolute value
    // to the error in the extremes
    // units: rad
    
    double b = (y1 + y_max_err - a * (x_max_err-x1))/2;
    
    // increase b in 1/2 of y_lsb, so that fw truncate operation on the of the result 
    // is equivalent to a round function instead of a floor function
    b += 0.5;
    
    // shift left and round
    long int a_int = (long int)(round(a * (pow(2, a_extra_bits))));
    long int b_int = (long int)(round(b * (pow(2, b_extra_bits))));
    
    // tuck a, b constants into the bit size of the output (un)signed integer
    std::vector <long int> as = {a_min, a_int, a_max}; 
    std::vector <long int> bs = {b_min, b_int, b_max}; 
    
    std::sort (as.begin(), as.end());
    std::sort (bs.begin(), bs.end());
    
    a_int = as.at(1);
    b_int = bs.at(1);

    // convert a, b to two's complement
    auto a_signed = a_int % (long int) (pow(2, a_size));
    auto b_signed = b_int % (long int) (pow(2, b_size));

    // convert x_msb to two's complement signed
    lut[(int) (x_msb % (long int) (pow(2, msb_num)))] = {a_signed, b_signed};
  }
  return lut;      
}