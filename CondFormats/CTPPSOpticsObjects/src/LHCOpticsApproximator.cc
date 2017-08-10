#include "CondFormats/CTPPSOpticsObjects/interface/LHCOpticsApproximator.h"

#include <vector>
#include <iostream>
#include "TROOT.h"
#include "TMath.h"
#include <boost/shared_ptr.hpp>

ClassImp(LHCOpticsApproximator);
ClassImp(LHCApertureApproximator);


void LHCOpticsApproximator::Init()
{
  out_polynomials.clear();
  apertures_.clear();
  out_polynomials.push_back(&x_parametrisation);
  out_polynomials.push_back(&theta_x_parametrisation);
  out_polynomials.push_back(&y_parametrisation);
  out_polynomials.push_back(&theta_y_parametrisation);

  coord_names.clear();
  coord_names.push_back("x");
  coord_names.push_back("theta_x");
  coord_names.push_back("y");
  coord_names.push_back("theta_y");
  coord_names.push_back("ksi");

  s_begin_ = 0.0;
  s_end_ = 0.0;
  trained_ = false;
}


LHCOpticsApproximator::LHCOpticsApproximator(std::string name, std::string title, TMultiDimFet::EMDFPolyType polynom_type, std::string beam_direction, double nominal_beam_energy)
: x_parametrisation(5, polynom_type, "k"),
  theta_x_parametrisation(5, polynom_type, "k"),
  y_parametrisation(5, polynom_type, "k"),
  theta_y_parametrisation(5, polynom_type, "k")
{
//  std::cout<<"LHCOpticsApproximator(std::string name, std::string title, TMultiDimFet::EMDFPolyType polynom_type) entered"<<std::endl;
  this->SetName(name.c_str());
  this->SetTitle(title.c_str());
  Init();

  if(beam_direction == "lhcb1")
    beam = lhcb1;
  else if(beam_direction == "lhcb2")
    beam = lhcb2;
  else
    beam = lhcb1;

  nominal_beam_energy_ = nominal_beam_energy;
  nominal_beam_momentum_ = TMath::Sqrt(nominal_beam_energy_*nominal_beam_energy_ - 0.938272029*0.938272029);
//  std::cout<<"LHCOpticsApproximator(std::string name, std::string title, TMultiDimFet::EMDFPolyType polynom_type) left"<<std::endl;
}


LHCOpticsApproximator::LHCOpticsApproximator()
{
//  std::cout<<"LHCOpticsApproximator::LHCOpticsApproximator() entered"<<std::endl;
  Init();
  beam = lhcb1;
  nominal_beam_energy_ = 7000;
  nominal_beam_momentum_ = TMath::Sqrt(nominal_beam_energy_*nominal_beam_energy_ - 0.938272029*0.938272029);
//  std::cout<<"LHCOpticsApproximator::LHCOpticsApproximator() left"<<std::endl;
}



bool LHCOpticsApproximator::Transport(double *in, double *out, bool check_apertures, bool invert_beam_coord_sytems)
{
  if (in == NULL || out == NULL || !trained_)
    return false;

  bool res = CheckInputRange(in);

  double in_corrected[5];

  if (beam == lhcb1 || !invert_beam_coord_sytems)
  {
    in_corrected[0] = in[0];
    in_corrected[1] = in[1];
    in_corrected[2] = in[2];
    in_corrected[3] = in[3];
    in_corrected[4] = in[4];
    out[0] = x_parametrisation.Eval(in_corrected);
    out[1] = theta_x_parametrisation.Eval(in_corrected);
    out[2] = y_parametrisation.Eval(in_corrected);
    out[3] = theta_y_parametrisation.Eval(in_corrected);
    out[4] = in[4];
  }
  else
  {
    in_corrected[0] = -in[0];
    in_corrected[1] = -in[1];
    in_corrected[2] = in[2];
    in_corrected[3] = in[3];
    in_corrected[4] = in[4];
    out[0] = -x_parametrisation.Eval(in_corrected);
    out[1] = -theta_x_parametrisation.Eval(in_corrected);
    out[2] = y_parametrisation.Eval(in_corrected);
    out[3] = theta_y_parametrisation.Eval(in_corrected);
    out[4] = in[4];
  }

  if (check_apertures)
  {
    for(unsigned int i=0; i<apertures_.size(); i++)
    {
      res = res && apertures_[i].CheckAperture(in);
    }
  }

  return res;
}



bool LHCOpticsApproximator::Transport(const MadKinematicDescriptor *in, MadKinematicDescriptor *out, bool check_apertures,
	bool invert_beam_coord_sytems)
{
  if (in == NULL || out == NULL || !trained_)
    return false;

  double input[5];
  double output[5];
  input[0] = in->x;
  input[1] = in->theta_x;
  input[2] = in->y;
  input[3] = in->theta_y;
  input[4] = in->ksi;

  bool res = Transport(input, output, check_apertures, invert_beam_coord_sytems);

  out->x = output[0];
  out->theta_x = output[1];
  out->y = output[2];
  out->theta_y = output[3];
  out->ksi = output[4];

  return res;
}



bool LHCOpticsApproximator::Transport_m_GeV(double in_pos[3], double in_momentum[3], double out_pos[3], double out_momentum[3],
	double z2_z1_dist, bool check_apertures, bool invert_beam_coord_sytems)
{
  double in[5];
  double out[5];
  double part_mom = 0.0;
  for(int i=0; i<3; i++)
    part_mom += in_momentum[i]*in_momentum[i];

  part_mom = TMath::Sqrt(part_mom);

  in[0] = in_pos[0];
  in[1] = in_momentum[0]/nominal_beam_momentum_;
  in[2] = in_pos[1];
  in[3] = in_momentum[1]/nominal_beam_momentum_;
  in[4] = (part_mom-nominal_beam_momentum_)/nominal_beam_momentum_;

  bool res = Transport(in, out, check_apertures, invert_beam_coord_sytems);

  out_pos[0] = out[0];
  out_pos[1] = out[2];
  out_pos[2] = in_pos[2] + z2_z1_dist;

  out_momentum[0] = out[1]*nominal_beam_momentum_;
  out_momentum[1] = out[3]*nominal_beam_momentum_;
  double part_out_total_mom = (out[4]+1)*nominal_beam_momentum_;
  out_momentum[2] = TMath::Sqrt(part_out_total_mom*part_out_total_mom - out_momentum[0]*out_momentum[0] - out_momentum[1]*out_momentum[1]);
  out_momentum[2] = TMath::Sign(out_momentum[2], in_momentum[2]);

  return res;
}



LHCOpticsApproximator::LHCOpticsApproximator(const LHCOpticsApproximator &org) : TNamed(org), x_parametrisation(org.x_parametrisation), theta_x_parametrisation(org.theta_x_parametrisation), y_parametrisation(org.y_parametrisation), theta_y_parametrisation(org.theta_y_parametrisation)
{
//  std::cout<<"LHCOpticsApproximator::LHCOpticsApproximator(const LHCOpticsApproximator &org) entered"<<std::endl;
    void Init();
    s_begin_ = org.s_begin_;
    s_end_ = org.s_end_;
    trained_ = org.trained_;
    apertures_ = org.apertures_;
    beam = org.beam;
    nominal_beam_energy_ = org.nominal_beam_energy_;
    nominal_beam_momentum_ = org.nominal_beam_momentum_;
//  std::cout<<"LHCOpticsApproximator::LHCOpticsApproximator(const LHCOpticsApproximator &org) left"<<std::endl;
}


LHCOpticsApproximator & LHCOpticsApproximator::operator=(const LHCOpticsApproximator &org)
{
  if(this!=&org)
  {
    void Init();
    TNamed::operator=(org);
    s_begin_ = org.s_begin_;
    s_end_ = org.s_end_;
    trained_ = org.trained_;

    x_parametrisation = org.x_parametrisation;
    theta_x_parametrisation = org.theta_x_parametrisation;
    y_parametrisation = org.y_parametrisation;
    theta_y_parametrisation = org.theta_y_parametrisation;
    apertures_ = org.apertures_;
    beam = org.beam;
    nominal_beam_energy_ = org.nominal_beam_energy_;
    nominal_beam_momentum_ = org.nominal_beam_momentum_;
  }

  return *this;
}


void LHCOpticsApproximator::Train(TTree *inp_tree, std::string data_prefix, polynomials_selection mode, int max_degree_x, int max_degree_tx, int max_degree_y, int max_degree_ty, bool common_terms, double *prec)
{
  if(inp_tree==NULL)
    return;

  //PrintCurrentMemoryUsage("Train, begin");

  InitializeApproximators(mode, max_degree_x, max_degree_tx, max_degree_y, max_degree_ty, common_terms);
  std::cout<<this->GetName()<<" is being trained..."<<std::endl;

  //in-variables
  //x_in, theta_x_in, y_in, theta_y_in, ksi_in, s_in
  double in_var[6];

  //out-variables
  //x_out, theta_x_out, y_out, theta_y_out, ksi_out, s_out, valid_out;
  double out_var[7];

  //in- out-lables
  std::string x_in_lab = "x_in";
  std::string theta_x_in_lab = "theta_x_in";
  std::string y_in_lab = "y_in";
  std::string theta_y_in_lab = "theta_y_in";
  std::string ksi_in_lab = "ksi_in";
  std::string s_in_lab = "s_in";

  std::string x_out_lab = data_prefix + "_x_out";
  std::string theta_x_out_lab = data_prefix + "_theta_x_out";
  std::string y_out_lab = data_prefix + "_y_out";
  std::string theta_y_out_lab = data_prefix + "_theta_y_out";
  std::string ksi_out_lab = data_prefix + "_ksi_out";
  std::string s_out_lab = data_prefix + "_s_out";
  std::string valid_out_lab = data_prefix + "_valid_out";

  //disable not needed branches to speed up the readin
  inp_tree->SetBranchStatus("*",0);  //disable all branches
  inp_tree->SetBranchStatus(x_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_x_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(y_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_y_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(ksi_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(x_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_x_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(y_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_y_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(ksi_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(valid_out_lab.c_str(),1);

  //set input data adresses
  inp_tree->SetBranchAddress(x_in_lab.c_str(), &(in_var[0]) );
  inp_tree->SetBranchAddress(theta_x_in_lab.c_str(), &(in_var[1]) );
  inp_tree->SetBranchAddress(y_in_lab.c_str(), &(in_var[2]) );
  inp_tree->SetBranchAddress(theta_y_in_lab.c_str(), &(in_var[3]) );
  inp_tree->SetBranchAddress(ksi_in_lab.c_str(), &(in_var[4]) );
  inp_tree->SetBranchAddress(s_in_lab.c_str(), &(in_var[5]) );

  //set output data adresses
  inp_tree->SetBranchAddress(x_out_lab.c_str(), &(out_var[0]) );
  inp_tree->SetBranchAddress(theta_x_out_lab.c_str(), &(out_var[1]) );
  inp_tree->SetBranchAddress(y_out_lab.c_str(), &(out_var[2]) );
  inp_tree->SetBranchAddress(theta_y_out_lab.c_str(), &(out_var[3]) );
  inp_tree->SetBranchAddress(ksi_out_lab.c_str(), &(out_var[4]) );
  inp_tree->SetBranchAddress(s_out_lab.c_str(), &(out_var[5]) );
  inp_tree->SetBranchAddress(valid_out_lab.c_str(), &(out_var[6]) );

  Long64_t entries = inp_tree->GetEntries();
  if(entries>0)
  {
    inp_tree->SetBranchStatus(s_in_lab.c_str(),1);
    inp_tree->SetBranchStatus(s_out_lab.c_str(),1);
    inp_tree->GetEntry(0);
    s_begin_ = in_var[5];
    s_end_ = out_var[5];
    inp_tree->SetBranchStatus(s_in_lab.c_str(),0);
    inp_tree->SetBranchStatus(s_out_lab.c_str(),0);
  }

  //set input and output variables for fitting
  for(Long64_t i=0; i<entries; ++i)
  {
    inp_tree->GetEntry(i);
    if(out_var[6] != 0)  //if out data valid
    {
      x_parametrisation.AddRow(in_var, out_var[0], 0);
      theta_x_parametrisation.AddRow(in_var, out_var[1], 0);
      y_parametrisation.AddRow(in_var, out_var[2], 0);
      theta_y_parametrisation.AddRow(in_var, out_var[3], 0);
    }
  }

  std::cout<<"Optical functions parametrizations from "<<s_begin_<<" to "<<s_end_<<std::endl;
  PrintInputRange();
  for(int i=0; i<4; i++)
  {
    double best_precision=0.0;
    if(prec)
      best_precision = prec[i];
    out_polynomials[i]->FindParameterization(best_precision);
  }

  trained_ = true;
}


void LHCOpticsApproximator::InitializeApproximators(polynomials_selection mode, int max_degree_x, int max_degree_tx, int max_degree_y, int max_degree_ty, bool common_terms)
{
  SetDefaultAproximatorSettings(x_parametrisation, X, max_degree_x);
  SetDefaultAproximatorSettings(theta_x_parametrisation, THETA_X, max_degree_tx);
  SetDefaultAproximatorSettings(y_parametrisation, Y, max_degree_y);
  SetDefaultAproximatorSettings(theta_y_parametrisation, THETA_Y, max_degree_ty);

  if(mode == PREDEFINED)
  {
    SetTermsManually(x_parametrisation, X, max_degree_x, common_terms);
    SetTermsManually(theta_x_parametrisation, THETA_X, max_degree_tx, common_terms);
    SetTermsManually(y_parametrisation, Y, max_degree_y, common_terms);
    SetTermsManually(theta_y_parametrisation, THETA_Y, max_degree_ty, common_terms);
  }
}


void LHCOpticsApproximator::SetDefaultAproximatorSettings(TMultiDimFet &approximator, variable_type var_type, int max_degree)
{
  if(max_degree<1 || max_degree>20)
    max_degree = 10;

  if(var_type == X || var_type == THETA_X)
  {
    Int_t mPowers[] = { 2, 4, 2, 4, max_degree };
    approximator.SetMaxPowers (mPowers);
    approximator.SetMaxFunctions (3000);
    approximator.SetMaxStudy (3000);
    approximator.SetMaxTerms (3000);
    approximator.SetPowerLimit (1.6);
//    approximator.SetMinAngle (2e-4);
//    approximator.SetMaxAngle (10);
    approximator.SetMinRelativeError (1e-13);
  }

  if(var_type == Y || var_type == THETA_Y)
  {
    Int_t mPowers[] = { 2, 4, 2, 4, max_degree };
    approximator.SetMaxPowers (mPowers);
    approximator.SetMaxFunctions (3000);
    approximator.SetMaxStudy (3000);
    approximator.SetMaxTerms (3000);
    approximator.SetPowerLimit (1.6);
//    approximator.SetMinAngle (2e-4);
//    approximator.SetMaxAngle (10);
    approximator.SetMinRelativeError (1e-13);
  }
}


void LHCOpticsApproximator::SetTermsManually(TMultiDimFet &approximator, variable_type variable, int max_degree, bool common_terms)
{
  if(max_degree<1 || max_degree>20)
    max_degree = 10;

  //put terms of shape:
  //1,0,0,0,t    0,1,0,0,t    0,2,0,0,t    0,3,0,0,t    0,0,0,0,t
  //t: 0,1,...,max_degree
//  int total_terms = 5*(max_degree+1);
//  int table_size = total_terms*5;
//  Int_t powers[table_size];

  std::vector<Int_t> term_literals;
  term_literals.reserve(5000);

  if(variable == X || variable == THETA_X)
  {
    //1,0,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(1);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
    //0,1,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(1);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
    //0,2,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(2);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
    //0,3,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(3);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
    //0,0,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
  }

  if(variable == Y || variable == THETA_Y)
  {
    //0,0,1,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(1);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
    //0,0,0,1,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(1);
      term_literals.push_back(i);
    }
    //0,0,0,2,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(2);
      term_literals.push_back(i);
    }
    //0,0,0,3,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(3);
      term_literals.push_back(i);
    }
    //0,0,0,0,t
    for(int i=0; i<=max_degree; ++i)
    {
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(0);
      term_literals.push_back(i);
    }
  }

  //push common terms
  if(common_terms)
  {
    term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(0);

    term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1), term_literals.push_back(1);

    term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0);
    term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0);
    term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(0);
    term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(0);

    term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1);
    term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1);
    term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(0), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(2), term_literals.push_back(1);
    term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(0), term_literals.push_back(1), term_literals.push_back(1);
    term_literals.push_back(0), term_literals.push_back(0), term_literals.push_back(2), term_literals.push_back(1), term_literals.push_back(1);
  }

  Int_t powers[term_literals.size()];
  for(unsigned int i=0; i<term_literals.size(); ++i)
  {
    powers[i] = term_literals[i];
  }
  approximator.SetPowers(powers, term_literals.size()/5);
}


void LHCOpticsApproximator::Test(TTree *inp_tree, TFile *f_out, std::string data_prefix, std::string base_out_dir)
{
  if(inp_tree==NULL || f_out==NULL)
    return;

  std::cout<<this->GetName()<<" is being tested..."<<std::endl;
  //in-variables
  //x_in, theta_x_in, y_in, theta_y_in, ksi_in, s_in
  double in_var[6];

  //out-variables
  //x_out, theta_x_out, y_out, theta_y_out, ksi_out, s_out, valid_out;
  double out_var[7];

  //in- out-lables
  std::string x_in_lab = "x_in";
  std::string theta_x_in_lab = "theta_x_in";
  std::string y_in_lab = "y_in";
  std::string theta_y_in_lab = "theta_y_in";
  std::string ksi_in_lab = "ksi_in";
  std::string s_in_lab = "s_in";

  std::string x_out_lab = data_prefix + "_x_out";
  std::string theta_x_out_lab = data_prefix + "_theta_x_out";
  std::string y_out_lab = data_prefix + "_y_out";
  std::string theta_y_out_lab = data_prefix + "_theta_y_out";
  std::string ksi_out_lab = data_prefix + "_ksi_out";
  std::string s_out_lab = data_prefix + "_s_out";
  std::string valid_out_lab = data_prefix + "_valid_out";

  //disable not needed branches to speed up the readin
  inp_tree->SetBranchStatus("*",0);  //disable all branches
  inp_tree->SetBranchStatus(x_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_x_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(y_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_y_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(ksi_in_lab.c_str(),1);
  inp_tree->SetBranchStatus(x_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_x_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(y_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(theta_y_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(ksi_out_lab.c_str(),1);
  inp_tree->SetBranchStatus(valid_out_lab.c_str(),1);

  //set input data adresses
  inp_tree->SetBranchAddress(x_in_lab.c_str(), &(in_var[0]) );
  inp_tree->SetBranchAddress(theta_x_in_lab.c_str(), &(in_var[1]) );
  inp_tree->SetBranchAddress(y_in_lab.c_str(), &(in_var[2]) );
  inp_tree->SetBranchAddress(theta_y_in_lab.c_str(), &(in_var[3]) );
  inp_tree->SetBranchAddress(ksi_in_lab.c_str(), &(in_var[4]) );
  inp_tree->SetBranchAddress(s_in_lab.c_str(), &(in_var[5]) );

  //set output data adresses
  inp_tree->SetBranchAddress(x_out_lab.c_str(), &(out_var[0]) );
  inp_tree->SetBranchAddress(theta_x_out_lab.c_str(), &(out_var[1]) );
  inp_tree->SetBranchAddress(y_out_lab.c_str(), &(out_var[2]) );
  inp_tree->SetBranchAddress(theta_y_out_lab.c_str(), &(out_var[3]) );
  inp_tree->SetBranchAddress(ksi_out_lab.c_str(), &(out_var[4]) );
  inp_tree->SetBranchAddress(s_out_lab.c_str(), &(out_var[5]) );
  inp_tree->SetBranchAddress(valid_out_lab.c_str(), &(out_var[6]) );

  //test histogramms
  TH1D *err_hists[4];
  TH2D *err_inp_cor_hists[4][5];
  TH2D *err_out_cor_hists[4][5];

  AllocateErrorHists(err_hists);
  AllocateErrorInputCorHists(err_inp_cor_hists);
  AllocateErrorOutputCorHists(err_out_cor_hists);

  Long64_t entries = inp_tree->GetEntries();
  //set input and output variables for fitting
  for(Long64_t i=0; i<entries; ++i)
  {
    double errors[4];
    inp_tree->GetEntry(i);

    errors[0] = out_var[0] - x_parametrisation.Eval(in_var);
    errors[1] = out_var[1] - theta_x_parametrisation.Eval(in_var);
    errors[2] = out_var[2] - y_parametrisation.Eval(in_var);
    errors[3] = out_var[3] - theta_y_parametrisation.Eval(in_var);

    FillErrorHistograms(errors, err_hists);
    FillErrorDataCorHistograms(errors, in_var, err_inp_cor_hists);
    FillErrorDataCorHistograms(errors, out_var, err_out_cor_hists);
  }

  WriteHistograms(err_hists, err_inp_cor_hists, err_out_cor_hists, f_out, base_out_dir);
  std::cout<<"Histograms have been written."<<std::endl;

  DeleteErrorHists(err_hists);
  DeleteErrorCorHistograms(err_inp_cor_hists);
  DeleteErrorCorHistograms(err_out_cor_hists);
}


void LHCOpticsApproximator::AllocateErrorHists(TH1D *err_hists[4])
{
  std::vector<std::string> error_labels;
  error_labels.push_back("x error");
  error_labels.push_back("theta_x error");
  error_labels.push_back("y error");
  error_labels.push_back("theta_y error");

  for(int i=0; i<4; ++i)
  {
    err_hists[i] = new TH1D(error_labels[i].c_str(), error_labels[i].c_str(), 100, -0.0000000001, 0.0000000001);
    err_hists[i]->SetXTitle(error_labels[i].c_str());
    err_hists[i]->SetYTitle("counts");
    err_hists[i]->SetDirectory(0);
    err_hists[i]->SetCanExtend(TH1::kAllAxes);
  }
}


void LHCOpticsApproximator::TestAperture(TTree *inp_tree, TTree *out_tree)  //x, theta_x, y, theta_y, ksi, mad_accepted, parametriz_accepted
{
  if(inp_tree==NULL || out_tree==NULL)
    return;

  Long64_t entries = inp_tree->GetEntries();
  double entry[7];
  double parametrization_out[5];

  inp_tree->SetBranchAddress("x", &(entry[0]) );
  inp_tree->SetBranchAddress("theta_x", &(entry[1]) );
  inp_tree->SetBranchAddress("y", &(entry[2]) );
  inp_tree->SetBranchAddress("theta_y", &(entry[3]) );
  inp_tree->SetBranchAddress("ksi", &(entry[4]) );
  inp_tree->SetBranchAddress("mad_accept", &(entry[5]) );
  inp_tree->SetBranchAddress("par_accept", &(entry[6]) );

  out_tree->SetBranchAddress("x", &(entry[0]) );
  out_tree->SetBranchAddress("theta_x", &(entry[1]) );
  out_tree->SetBranchAddress("y", &(entry[2]) );
  out_tree->SetBranchAddress("theta_y", &(entry[3]) );
  out_tree->SetBranchAddress("ksi", &(entry[4]) );
  out_tree->SetBranchAddress("mad_accept", &(entry[5]) );
  out_tree->SetBranchAddress("par_accept", &(entry[6]) );

//  int ind=0;
  for(Long64_t i=0; i<entries; i++)
  {
    inp_tree->GetEntry(i);
//    for(int j=0; j<7; j++)
//      std::cout<<entry[j]<<" ";
//    std::cout<<std::endl;

    bool res = Transport(entry, parametrization_out, true);
//    for(int j=0; j<5; j++)
//      std::cout<<parametrization_out[j]<<" ";
//    std::cout<<"TestAperture "<<res<<std::endl;

    if( res )
      entry[6] = 1.0;
    else
      entry[6] = 0.0;

    out_tree->Fill();
//    if(ind++>300)
//      exit(0);
  }
}


void LHCOpticsApproximator::AllocateErrorInputCorHists(TH2D *err_inp_cor_hists[4][5])
{
  std::vector<std::string> error_labels;
  std::vector<std::string> data_labels;

  error_labels.push_back("x error");
  error_labels.push_back("theta_x error");
  error_labels.push_back("y error");
  error_labels.push_back("theta_y error");

  data_labels.push_back("x input");
  data_labels.push_back("theta_x input");
  data_labels.push_back("y input");
  data_labels.push_back("theta_y input");
  data_labels.push_back("ksi input");

  for(int eri=0; eri<4; ++eri)
  {
    for(int dati=0; dati<5; ++dati)
    {
      std::string name = error_labels[eri] + " vs. " + data_labels[dati];
      std::string title = name;
      err_inp_cor_hists[eri][dati] = new TH2D(name.c_str(), title.c_str(), 100,-0.0000000001,0.0000000001,100,-0.0000000001,0.0000000001);
      err_inp_cor_hists[eri][dati]->SetXTitle(error_labels[eri].c_str());
      err_inp_cor_hists[eri][dati]->SetYTitle(data_labels[dati].c_str());
      err_inp_cor_hists[eri][dati]->SetDirectory(0);
      err_inp_cor_hists[eri][dati]->SetCanExtend(TH2::kAllAxes);
    }
  }
}

void LHCOpticsApproximator::AllocateErrorOutputCorHists(TH2D *err_out_cor_hists[4][5])
{
  std::vector<std::string> error_labels;
  std::vector<std::string> data_labels;

  error_labels.push_back("x error");
  error_labels.push_back("theta_x error");
  error_labels.push_back("y error");
  error_labels.push_back("theta_y error");

  data_labels.push_back("x output");
  data_labels.push_back("theta_x output");
  data_labels.push_back("y output");
  data_labels.push_back("theta_y output");
  data_labels.push_back("ksi output");

  for(int eri=0; eri<4; ++eri)
  {
    for(int dati=0; dati<5; ++dati)
    {
      std::string name = error_labels[eri] + " vs. " + data_labels[dati];
      std::string title = name;
      err_out_cor_hists[eri][dati] = new TH2D(name.c_str(), title.c_str(), 100,-0.0000000001,0.0000000001,100,-0.0000000001,0.0000000001);
      err_out_cor_hists[eri][dati]->SetXTitle(error_labels[eri].c_str());
      err_out_cor_hists[eri][dati]->SetYTitle(data_labels[dati].c_str());
      err_out_cor_hists[eri][dati]->SetDirectory(0);
      err_out_cor_hists[eri][dati]->SetCanExtend(TH2::kAllAxes);
    }
  }
}


void LHCOpticsApproximator::FillErrorHistograms(double errors[4], TH1D *err_hists[4])
{
  for(int i=0; i<4; ++i)
  {
    err_hists[i]->Fill(errors[i]);
  }
}


void LHCOpticsApproximator::FillErrorDataCorHistograms(double errors[4], double var[5], TH2D *err_cor_hists[4][5])
{
  for(int eri=0; eri<4; ++eri)
  {
    for(int dati=0; dati<5; ++dati)
    {
      err_cor_hists[eri][dati]->Fill(errors[eri], var[dati]);
    }
  }
}


void LHCOpticsApproximator::DeleteErrorHists(TH1D *err_hists[4])
{
  for(int i=0; i<4; ++i)
  {
    delete err_hists[i];
  }
}

void LHCOpticsApproximator::DeleteErrorCorHistograms(TH2D *err_cor_hists[4][5])
{
  for(int eri=0; eri<4; ++eri)
  {
    for(int dati=0; dati<5; ++dati)
    {
      delete err_cor_hists[eri][dati];
    }
  }
}


void LHCOpticsApproximator::WriteHistograms(TH1D *err_hists[4], TH2D *err_inp_cor_hists[4][5], TH2D *err_out_cor_hists[4][5], TFile *f_out, std::string base_out_dir)
{
  if(f_out==NULL)
    return;

  f_out->cd();
  if(!gDirectory->cd(base_out_dir.c_str()))
    gDirectory->mkdir(base_out_dir.c_str());

  gDirectory->cd(base_out_dir.c_str());
  gDirectory->mkdir(this->GetName());
  gDirectory->cd(this->GetName());
  gDirectory->mkdir("x");
  gDirectory->mkdir("theta_x");
  gDirectory->mkdir("y");
  gDirectory->mkdir("theta_y");

  gDirectory->cd("x");
  err_hists[0]->Write("", TObject::kWriteDelete);
  for(int i=0; i<5; i++)
  {
    err_inp_cor_hists[0][i]->Write("", TObject::kWriteDelete);
    err_out_cor_hists[0][i]->Write("", TObject::kWriteDelete);
  }

  gDirectory->cd("..");
  gDirectory->cd("theta_x");
  err_hists[1]->Write("", TObject::kWriteDelete);
  for(int i=0; i<5; i++)
  {
    err_inp_cor_hists[1][i]->Write("", TObject::kWriteDelete);
    err_out_cor_hists[1][i]->Write("", TObject::kWriteDelete);
  }

  gDirectory->cd("..");
  gDirectory->cd("y");
  err_hists[2]->Write("", TObject::kWriteDelete);
  for(int i=0; i<5; i++)
  {
    err_inp_cor_hists[2][i]->Write("", TObject::kWriteDelete);
    err_out_cor_hists[2][i]->Write("", TObject::kWriteDelete);
  }

  gDirectory->cd("..");
  gDirectory->cd("theta_y");
  err_hists[3]->Write("", TObject::kWriteDelete);
  for(int i=0; i<5; i++)
  {
    err_inp_cor_hists[3][i]->Write("", TObject::kWriteDelete);
    err_out_cor_hists[3][i]->Write("", TObject::kWriteDelete);
  }
  gDirectory->cd("..");
  gDirectory->cd("..");
}

void LHCOpticsApproximator::PrintInputRange() const
{
  const TVectorD* min_var = x_parametrisation.GetMinVariables();
  const TVectorD* max_var = x_parametrisation.GetMaxVariables();

  std::cout<<"Covered input parameters range:"<<std::endl;
  for(int i=0; i<5; i++)
  {
    std::cout<<(*min_var)(i)<<" < "<<coord_names[i]<<" < "<<(*max_var)(i)<<std::endl;
  }
  std::cout<<std::endl;
}


bool LHCOpticsApproximator::CheckInputRange(double *in, bool invert_beam_coord_sytems) const
{
  double in_corrected[5];
  if(beam==lhcb1 || !invert_beam_coord_sytems)
  {
    in_corrected[0] = in[0];
    in_corrected[1] = in[1];
    in_corrected[2] = in[2];
    in_corrected[3] = in[3];
    in_corrected[4] = in[4];
  }
  else
  {
    in_corrected[0] = -in[0];
    in_corrected[1] = -in[1];
    in_corrected[2] = in[2];
    in_corrected[3] = in[3];
    in_corrected[4] = in[4];
  }

  const TVectorD* min_var = x_parametrisation.GetMinVariables();
  const TVectorD* max_var = x_parametrisation.GetMaxVariables();
  bool res = true;

  for(int i=0; i<5; i++)
  {
    res = res && in_corrected[i]>=(*min_var)(i) && in_corrected[i]<=(*max_var)(i);
  }

  return res;
}


void LHCOpticsApproximator::AddRectEllipseAperture(const LHCOpticsApproximator &in, double rect_x, double rect_y, double r_el_x, double r_el_y)
{
  apertures_.push_back(LHCApertureApproximator(in, rect_x, rect_y, r_el_x, r_el_y, LHCApertureApproximator::RECTELLIPSE));
}


//////////////////////////////////////////////////////////////////

LHCApertureApproximator::LHCApertureApproximator()
{
  rect_x_ = rect_y_ = r_el_x_ = r_el_y_ = 0.0;
  ap_type_ = NO_APERTURE;
}



LHCApertureApproximator::LHCApertureApproximator(const LHCOpticsApproximator &in, double rect_x, double rect_y, double r_el_x, double r_el_y,
        aperture_type type) : LHCOpticsApproximator(in)
{
  rect_x_ = rect_x;
  rect_y_ = rect_y;
  r_el_x_ = r_el_x;
  r_el_y_ = r_el_y;
  ap_type_ = type;
}



bool LHCApertureApproximator::CheckAperture(double *in, bool invert_beam_coord_sytems)
{
  double out[5];
  bool result = Transport(in, out, false, invert_beam_coord_sytems);

  if (ap_type_==RECTELLIPSE)
  {
    result = result && out[0]<rect_x_ && out[0]>-rect_x_ && out[2]<rect_y_ && out[2]>-rect_y_ &&
        ( out[0]*out[0]/(r_el_x_*r_el_x_) + out[2]*out[2]/(r_el_y_*r_el_y_) < 1 );
  }

  return result;
}

/*
bool LHCApertureApproximator::CheckAperture(MadKinematicDescriptor *in)  //x, thx. y, thy, ksi
{
  MadKinematicDescriptor out;
  bool result = Transport(in, &out);
  if(ap_type_==RECTELLIPSE)
  {

    result = result && out.x<rect_x_ && out.x>-rect_x_ && out.y<rect_y_ && out.y>-rect_y_ &&
        ( out.x*out.x/(r_el_x_*r_el_x_) + out.y*out.y/(r_el_y_*r_el_y_) < 1 );
  }
  return result;
}
*/

void LHCOpticsApproximator::PrintOpticalFunctions() const
{
  std::cout<<std::endl<<"Linear terms of optical functions:"<<std::endl;
  for(int i=0; i<4; i++)
  {
    PrintCoordinateOpticalFunctions(*out_polynomials[i], coord_names[i], coord_names);
  }
}

void LHCOpticsApproximator::PrintCoordinateOpticalFunctions(TMultiDimFet &parametrization, const std::string &coord_name,
	const std::vector<std::string> &input_vars) const
{
  double in[5];
  double d_out_d_in[5];
  double d_par = 1e-5;
  double bias = 0;

  for(int j=0; j<5; j++)
      in[j]=0.0;

  const TVectorD* min_var = x_parametrisation.GetMinVariables();
  const TVectorD* max_var = x_parametrisation.GetMaxVariables();

  bias = parametrization.Eval(in);

  for(int i=0; i<5; i++)
  {
    for(int j=0; j<5; j++)
      in[j]=0.0;

    d_par = -((*max_var)[i]-(*min_var)[i])/10.0;
    in[i] = d_par;
    d_out_d_in[i] = parametrization.Eval(in);
    in[i] = 0.0;
    d_out_d_in[i] = d_out_d_in[i] - parametrization.Eval(in);
    d_out_d_in[i] = d_out_d_in[i]/d_par;
  }
  std::cout<<coord_name<<" = "<<bias;
  for(int i=0; i<5; i++)
  {
    std::cout<<" + "<<d_out_d_in[i]<<"*"<<input_vars[i];
  }
  std::cout<<std::endl;
}



//real angles in the matrix, MADX convention used only for input
void LHCOpticsApproximator::GetLineariasedTransportMatrixX(
    double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy, 
    double mad_init_xi, TMatrixD &transp_matrix, double d_mad_x, double d_mad_thx)
{
  double MADX_momentum_correction_factor = 1.0 + mad_init_xi;
  transp_matrix.ResizeTo(2,2);
  double in[5];
  in[0] = mad_init_x;
  in[1] = mad_init_thx;
  in[2] = mad_init_y;
  in[3] = mad_init_thy;
  in[4] = mad_init_xi;
  
  double out[5];
  
  Transport(in, out);
  double x1 = out[0];
  double thx1 = out[1];
  
  in[0] = mad_init_x + d_mad_x;
  Transport(in, out);
  double x2_dx = out[0];
  double thx2_dx = out[1];
  
  in[0] = mad_init_x;
  in[1] = mad_init_thx + d_mad_thx;  //?
  Transport(in, out);
  double x2_dthx = out[0];
  double thx2_dthx = out[1];
  
//  | dx/dx,   dx/dthx    |
//  | dthx/dx, dtchx/dthx |
  
  transp_matrix(0,0) = (x2_dx-x1)/d_mad_x;
  transp_matrix(1,0) = (thx2_dx-thx1)/(d_mad_x*MADX_momentum_correction_factor);
  transp_matrix(0,1) = MADX_momentum_correction_factor*(x2_dthx-x1)/d_mad_thx;
  transp_matrix(1,1) = (thx2_dthx-thx1)/d_mad_thx;
}


//real angles in the matrix, MADX convention used only for input
void LHCOpticsApproximator::GetLineariasedTransportMatrixY(
    double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy, 
    double mad_init_xi, TMatrixD &transp_matrix, double d_mad_y, double d_mad_thy)
{
  double MADX_momentum_correction_factor = 1.0 + mad_init_xi;
  transp_matrix.ResizeTo(2,2);
  double in[5];
  in[0] = mad_init_x;
  in[1] = mad_init_thx;
  in[2] = mad_init_y;
  in[3] = mad_init_thy;
  in[4] = mad_init_xi;
  
  double out[5];
  
  Transport(in, out);
  double y1 = out[2];
  double thy1 = out[3];
  
  in[2] = mad_init_y + d_mad_y;
  Transport(in, out);
  double y2_dy = out[2];
  double thy2_dy = out[3];
  
  in[2] = mad_init_y;
  in[3] = mad_init_thy + d_mad_thy;  //?
  Transport(in, out);
  double y2_dthy = out[2];
  double thy2_dthy = out[3];
  
//  | dy/dy,   dy/dthy    |
//  | dthy/dy, dtchy/dthy |
  
  transp_matrix(0,0) = (y2_dy-y1)/d_mad_y;
  transp_matrix(1,0) = (thy2_dy-thy1)/(d_mad_y*MADX_momentum_correction_factor);
  transp_matrix(0,1) = MADX_momentum_correction_factor*(y2_dthy-y1)/d_mad_thy;
  transp_matrix(1,1) = (thy2_dthy-thy1)/d_mad_thy;
}

//MADX convention used only for input
double LHCOpticsApproximator::GetDx(
    double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy, 
    double mad_init_xi, double d_mad_xi)
{
  double in[5];
  in[0] = mad_init_x;
  in[1] = mad_init_thx;
  in[2] = mad_init_y;
  in[3] = mad_init_thy;
  in[4] = mad_init_xi;
  
  double out[5];
  
  Transport(in, out);
  double x1 = out[0];
  
  in[4] = mad_init_xi + d_mad_xi;
  Transport(in, out);
  double x2_dxi = out[0];
  double dispersion = (x2_dxi-x1)/d_mad_xi;
  
  return dispersion;
}

//MADX convention used only for input
//angular dispersion
double LHCOpticsApproximator::GetDxds(
    double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy, 
    double mad_init_xi, double d_mad_xi)
{
  double MADX_momentum_correction_factor = 1.0 + mad_init_xi;
  double in[5];
  in[0] = mad_init_x;
  in[1] = mad_init_thx;
  in[2] = mad_init_y;
  in[3] = mad_init_thy;
  in[4] = mad_init_xi;
  
  double out[5];
  
  Transport(in, out);
  double thx1 = out[1]/MADX_momentum_correction_factor;
  
  in[4] = mad_init_xi + d_mad_xi;
  Transport(in, out);
  double thx2_dxi = out[1]/MADX_momentum_correction_factor;
  double dispersion = (thx2_dxi-thx1)/d_mad_xi;
  
  return dispersion;
}
