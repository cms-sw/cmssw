#ifndef LHC_OPTICS_APPROXIMATOR__
#define LHC_OPTICS_APPROXIMATOR__

#include <string>
#include <iostream>
#include "TNamed.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"

#include "TMultiDimFet.h"



struct MadKinematicDescriptor
{
  double x;
  double theta_x;
  double y;
  double theta_y;
  double ksi;

  void SetValues(double *in)
  {
    x = in[0];
    theta_x = in[1];
    y = in[2];
    theta_y = in[3];
    ksi = in[4];
  }
};



class LHCApertureApproximator;



/**
class finds the parametrisation of MADX proton transport and transports the protons according to it
5 phase space variables are taken in to confoguration: x, y, theta_x, theta_y, xi
In general: x and y are in m, theta_x and theta_y in rad, and xi is negative in case of momentum loss
**/
class LHCOpticsApproximator : public TNamed
{
  public:
    LHCOpticsApproximator();

    //begin and end position along the beam of the particle to transport, training_tree, prefix of data branch in the tree
    LHCOpticsApproximator(std::string name, std::string title, TMultiDimFet::EMDFPolyType polynom_type,
        std::string beam_direction, double nominal_beam_energy);

    LHCOpticsApproximator(const LHCOpticsApproximator &org);

    LHCOpticsApproximator & operator=(const LHCOpticsApproximator &org);

    enum polynomials_selection {AUTOMATIC, PREDEFINED};
    enum beam_type {lhcb1, lhcb2};

    void Train(TTree *inp_tree, std::string data_prefix = std::string("def"), polynomials_selection mode = PREDEFINED,
		int max_degree_x = 10, int max_degree_tx = 10, int max_degree_y = 10, int max_degree_ty = 10, bool common_terms = false, double *prec=NULL);

    void Test(TTree *inp_tree, TFile *f_out, std::string data_prefix = std::string("def"), std::string base_out_dir = std::string(""));

    void TestAperture(TTree *in_tree, TTree *out_tree);

	/// \brief transports a proton
	/// structure of input and output vectors: x, theta_x, y, theta_y, xi
	/// check_apertures: flag whether appertures should be checked
	/// invert_beam_coord_sytems: set to true unless you have a reason to do differently
	/// returns true if transport possible, double
    bool Transport(double *in, double *out, bool check_apertures=false, bool invert_beam_coord_sytems=true);

	/// \brief transports a proton
    bool Transport(const MadKinematicDescriptor *in, MadKinematicDescriptor *out, bool check_apertures=false, bool invert_beam_coord_sytems=true);

	/// \brief transports a proton
	/// pos: position (x, y and z) in m
	/// momentum: momentum in GeV
    bool Transport_m_GeV(double in_pos[3], double in_momentum[3], double out_pos[3], double out_momentum[3], double z2_z1_dist,
        bool check_apertures, bool invert_beam_coord_sytems=true);

    void PrintInputRange() const;

    bool CheckInputRange(double *in, bool invert_beam_coord_sytems=true) const;

    void AddRectEllipseAperture(const LHCOpticsApproximator &in, double rect_x, double rect_y, double r_el_x, double r_el_y);

    void PrintOpticalFunctions() const;

    void PrintCoordinateOpticalFunctions(TMultiDimFet &parametrization, const std::string &coord_name,
		const std::vector<std::string> &input_vars) const;

    void GetLineariasedTransportMatrixX(double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy,
        double mad_init_xi, TMatrixD &tr_matrix, double d_mad_x=10e-6, double d_mad_thx=10e-6);

    void GetLineariasedTransportMatrixY(double mad_init_x, double mad_init_thx, double mad_init_y, double mad_init_thy,
        double mad_init_xi, TMatrixD &tr_matrix, double d_mad_y=10e-6, double d_mad_thy=10e-6);

    double GetDx(double mad_init_x, double mad_init_thx, double mad_init_y,
        double mad_init_thy, double mad_init_xi, double d_mad_xi=0.001);

    double GetDxds(double mad_init_x, double mad_init_thx, double mad_init_y,
        double mad_init_thy, double mad_init_xi, double d_mad_xi=0.001);

    std::vector<LHCApertureApproximator> GetApertures() const
	{
		return apertures_;
	}

    double GetSBegin() const
    {
      return s_begin_;
    }

    double GetSEnd() const
    {
      return s_end_;
    }

	beam_type GetBeamType() const
	{
		return beam;
	}

    double GetBeamEnergy() const
    {
      return nominal_beam_energy_;
    }

    double GetBeamMomentum() const
    {
      return nominal_beam_momentum_;
    }

  private:
    void Init();
    double s_begin_;	///< begin of transport along the reference orbit
    double s_end_;		///< end of transport along the reference orbit
    beam_type beam;
    double nominal_beam_energy_;	///< GeV
    double nominal_beam_momentum_;  ///< GeV
    bool trained_;  	///< trained polynomials
    std::vector<TMultiDimFet*> out_polynomials;  //! pointers to polynomials
    std::vector<std::string> coord_names;
    std::vector<LHCApertureApproximator> apertures_;  ///< apertures on the way

    TMultiDimFet x_parametrisation;  		///< polynomial approximation for x
    TMultiDimFet theta_x_parametrisation;	///< polynomial approximation for theta_x
    TMultiDimFet y_parametrisation;			///< polynomial approximation for y
    TMultiDimFet theta_y_parametrisation;	///< polynomial approximation for theta_y

    enum variable_type {X, THETA_X, Y, THETA_Y};

    void InitializeApproximators(polynomials_selection mode, int max_degree_x, int max_degree_tx, int max_degree_y, int max_degree_ty, bool common_terms);
    void SetDefaultAproximatorSettings(TMultiDimFet &approximator, variable_type var_type, int max_degree);
    void SetTermsManually(TMultiDimFet &approximator, variable_type variable, int max_degree, bool common_terms);

    void AllocateErrorHists(TH1D *err_hists[4]);
    void AllocateErrorInputCorHists(TH2D *err_inp_cor_hists[4][5]);
    void AllocateErrorOutputCorHists(TH2D *err_out_cor_hists[4][5]);

    void DeleteErrorHists(TH1D *err_hists[4]);
    void DeleteErrorCorHistograms(TH2D *err_cor_hists[4][5]);

    void FillErrorHistograms(double errors[4], TH1D *err_hists[4]);
    void FillErrorDataCorHistograms(double errors[4], double var[5], TH2D *err_cor_hists[4][5]);

    void WriteHistograms(TH1D *err_hists[4], TH2D *err_inp_cor_hists[4][5], TH2D *err_out_cor_hists[4][5], TFile *f_out, std::string base_out_dir);

    ClassDef(LHCOpticsApproximator,1)
};



/// \brief Aperture approximator
class LHCApertureApproximator : public LHCOpticsApproximator
{
  public:
    enum aperture_type {NO_APERTURE, RECTELLIPSE};

    LHCApertureApproximator();

    LHCApertureApproximator(const LHCOpticsApproximator &in, double rect_x, double rect_y, double r_el_x, double r_el_y,
        aperture_type type = RECTELLIPSE);

    bool CheckAperture(double *in, bool invert_beam_coord_sytems=true);

  private:
    double rect_x_, rect_y_, r_el_x_, r_el_y_;
    aperture_type ap_type_;

    ClassDef(LHCApertureApproximator,1)
};

#endif
