#ifndef DPEDataGenerator_h_
#define DPEDataGenerator_h_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "TChain.h"
#include "h101.h"
#include "TTree.h"
#include "TNtupleD.h"


struct scoring_plane_data
{
	scoring_plane_data()
	{
		x = 0;
		theta_x = 0;
		y = 0;
		theta_y = 0;
		xi = 0;
		s = 0;
		accepted = 0;
	}
	double x, theta_x, y, theta_y, xi, s, accepted;
};

struct scoring_plane_info
{
  std::string x_lab;
  std::string theta_x_lab;
  std::string y_lab;
  std::string theta_y_lab;
  std::string xi_lab;
  std::string s_lab;
  std::string accepted_lab;
  scoring_plane_data data;
};

struct mass_info
{
	std::string mass_lab;
	double mass;
};

//direction, marker
typedef std::map<int, std::map<std::string, scoring_plane_info> > output_tree_shape;
//
//struct ip_data
//{
//	ip_data()
//	{
//		x=y=px=py=pz=xi=t_0=phi_0=thetax=thetay=xi_0;
//	}
//	double x,y,px,py,pz,xi,t_0,phi_0,thetax,thetay,xi_0;
//};

struct ip_info
{
	std::string x_lab, y_lab, px_lab, py_lab, pz_lab, xi_lab, t_0_lab, phi_0_lab, thetax_lab, 
			thetay_lab, xi_0_lab;
	MADProton data;
};

typedef std::map<int, ip_info> output_tree_input_labels;

//direction, marker, particle number, proton parameters
typedef std::map<int, std::map<std::string, std::map<int, scoring_plane_data> > > output_data;


class DPEDataGenerator
{
  public:
  	DPEDataGenerator(std::string file_name, std::string out_file_name);
  	~DPEDataGenerator();
  	void SimulateDPEEvents(std::string config_file_r, std::string config_file_l);
  	
  private:
  	std::vector<std::string> input_root_file_names;
  	TChain *input_chain_;
  	h101 *input_data_source_;
  	TFile *out_file_;
  	TTree *out_tree_;
  	output_tree_shape tree_branches_;
  	output_tree_input_labels tree_ip_info_;
  	mass_info tree_mass_info_;
  	output_data output_data_;
  	
  	//configuarion data
  	int number_of_input_particles_per_iteration;
  	std::string MAD_input_file_name;
  	std::string DPE_tree_name_;
  	std::string DPE_tree_file_name_;

  	void LoadRootChain(std::string file_name);
  	int GetInputFileNames(std::string file_name);
  	void CreateOutputROOTFile(std::vector<std::string> scoring_planes_r, std::vector<std::string> scoring_planes_l);
  	void LoadTextData(int direction);  //+1 right, -1 left
  	void AppendROOTFile(const MADProtonPairCollection &protons);
  	void CloseOutputROOTFile();
};

#endif
