#include "DPEDataGenerator.h"
#include "MADParamGenerator.h"
#include "TFile.h"

DPEDataGenerator::DPEDataGenerator(std::string file_name, std::string out_file_name)
{
	number_of_input_particles_per_iteration = 12000;
	MAD_input_file_name = "part.in";
	DPE_tree_name_ = "DPETree";
	DPE_tree_file_name_ = out_file_name;
	
	input_chain_ = new TChain("h101");
	LoadRootChain(file_name);
	input_data_source_ = new h101(input_chain_);
	out_file_ = NULL;
	out_tree_ = NULL;
}

DPEDataGenerator::~DPEDataGenerator()
{
	delete input_chain_;
	delete input_data_source_;
}


int DPEDataGenerator::GetInputFileNames(std::string file_name)
{
	input_root_file_names.clear();
	std::fstream istr;
	istr.open(file_name.c_str(), std::ios::in);
	std::string input_file_name;
	
	while(istr.good() && !istr.fail() && !istr.eof())
	{
		istr>>input_file_name;
		if(!istr.eof())
		{
			input_root_file_names.push_back(input_file_name);
		}
	}
	
	istr.close();
	
	return input_root_file_names.size();
}


void DPEDataGenerator::LoadRootChain(std::string file_name)
{
	GetInputFileNames(file_name);
	for(unsigned i = 0; i<input_root_file_names.size(); i++)
	{
		input_chain_->AddFile(input_root_file_names[i].c_str());
		std::cout<<"DPE file added: "<<input_root_file_names[i].c_str()<<" "<<std::endl;
	}
	std::cout<<"Total numer of events: "<<input_chain_->GetEntries()<<std::endl;
}


void DPEDataGenerator::SimulateDPEEvents(std::string config_file_r, std::string config_file_l)
{
	MADParamGenerator mad_conf_gen_beam_r;
	MADParamGenerator mad_conf_gen_beam_l;
	
	mad_conf_gen_beam_r.OpenXMLConfigurationFile(config_file_r);
	mad_conf_gen_beam_l.OpenXMLConfigurationFile(config_file_l);
	
	std::vector<std::string> scoring_planes_r = mad_conf_gen_beam_r.GetAllScoringPlanes(1);
	std::vector<std::string> scoring_planes_l = mad_conf_gen_beam_l.GetAllScoringPlanes(1);
	
	CreateOutputROOTFile(scoring_planes_r, scoring_planes_l);
	
	for(Long64_t i=0; i<input_data_source_->GetEntries(); i+=number_of_input_particles_per_iteration)
	{
		std::cout<<"Entries "<<i<<" to "<<i+number_of_input_particles_per_iteration<<" out of "
		<<input_data_source_->GetEntries()<<" being processed."<<std::endl;
		MADProtonPairCollection protons = input_data_source_->GetMADProtonPairs(i, 
				number_of_input_particles_per_iteration);
		
		output_data_.clear();
		//right beam
		input_data_source_->WriteMADInputParticles(protons, 1, MAD_input_file_name);
		mad_conf_gen_beam_r.RunMADXWithExternalData(1, protons.size());
		LoadTextData(1);
		
		//left beam
		input_data_source_->WriteMADInputParticles(protons, -1, MAD_input_file_name);
		mad_conf_gen_beam_l.RunMADXWithExternalData(1, protons.size());
		LoadTextData(-1);
		
		AppendROOTFile(protons);
	}
	
	CloseOutputROOTFile();
}


void DPEDataGenerator::CreateOutputROOTFile(std::vector<std::string> scoring_planes_r, std::vector<std::string> scoring_planes_l)
{
	out_file_ = TFile::Open(DPE_tree_file_name_.c_str(), "RECREATE");
	std::string labels;
	std::string lab;
	
	std::string ip_r_lab = "ip_r_";
	std::string ip_l_lab = "ip_l_";
	std::string ip_mass_lab = "ip_mass";
	
	tree_mass_info_.mass_lab = ip_mass_lab;
	labels = ip_mass_lab;
	
	std::string x_lab = "x";
	std::string y_lab = "y";
	std::string px_lab = "px";
	std::string py_lab = "py";
	std::string pz_lab = "pz";
	std::string xi_lab = "xi";
	std::string t_0_lab = "t_0";
	std::string phi_0_lab = "phi_0";
	std::string thetax_lab = "theta_x";
	std::string thetay_lab = "theta_y";
	std::string xi_0_lab = "xi_0";
	
	//beam right
	lab = ip_r_lab + x_lab;
	tree_ip_info_[1].x_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + y_lab;
	tree_ip_info_[1].y_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + px_lab;
	tree_ip_info_[1].px_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + py_lab;
	tree_ip_info_[1].py_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + pz_lab;
	tree_ip_info_[1].pz_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + xi_lab;
	tree_ip_info_[1].xi_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + t_0_lab;
	tree_ip_info_[1].t_0_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + phi_0_lab;
	tree_ip_info_[1].phi_0_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + thetax_lab;
	tree_ip_info_[1].thetax_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + thetay_lab;
	tree_ip_info_[1].thetay_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_r_lab + xi_0_lab;
	tree_ip_info_[1].xi_0_lab = lab;
	labels = labels + ":" + lab;
	
	//beam left
	lab = ip_l_lab + x_lab;
	tree_ip_info_[-1].x_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + y_lab;
	tree_ip_info_[-1].y_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + px_lab;
	tree_ip_info_[-1].px_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + py_lab;
	tree_ip_info_[-1].py_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + pz_lab;
	tree_ip_info_[-1].pz_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + xi_lab;
	tree_ip_info_[-1].xi_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + t_0_lab;
	tree_ip_info_[-1].t_0_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + phi_0_lab;
	tree_ip_info_[-1].phi_0_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + thetax_lab;
	tree_ip_info_[-1].thetax_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + thetay_lab;
	tree_ip_info_[-1].thetay_lab = lab;
	labels = labels + ":" + lab;
	
	lab = ip_l_lab + xi_0_lab;
	tree_ip_info_[-1].xi_0_lab = lab;
	labels = labels + ":" + lab;
	
	
	//output data labels
	std::string dir_prefix_r = "r_";
	std::string dir_prefix_l = "l_";
	
	std::string x_suf = "_x";
	std::string y_suf = "_y";
	std::string theta_x_suf = "_theta_x";
	std::string theta_y_suf = "_theta_y";
	std::string xi_suf = "_xi";
	std::string s_suf = "_s";
	std::string accepted_suf = "_accepted";
	
	for(unsigned int i=0; i<scoring_planes_r.size(); i++)
	{
		lab = dir_prefix_r + scoring_planes_r[i] + x_suf;
		tree_branches_[1][scoring_planes_r[i]].x_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + y_suf;
		tree_branches_[1][scoring_planes_r[i]].y_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + theta_x_suf;
		tree_branches_[1][scoring_planes_r[i]].theta_x_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + theta_y_suf;
		tree_branches_[1][scoring_planes_r[i]].theta_y_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + xi_suf;
		tree_branches_[1][scoring_planes_r[i]].xi_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + s_suf;
		tree_branches_[1][scoring_planes_r[i]].s_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_r + scoring_planes_r[i] + accepted_suf;
		tree_branches_[1][scoring_planes_r[i]].accepted_lab = lab;
		labels = labels + ":" + lab;
	}
	
	for(unsigned int i=0; i<scoring_planes_l.size(); i++)
	{
		lab = dir_prefix_l + scoring_planes_l[i] + x_suf;
		tree_branches_[-1][scoring_planes_l[i]].x_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + y_suf;
		tree_branches_[-1][scoring_planes_l[i]].y_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + theta_x_suf;
		tree_branches_[-1][scoring_planes_l[i]].theta_x_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + theta_y_suf;
		tree_branches_[-1][scoring_planes_l[i]].theta_y_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + xi_suf;
		tree_branches_[-1][scoring_planes_l[i]].xi_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + s_suf;
		tree_branches_[-1][scoring_planes_l[i]].s_lab = lab;
		labels = labels + ":" + lab;
		
		lab = dir_prefix_l + scoring_planes_l[i] + accepted_suf;
		tree_branches_[-1][scoring_planes_l[i]].accepted_lab = lab;
		labels = labels + ":" + lab;
	}
	
	out_tree_ = new TNtupleD(DPE_tree_name_.c_str(), DPE_tree_name_.c_str(), labels.c_str());
	//std::cout<<labels<<std::endl;
	//exit(0);
}


void DPEDataGenerator::LoadTextData(int direction)  //+1 right, -1 left
{
	if(direction>0)
		direction = 1;
	else if(direction<0)
		direction = -1;
	
	int m = 0, n = 0, k = 0;
	//   std::cout << "Scoring plane id:" << section << " Out root filename:" << name << std::endl;
	
	std::ifstream ifs;
	ifs.open("trackone");
	
	long int i;
	//	*     NUMBER       TURN    X   PX    Y   PY    T   PT    S    E 
	//	$         %d         %*d %le  %le  %le  %le  %*le  %le  %le  %*le 
	
	
	std::string form1 = "%d %*hd %le %le %le %le %*le %le %le";
	std::string form2 = "%*s %*hd %*hd %hd %hd %s";
	std::string ln1;
	char ln2[200];
	char ln3[200];
	
	Double_t x1, px1, y1, py1, pt1;
	
	getline (ifs, ln1);
	k++;
	
	// skip comment line
	while (ln1[0] == '@'){
		getline (ifs, ln1);
		k++;
	}
	
	// skip 2 lines
	if (ln1[0] == '*'){
		getline (ifs, ln1);
		k++;
	}
	
	if (ln1[0] == '$'){
		getline (ifs, ln1);
		k++;
	}
	
	std::string scoring_plane;
	while( ifs.good() && !ifs.eof())
	{
		if( ln1[0] == '#' )
		{
			int m, n;
			sscanf (ln1.c_str (), "%*s %*hd %*hd %hd %hd %s", &m, &n, &ln2);
			std::cout << m << "|" << n << "|" << ln1 << std::endl;
			//ln2 marker name
			scoring_plane = ln2;
			getline(ifs,ln1);
			while( ifs.good() && !ifs.eof() && ln1[0] != '#')
			{
				scoring_plane_data data;
				sscanf (ln1.c_str(), form1.c_str(), &n, &data.x, &data.theta_x, &data.y, &data.theta_y, &data.xi, &data.s);
				data.accepted = 1;
				output_data_[direction][scoring_plane][n] = data;
//				std::cout<<ln1<<std::endl;
//				std::cout<<"direction: "<<direction<<"  n:"<<n<<std::endl;
				getline(ifs,ln1);
			}
		}
		else
		{
			getline(ifs,ln1);
		}
	}	
	ifs.close();
}


void DPEDataGenerator::AppendROOTFile(const MADProtonPairCollection &protons)
{
	std::cout<<"DPEDataGenerator::AppendROOTFile: "<<std::endl;
	std::cout<<"	setting branches..."<<std::endl;
	
	//std::cout<<"	mass branch"<<std::endl;
	out_tree_->SetBranchAddress(tree_mass_info_.mass_lab.c_str(), &tree_mass_info_.mass);
	
	//ip protons:  x,y,px,py,pz,xi,t_0,phi_0,thetax,thetay,xi_0

	//std::cout<<"	input left side"<<std::endl;
	out_tree_->SetBranchAddress(tree_ip_info_[-1].x_lab.c_str(), &tree_ip_info_[-1].data.x);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].y_lab.c_str(), &tree_ip_info_[-1].data.y);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].px_lab.c_str(), &tree_ip_info_[-1].data.px);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].py_lab.c_str(), &tree_ip_info_[-1].data.py);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].pz_lab.c_str(), &tree_ip_info_[-1].data.pz);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].xi_lab.c_str(), &tree_ip_info_[-1].data.xi);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].t_0_lab.c_str(), &tree_ip_info_[-1].data.t_0);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].phi_0_lab.c_str(), &tree_ip_info_[-1].data.phi_0);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].thetax_lab.c_str(), &tree_ip_info_[-1].data.thetax);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].thetay_lab.c_str(), &tree_ip_info_[-1].data.thetay);
	out_tree_->SetBranchAddress(tree_ip_info_[-1].xi_0_lab.c_str(), &tree_ip_info_[-1].data.xi_0);

	//std::cout<<"	input right side"<<std::endl;
	out_tree_->SetBranchAddress(tree_ip_info_[1].x_lab.c_str(), &tree_ip_info_[1].data.x);
	out_tree_->SetBranchAddress(tree_ip_info_[1].y_lab.c_str(), &tree_ip_info_[1].data.y);
	out_tree_->SetBranchAddress(tree_ip_info_[1].px_lab.c_str(), &tree_ip_info_[1].data.px);
	out_tree_->SetBranchAddress(tree_ip_info_[1].py_lab.c_str(), &tree_ip_info_[1].data.py);
	out_tree_->SetBranchAddress(tree_ip_info_[1].pz_lab.c_str(), &tree_ip_info_[1].data.pz);
	out_tree_->SetBranchAddress(tree_ip_info_[1].xi_lab.c_str(), &tree_ip_info_[1].data.xi);
	out_tree_->SetBranchAddress(tree_ip_info_[1].t_0_lab.c_str(), &tree_ip_info_[1].data.t_0);
	out_tree_->SetBranchAddress(tree_ip_info_[1].phi_0_lab.c_str(), &tree_ip_info_[1].data.phi_0);
	out_tree_->SetBranchAddress(tree_ip_info_[1].thetax_lab.c_str(), &tree_ip_info_[1].data.thetax);
	out_tree_->SetBranchAddress(tree_ip_info_[1].thetay_lab.c_str(), &tree_ip_info_[1].data.thetay);
	out_tree_->SetBranchAddress(tree_ip_info_[1].xi_0_lab.c_str(), &tree_ip_info_[1].data.xi_0);
	
	output_tree_shape::iterator it;
	std::map<std::string, scoring_plane_info>::iterator it_mark;

	//std::cout<<"	output branches"<<std::endl;
	for(it = tree_branches_.begin(); it!=tree_branches_.end(); ++it)
	{
		for(it_mark = it->second.begin(); it_mark != it->second.end(); ++it_mark)
		{
			//std::cout<<"	"<<it_mark->first<<std::endl;
			out_tree_->SetBranchAddress(it_mark->second.x_lab.c_str(), &it_mark->second.data.x);
			out_tree_->SetBranchAddress(it_mark->second.theta_x_lab.c_str(), &it_mark->second.data.theta_x);
			out_tree_->SetBranchAddress(it_mark->second.y_lab.c_str(), &it_mark->second.data.y);
			out_tree_->SetBranchAddress(it_mark->second.theta_y_lab.c_str(), &it_mark->second.data.theta_y);
			out_tree_->SetBranchAddress(it_mark->second.xi_lab.c_str(), &it_mark->second.data.xi);
			out_tree_->SetBranchAddress(it_mark->second.s_lab.c_str(), &it_mark->second.data.s);
			out_tree_->SetBranchAddress(it_mark->second.x_lab.c_str(), &it_mark->second.data.x);
			out_tree_->SetBranchAddress(it_mark->second.accepted_lab.c_str(), &it_mark->second.data.accepted);
		}
	}
	
	//std::cout<<"	branch addresses set"<<std::endl;
	
	for(int i=0; i<protons.size(); ++i)
	{
		tree_mass_info_.mass = protons[i].mass;
		tree_ip_info_[-1].data = protons[i].l;
		tree_ip_info_[1].data = protons[i].r;
		
		std::map<int, scoring_plane_data>::iterator data_it;
		for(it = tree_branches_.begin(); it!=tree_branches_.end(); ++it)
		{
			for(it_mark = it->second.begin(); it_mark != it->second.end(); ++it_mark)
			{
				data_it = output_data_[it->first][it_mark->first].find(i+1);
				if(data_it==output_data_[it->first][it_mark->first].end())
				{
					it_mark->second.data = scoring_plane_data();
					//std::cout<<it->first<<" "<<it_mark->first<<" not found"<<std::endl;
				}
				else
				{
					it_mark->second.data = data_it->second;
//					if(it->first==-1)
//						std::cout<<it->first<<" "<<it_mark->first<<" found"<<std::endl;
				}
			}
		}
		out_tree_->Fill();
	}
	
	//output_data_
//	output_data::iterator d_it;
//	std::map<std::string, std::map<int, scoring_plane_data> >::iterator d_mark_it;
//	std::map<int, scoring_plane_data>::iterator d_part_it;
	
//	for(d_it = output_data_.begin(); d_it!=output_data_.end(); ++d_it)
//	{
//		for(d_mark_it = d_it->second.begin(); d_mark_it!=d_it->second.end(); ++d_mark_it)
//		{
//			for(d_part_it = d_mark_it->second.begin(); d_part_it!=d_mark_it->second.end(); ++d_part_it)
//			{
//				std::cout<<d_it->first<<" "<<d_mark_it->first<<" "<<d_part_it->first<<std::endl;
//			}
//		}
//	}
	
	
	//save data
	out_file_->cd();
	out_tree_->Write(NULL, TObject::kOverwrite);
}

void DPEDataGenerator::CloseOutputROOTFile()
{
	out_file_->cd();
	out_tree_->Write(NULL, TObject::kOverwrite);
	
	delete out_tree_;
	out_tree_ = NULL;
	out_file_->Close();
}
