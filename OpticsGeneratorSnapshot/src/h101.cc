#define h101_cxx
#include "h101.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
#include <TVector3.h>
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/LorentzVector.h>


//beta=0.55 m
const double h101::mp0_ = 0.9382723128;  //Gev
const double h101::E0_ = 7e3;  // [GeV] cms energy of a proton
const double h101::p0_ = TMath::Sqrt(E0_*E0_ - mp0_*mp0_);

//parameters for beta*=0.55m
const double h101::BeamPosX_ = 0.0005; //m
const double h101::BeamPosY_ = 0.000; //m
const double h101::BeamSigmaX_ = 11.81e-6;
const double h101::BeamSigmaY_ = 11.81e-6;

const double h101::MeanAlpha_ = 142.5e-6;  //crossing angle [rad]
const double h101::SigmaTheta_ = 30.4e-6;  //beam divergence [rad]

const double h101::SigmaXi_ = 1e-4;
//const double h101::SigmaXi_ = 0;



/*
//beta=2 m
const double h101::mp0_ = 0.9382723128;  //Gev
const double h101::E0_ = 7e3;  // [GeV] cms energy of a proton
const double h101::p0_ = TMath::Sqrt(E0_*E0_ - mp0_*mp0_);

//parameters for beta*=0.55m
const double h101::BeamPosX_ = 0.0003220439972; //m
const double h101::BeamPosY_ = 0.000; //m
const double h101::BeamSigmaX_ = 22.51e-6;
const double h101::BeamSigmaY_ = 22.51e-6;

const double h101::MeanAlpha_ = 91.78299688e-6;  //crossing angle [rad]
const double h101::SigmaTheta_ = 22.5e-6;  //beam divergence [rad]
//const double h101::SigmaTheta_ = 0e-6;  //beam divergence [rad]

const double h101::SigmaXi_ = 1e-4;
//const double h101::SigmaXi_ = 0;
*/


/*
//beta=90 m
const double h101::mp0_ = 0.9382723128;  //Gev
const double h101::E0_ = 7e3;  // [GeV] cms energy of a proton
const double h101::p0_ = TMath::Sqrt(E0_*E0_ - mp0_*mp0_);

//parameters for beta*=0.55m
const double h101::BeamPosX_ = 0.0; //m
const double h101::BeamPosY_ = 0.0; //m
const double h101::BeamSigmaX_ = 212.7e-6;
const double h101::BeamSigmaY_ = 212.7e-6;

const double h101::MeanAlpha_ = 0.0;  //crossing angle [rad]
const double h101::SigmaTheta_ = 3.34e-6;  //beam divergence [rad]

const double h101::SigmaXi_ = 1e-4;
//const double h101::SigmaXi_ = 0;
*/







double h101::Momentum_to_t(double px, double py, double pz) const //GeV
{
	pz = TMath::Abs(pz);
	double p02 = E0_*E0_ - mp0_*mp0_;
	double p0 = TMath::Sqrt(p02);
	double E = TMath::Sqrt(mp0_*mp0_ + px*px+py*py+pz*pz);
	double dE2 = (E0_ - E)*(E0_ - E);
	double dp2 = px*px + py*py + (p0-pz)*(p0-pz);
	
	return dE2 - dp2;
}


double h101::Momentum_to_ksi(double px, double py, double pz) const //GeV
{
	double p02 = E0_*E0_ - mp0_*mp0_;
	double p0 = TMath::Sqrt(p02);
	double p2 = px*px + py*py + pz*pz;
	double p = TMath::Sqrt(p2);
	double ksi = (p - p0)/p0;
	return ksi;
}

void h101::WriteMADInputParticles(const MADProtonPairCollection &proton_pair_collection, int direction, std::string file_name)
{
	std::ofstream ofs(file_name.c_str());
	
	double x, theta_x, y, theta_y, ksi;
	
	ofs << "@ NAME             %07s \"PARTICLES\"" << std::endl;
	ofs << "@ TYPE             %04s \"USER\"" << std::endl;
	ofs << "@ TITLE            %34s \"EVENT\"" << std::endl;
	ofs << "@ ORIGIN           %19s \"MAD-X 3.00.03 Linux\"" << std::endl;
	ofs << "@ DATE             %08s \"22/02/06\"" << std::endl;
	ofs << "@ TIME             %08s \"11.11.11\"" << std::endl;
	
	ofs << "*   mken  trx      trpx       try     trpy       tt      tpt" << std::endl;
	ofs << "$   %s    %le      %le        %le     %le        %le     %le" << std::endl;
	
	
	for (unsigned int i = 0; i<proton_pair_collection.size(); i++)
	{
		if(direction>0)
		{
			ofs.precision(25);
			ofs << "    \"" << i + 1 << "\" " << proton_pair_collection[i].r.x << " " << proton_pair_collection[i].r.thetax << " " << proton_pair_collection[i].r.y 
			<< " " << proton_pair_collection[i].r.thetay << " 0.0 " << proton_pair_collection[i].r.xi << " " << std::endl;
		}
		else if(direction<0)
		{
			ofs.precision(25);
			ofs << "    \"" << i + 1 << "\" " << proton_pair_collection[i].l.x << " " << proton_pair_collection[i].l.thetax << " " << proton_pair_collection[i].l.y 
			<< " " << proton_pair_collection[i].l.thetay << " 0.0 " << proton_pair_collection[i].l.xi << " " << std::endl;
		}
	}
	ofs.close();
}


double h101::Momentum_to_phi(double px, double py, double pz) const //GeV
{
	TVector3 vec(px, py, pz);
	return vec.Phi();
}


//beg_evt: first event
//event_no: number of events
//direction: beam direction, >0 right, <0 left
MADProtonPairCollection h101::GetMADProtonPairs(Long64_t first_event, Long64_t event_no)
{
	if (fChain == 0)
		return MADProtonPairCollection();
	
	Long64_t nentries = fChain->GetEntries();
	Long64_t last_event = first_event + event_no;
	
	if(first_event>nentries)
		return MADProtonPairCollection();
	
	MADProtonPair pair;
	MADProtonPairCollection pair_collection;
	
	Long64_t nbytes = 0, nb = 0;
	for (Long64_t jentry=first_event; jentry<nentries && jentry<last_event; jentry++)
	{
		Long64_t ientry = LoadTree(jentry);
		if (ientry < 0) break;
		nb = fChain->GetEntry(jentry);   nbytes += nb;
		
		bool r_found = GetForwardProton(1, pair.r);
		bool l_found = GetForwardProton(-1, pair.l);
		pair.mass = sqrt(pair.r.xi_0 * pair.l.xi_0)*2.0*E0_;
		
		if(r_found && l_found)
		{
			SmearForwardProtonPair(pair);
			CorrectBeamCoordinateSigns(pair);
			pair_collection.push_back(pair);
		}
	}
	return pair_collection;
}


void h101::CorrectBeamCoordinateSigns(MADProtonPair &proton_pair)
{
	proton_pair.l.x = -proton_pair.l.x;
	proton_pair.l.thetax = -proton_pair.l.thetax;
}


bool h101::GetForwardProton(int direction, MADProton &proton)
{
	bool found = false;
	for(int i=0; i<Nhep; i++)
	{
		int ISTHEP = Jsdhep[i]/16000000*100 + Jsmhep[i]/16000000;
		if(ISTHEP==1 && Idhep[i]==2212 && Jsdhep[i]==0 && Phep[i][3]>3500 && Phep[i][2]*direction>0)
		{
			proton.x = 0;
			proton.y = 0;
			proton.px = Phep[i][0];
			proton.py = Phep[i][1];
			proton.pz = Phep[i][2];
			proton.xi_0 = Momentum_to_ksi(proton.px, proton.py, proton.pz);
			proton.t_0 = Momentum_to_t(proton.px, proton.py, proton.pz);
			proton.phi_0 = Momentum_to_phi(proton.px, proton.py, proton.pz);
			found = true;
			break;
		}
	}
	
	return found;
}


void h101::SmearForwardProtonPair(MADProtonPair &proton_pair)
{
	/// generate energy/angle smearing
	double al1 = rand->Gaus(MeanAlpha_, 0);
	double al2 = -rand->Gaus(MeanAlpha_, 0);
	double ph1 = rand->Rndm() * 2. * TMath::Pi();
	double ph2 = rand->Rndm() * 2. * TMath::Pi();
	double th1 = rand->Gaus(0, SigmaTheta_);
	double th2 = rand->Gaus(0, SigmaTheta_);
	double xi1 = rand->Gaus(0, SigmaXi_);
	double xi2 = rand->Gaus(0, SigmaXi_);
	
	/// compute transform parameters
	double m = mp0_;
	double p_nom = p0_;
	
	CLHEP::Hep3Vector p1(cos(al1)*sin(th1)*cos(ph1) + sin(al1)*cos(th1), sin(th1)*sin(ph1), -sin(al1)*sin(th1)*cos(ph1) + cos(al1)*cos(th1));  
	p1 *= (1. + xi1) * p_nom;
	double E1 = sqrt(p1.mag2() + m*m);
	CLHEP::HepLorentzVector P1(p1, E1);
	
	CLHEP::Hep3Vector p2(cos(al2)*sin(th2)*cos(ph2) + sin(al2)*cos(th2), sin(th2)*sin(ph2), -sin(al2)*sin(th2)*cos(ph2) + cos(al2)*cos(th2));  
	p2 *= -(1. + xi2) * p_nom;
	double E2 = sqrt(p2.mag2() + m*m);
	CLHEP::HepLorentzVector P2(p2, E2);
	
	double factor = (P1 + P2).mag() / 2. / E0_;             /// energy correction factor
	
	CLHEP::Hep3Vector dir = p1 + p2;                       /// boost direction
	double beta = dir.mag() / (E1 + E2);                  /// beta of boost
	P1.boost(dir, -beta);
	CLHEP::Hep3Vector axis(P1.y(), -P1.x(), 0.);                 /// rotation axis
	double angle = -acos( P1.v().z() / P1.v().mag() );            /// angle of rotation
	
	SmearProton(proton_pair.r, angle, beta, factor, axis, dir);
	SmearProton(proton_pair.l, angle, beta, factor, axis, dir);
}


void h101::SmearProton(MADProton &proton, double angle, double beta, double factor, const CLHEP::Hep3Vector &axis, const CLHEP::Hep3Vector &dir)
{
	CLHEP::HepLorentzVector P = CLHEP::HepLorentzVector(proton.px, proton.py, proton.pz, (1.0+proton.xi_0)*E0_);
	//CLHEP::HepLorentzVector P = CLHEP::HepLorentzVector(0, 0, p0_, E0_);
	/// energy scaling
	CLHEP::Hep3Vector p = P.vect();
	double E = P.e() * factor, m = P.m();
	p = sqrt(E*E - m*m) / p.mag() * p;
	P = CLHEP::HepLorentzVector(p, E);
	
	/// rotation
	if (fabs(angle) > 1E-8) P = P.rotate(axis, angle);
	
	/// boost
	if (fabs(beta) > 1E-8) P = P.boost(dir, beta);
	proton.px = P.x();
	proton.py = P.y();
	proton.pz = P.z();
	proton.xi = (sqrt(P.x()*P.x() + P.y()*P.y() + P.z()*P.z()) - p0_)/p0_;
	proton.thetax = proton.px/p0_;
	proton.thetay = proton.py/p0_;
	
	proton.x = rand->Gaus(BeamPosX_, BeamSigmaX_);
	proton.y = rand->Gaus(BeamPosY_, BeamSigmaY_);
	
	//just for test purposes
	//	proton.px = 0;
	//	proton.py = 0;
	//	proton.pz = 7000;
	//proton.xi = 0;
	//proton.thetax = 142.5e-6;
	//proton.thetay = 0;
	
	//	proton.x = BeamPosX_;
	//	proton.y = BeamPosY_;
}


Long64_t h101::GetEntries()
{
	if(fChain)
		return fChain->GetEntries();
	else return 0;
}


void h101::Loop()
{
	//   In a ROOT session, you can do:
	//      Root > .L h101.C
	//      Root > h101 t
	//      Root > t.GetEntry(12); // Fill t data members with entry number 12
	//      Root > t.Show();       // Show values of entry 12
	//      Root > t.Show(16);     // Read and show values of entry 16
	//      Root > t.Loop();       // Loop on all entries
	//
	
	//     This is the loop skeleton where:
	//    jentry is the global entry number in the chain
	//    ientry is the entry number in the current Tree
	//  Note that the argument to GetEntry must be:
	//    jentry for TChain::GetEntry
	//    ientry for TTree::GetEntry and TBranch::GetEntry
	//
	//       To read only selected branches, Insert statements like:
	// METHOD1:
	//    fChain->SetBranchStatus("*",0);  // disable all branches
	//    fChain->SetBranchStatus("branchname",1);  // activate branchname
	// METHOD2: replace line
	//    fChain->GetEntry(jentry);       //read all branches
	//by  b_branchname->GetEntry(ientry); //read only this branch
	if (fChain == 0)
		return;
	
	Long64_t nentries = fChain->GetEntriesFast();
	
	Long64_t nbytes = 0, nb = 0;
	for (Long64_t jentry=0; jentry<nentries;jentry++)
	{
		Long64_t ientry = LoadTree(jentry);
		if (ientry < 0) break;
		nb = fChain->GetEntry(jentry);   nbytes += nb;
		// if (Cut(ientry) < 0) continue;
		
		for(int i=0; i<Nhep; i++)
		{
			int ISTHEP = Jsdhep[i]/16000000*100 + Jsmhep[i]/16000000;
			if(ISTHEP==1 && Idhep[i]==2212 && Jsdhep[i]==0 && Phep[i][3]>3500)
			{
				std::cout<<jentry<<"\t"<<i<<"\t"<<Jsdhep[i]<<"\t"<<ISTHEP<<"\t";
				for(int j=0; j<4; j++)
				{
					std::cout<<Phep[i][j]<<",\t";
				}
				double ksi = (Phep[i][3]-7000)/7000;
				std::cout<<ksi<<std::endl;
			}
		}
		std::cout<<std::endl;
	}
}



//ISTHEP is the particle status which has to be = 1 to have a "final particle"

