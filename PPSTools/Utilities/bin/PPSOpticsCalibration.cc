#include "PPSTools/Utilities/interface/PPSOpticsCalibrator.h"
#include <TEllipse.h>
#include <iomanip>
//using namespace std;
int main(int argc, char** argv)
{
    int Nevents = 1000000;
    float BeamE = 6500.;
    int   BeamL = 250.;
    float Xangle1 = 0.;
    float Xangle2 = 0.;
    bool  Align = false;
    bool  FixIP = false;
    std::string Beam1FileName = "";
    std::string Beam2FileName = "";
    std::string OutputFileName = "";

    // Declare the supported options.
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
         ("help"            ,"produce help message")
         ("Beam1"           ,po::value<std::string>(),"Beam 1 filename")
         ("Beam2"           ,po::value<std::string>(),"Beam 2 filename")
         ("Xangle1"         ,po::value<float>(), "Half Crossing angle Beam 1")
         ("Xangle2"         ,po::value<float>(), "Half Crossing angle Beam 2")
         ("BeamE"           ,po::value<float>(&BeamE)->default_value(6500.), "Beam energy (6500 GeV)")
         ("BeamL"           ,po::value<int>(&BeamL)->default_value(250)  , "Beam line length (250 m)")
         ("Nevents"         ,po::value<int>(&Nevents)->default_value(1000000)  , "Number of events (1M)")
         ("Align"           ,po::value<bool>(&Align)->default_value(false)  , "Align beamline (0-> No     1-> Yes")
         ("FixIP"           ,po::value<bool>(&FixIP)->default_value(false)  , "Fix beam at IP (0-> No     1-> Yes")
         ("output"          ,po::value<std::string>(&OutputFileName)->default_value("beamprofile.root"),"Output filename")
    ;
 
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) { std::cout << desc << "\n"; return 1; }

    //if (vm.count("Nevents")) Nevents=vm["Nevents"].as<int>();
    //else {std::cout << "Give the number of events: "; std::cin >> Nevents;}

    //if (vm.count("BeamL")) BeamL=vm["BeamL"].as<int>();
    //else {std::cout << "Beam line length not given. Using default 250 m"<<std::endl;}

    //if (vm.count("BeamE")) BeamE=vm["BeamE"].as<int>();
    //else {std::cout << "Beam line energy not given. Using default 6500 GeV"<<std::endl;}

    //if (vm.count("output")) FileName = vm["output"].as<std::string>();
    //if (FileName=="") {std::cout << "Output file name not given. Results will not be saved." << std::endl;}

    if (vm.count("Beam1"))  Beam1FileName = vm["Beam1"].as<std::string>();
    else {std::cout << "Give the beam 1 filename : "; std::cin >> Beam1FileName;}
    if (Beam1FileName.empty()) {std::cout << "Beam 1 file name not given. Exiting." << std::endl;exit(1);}

    if (vm.count("Beam2"))  Beam2FileName = vm["Beam2"].as<std::string>();
    else {std::cout << "Give the beam 2 filename : "; std::cin >> Beam2FileName;}
    if (Beam2FileName.empty()) {std::cout << "Beam 2 file name not given. Exiting." << std::endl;exit(1);}

    if (vm.count("Xangle1")) Xangle1 = vm["Xangle1"].as<float>();
    else {std::cout << "Give the beam 1 crossing angle : "; std::cin >> Xangle1;} 

    if (vm.count("Xangle2")) Xangle2 = vm["Xangle2"].as<float>();
    else {std::cout << "Give the beam 2 crossing angle : "; std::cin >> Xangle2;} 

    TFile* fout = nullptr;
    if (!OutputFileName.empty()) fout = new TFile(OutputFileName.c_str(),"recreate");

    std::ifstream tabfileB1(Beam1FileName.c_str());
    if (! tabfileB1.is_open()) cout << "\t ERROR: I Can't open \"" << Beam1FileName << "\"" << endl;
    std::ifstream tabfileB2(Beam2FileName.c_str());
    if (! tabfileB2.is_open()) cout << "\t ERROR: I Can't open \"" << Beam2FileName << "\"" << endl;

    std::vector<double> pos;
    pos.push_back(212.55);
    pos.push_back(215.7);
    pos.push_back(219.55);
    pos.push_back(-212.55);
    pos.push_back(-215.7);
    pos.push_back(-219.55);

    PPSOpticsCalibrator* ctpps_optc = new PPSOpticsCalibrator(Beam1FileName,Beam2FileName,BeamL);
    PPSTools::fBeamEnergy=BeamE;
    PPSTools::fBeamMomentum=sqrt(BeamE*BeamE-PPSTools::ProtonMassSQ);
    PPSTools::fCrossingAngleBeam1=Xangle1; // Boost is done in CMS ref frame, in which Xangle is negative
    PPSTools::fCrossingAngleBeam2=Xangle2; // ibdem
    ctpps_optc->ReadBeamPositionFromOpticsFile(tabfileB1);
    ctpps_optc->ReadBeamPositionFromOpticsFile(tabfileB2);
    if (Align) ctpps_optc->AlignBeamLine();
    double beamX=0.;
    double beamY=0.;
    if (FixIP) ctpps_optc->CalibrateBeamPositionatIP(beamX,beamY);
    ctpps_optc->BeamProfile(fout,Nevents);
    fout->Close();
};

PPSOpticsCalibrator::PPSOpticsCalibrator(const std::string b1,const std::string b2, int len):beamline_length(len+0.1)
{
        IP_Found=false;
        ParIdx_Found=false;
        Emmitance_OK=false;
        emittanceX=0.;
        emittanceY=0.;

        m_beamline56 = std::unique_ptr<H_BeamLine>(new H_BeamLine( 1, beamline_length )); // (direction, length)
        m_beamline45 = std::unique_ptr<H_BeamLine>(new H_BeamLine(-1, beamline_length )); //
        m_beamline56->fill( b1, 1, "IP5");
        m_beamline45->fill( b2,-1, "IP5");
        m_beamline45->offsetElements( 120, 0.097 );
        m_beamline56->offsetElements( 120,-0.097 );
/*
        std::cout << "========================== Beam line sector 45 =============================="<<std::endl;
        m_beamline45->showElements();
        std::cout << "========================== Beam line sector 56 =============================="<<std::endl;
        m_beamline56->showElements();
*/
};
void PPSOpticsCalibrator::AlignBeamLine()
{
     
     H_BeamLine* beamline=nullptr;
     for(int j:{45,56}) {
        if (j==45) continue;
        switch (j) {
               case 45: beamline = &*m_beamline45; break;
               case 56: beamline = &*m_beamline56; break;
        }
        int NumberOfElements = beamline->getNumberOfElements();
        std::cout << "Before the alingment"<< std::endl;
        for(int i=0;i<NumberOfElements;i++) {
            H_OpticalElement* opt = const_cast<H_OpticalElement*>(beamline->getElement(i));
            string name=opt->getName();
            if (OptElementType(opt->getTypeString())==Invalid) continue;
            if (opt->getTypeString().find("Drift")<opt->getTypeString().length()) continue;
            if (name[0]=='"') name=name.substr(1,name.length());
            if (name[name.length()-1]=='"') name=name.substr(0,name.length()-1);
            //if (opt->getK()!=0) AlignObject(opt,beamline,.005);
            beamline->alignElement(opt->getName(),0.1,0.1);
            //beamline->tiltElement(opt->getName(),0.01,0.001);
            std::cout << "Verifying the alingment"<< std::endl;
            H_BeamParticle h_p;
            h_p.setPosition(fBeamXatIP*mm_to_um,fBeamYatIP*mm_to_um,0.,0.,0.);
            h_p.computePath(beamline);
            for(int j=NumberOfElements-1;j>=0;j--) {
               H_OpticalElement* opt = const_cast<H_OpticalElement*>(beamline->getElement(j));
               if (OptElementType(opt->getTypeString())==Invalid) continue;
               h_p.propagate(opt->getS());
               printf("Element %-20s %-20s X = % 8.4f RelX = % 8.4f beam pos X % 8.4f aligned by X = % 8.4f\n",
                       opt->getName().c_str(),opt->getTypeString().c_str(),opt->getX(),opt->getRelX(),h_p.getX(),opt->getRelX()-h_p.getX());
               break;
            }
        } 
        std::cout << "==========================================="<<std::endl;
     }
     return;
}
void PPSOpticsCalibrator::CalibrateBeamPositionatIP(double& xpos,double& ypos)
{
     xpos=fBeamXatIP;
     xpos=fBeamYatIP;
     double sigx_target=0.001;
     double sigy_target=0.001;
     //double min_deltax=1e-4;
     //double min_deltay=1e-4;
//
     double sigx = 0.;
     double sigy = 0.;
     double sigx_min = 99.;
     double sigy_min = 99.;
     double last_sigx=0.;
     double last_sigy=0.;
     double rel_diff_x = 0;
     double rel_diff_y = 0;
//
     int dirx=1;
     int diry=1;
     int nInteractions=0;
     int nDirchgx = 0; // number of direction changing with current delta
     int nDirchgy = 0; // number of direction changing with current delta
     double deltax=1.0;
     double deltay=1.0;

     double fVtxMeanZ=0.;
     while(true) {
        nInteractions++;
        H_BeamParticle h_pp; H_BeamParticle h_pn;

        PPSTools::LorentzBoost(h_pp,1,"LAB");
        PPSTools::LorentzBoost(h_pn,-1,"LAB");

        h_pp.setPosition(-xpos*mm_to_um,ypos*mm_to_um,h_pp.getTX(),h_pp.getTY(),-fVtxMeanZ*cm_to_m); // the position is given in the CMS frame
        h_pn.setPosition(-xpos*mm_to_um,ypos*mm_to_um,h_pn.getTX(),h_pn.getTY(),-fVtxMeanZ*cm_to_m);
        h_pp.computePath(&*m_beamline45); h_pn.computePath(&*m_beamline56);
        sigx=0.;
        sigy=0.;
        BdistP.clear();
        BdistN.clear();
        int z1=1;
        int z2 = 3;
        int z3 = 4;
        for(unsigned int i:{z1,z2,z3}){
           h_pp.propagate(std::get<0>(PosP.at(i)));
           h_pn.propagate(std::get<0>(PosN.at(i)));

           double xp = std::get<1>(PosP.at(i)); double yp = std::get<2>(PosP.at(i)); // the beam position is given as positive (LHC frame)
           double xn = std::get<1>(PosN.at(i)); double yn = std::get<2>(PosN.at(i)); // ibdem

           BdistP.push_back(make_tuple<double,double,double>((double)std::get<0>(PosP.at(i)),h_pp.getX()*um_to_mm,h_pp.getY()*um_to_mm));
           sigx+=(pow((h_pp.getX()*um_to_mm-xp),2)+pow((h_pn.getX()*um_to_mm-xn),2));
           BdistN.push_back(make_tuple<double,double,double>((double)std::get<0>(PosN.at(i)),h_pn.getX()*um_to_mm,h_pn.getY()*um_to_mm));

           sigy+=(pow((h_pp.getY()*um_to_mm-yp),2)+pow((h_pn.getY()*um_to_mm-yn),2));
        }
        sigx=sqrt(sigx);
        sigy=sqrt(sigy);
        if (sigx<sigx_min) {
           sigx_min=sigx; // good, go on in this path
        } else {
           if (sigx>last_sigx) { dirx*=-1;nDirchgx++;}  // change direction
           if (nInteractions>1) {deltax*=0.9;nDirchgx=0;} // decrease delta
        }
        if (sigy<sigy_min) {
           sigy_min=sigy;
        } else {
           if (sigy>last_sigy) {diry*=-1;nDirchgy++;}
           if (nInteractions>1) {deltay*=0.9;nDirchgy=0;}
        }
        last_sigx=sigx;
        last_sigy=sigy;
        rel_diff_x =abs(sigx_min-last_sigx)/sigx_min;
        rel_diff_y =abs(sigy_min-last_sigy)/sigy_min;
        if (nInteractions>5&&rel_diff_x<sigx_target&&deltax<0.0001) deltax=0.;
        if (nInteractions>5&&rel_diff_y<sigy_target&&deltay<0.000001) deltay=0.;
        if (deltax==0&&deltay==0) break;
        xpos-=(dirx*deltax);
        ypos-=(diry*deltay);
        h_pp.resetPath();
        h_pn.resetPath();
     }
     //if (m_verbosity){
        //LogDebug("PPSHector::BeamPositionCalibration") 
          std::cout <<std::right<< std::setw(10) << std::setprecision(6) << std::fixed 
               << "Interaction number " << nInteractions << "\tX = " << xpos << " (mm) \tSigmaX = " << sigx_min << "\tDeltaX = " << deltax << "\n"
               << "                   "                  << "\tY = " << ypos << " (mm) \tSigmaY = " << sigy_min << "\tDeltaY = " << deltay << "\n"
               << "Calibrated beam positions: (in mm)\n"
               << " Z (m)   \t X (twiss) \t X (calib) \t Y (twiss) \t Y (calib) \t Delta X \t Delta Y\n";
        for(unsigned int i=0;i<BdistP.size();i++) {
            //LogDebug("PPSHector::BeamPositionCalibration")
                std::cout << std::setw(10) << std::setprecision(6) << std::fixed
                <<  std::get<0>(BdistP.at(i))<< " \t "<< std::get<1>(PosP.at(i))<< " \t "<< std::get<1>(BdistP.at(i))
                                              << " \t "<< std::get<2>(PosP.at(i))<< " \t " << std::get<2>(BdistP.at(i))
                                              << " \t "<< std::get<1>(BdistP.at(i))-std::get<1>(PosP.at(i))
                                              << " \t "<< std::get<2>(BdistP.at(i))-std::get<2>(PosP.at(i))
                                              << "\n";
        }
        for(unsigned int i=0;i<BdistN.size();i++) {
           //LogDebug("PPSHector::BeamPositionCalibration")
                std::cout
                 << -std::get<0>(BdistN.at(i))<< " \t "<<std::get<1>(PosN.at(i)) << " \t "<< std::get<1>(BdistN.at(i))
                                              << " \t "<<std::get<2>(PosN.at(i)) << " \t "<<std::get<2>(BdistN.at(i))
                                              << " \t "<<std::get<1>(BdistN.at(i))-std::get<1>(PosN.at(i))
                                              << " \t "<<std::get<2>(BdistN.at(i))-std::get<2>(PosN.at(i))
                                              << "\n";
        }
     //}
     fBeamXatIP=xpos;
     fBeamYatIP=ypos;
     return;
}
void PPSOpticsCalibrator::BeamProfile(TFile* fout,int Nevents)
{

     if (fout) fout->cd();
     double averageX_bp = 0;
     double averageY_bp = 0;
     double maxSX = 0;
     double maxSY = 0;
// Define the index to the position where the profile is to be drawn
     int z1 = 1;
     int z2 = 3;
     int z3 = 4;
     for(int i:{z1,z2,z3}) {
        averageX_bp+=std::get<1>(PosP.at(i)); averageY_bp+=std::get<2>(PosP.at(i));
        averageX_bp+=std::get<1>(PosN.at(i)); averageY_bp+=std::get<2>(PosN.at(i));
        maxSX = max(std::get<3>(PosP.at(i)),maxSX);
        maxSX = max(std::get<3>(PosN.at(i)),maxSX);
        maxSY = max(std::get<4>(PosP.at(i)),maxSY);
        maxSY = max(std::get<4>(PosN.at(i)),maxSY);
     }
     averageX_bp/=6; averageY_bp/=6;
     averageX_bp*=10;averageY_bp*=10; // next hundreths
     averageX_bp = ceil(averageX_bp)*100;
     averageY_bp = ceil(averageY_bp)*100;
     int width = (int)max(ceil(maxSX*10),ceil(maxSY*10));
     width*=100; // final convertion to microns
     width*=3.5;   // use 5 sigmas 
     TH2F* bp1f = new TH2F("bp1f",Form("Z=%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosP.at(z1))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);
     TH2F* bp2f = new TH2F("bp2f",Form("Z=%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosP.at(z2))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);
     TH2F* bp3f = new TH2F("bp3f",Form("Z=%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosP.at(z3))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);
     TH2F* bp1b = new TH2F("bp1b",Form("Z=-%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosN.at(z1))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);
     TH2F* bp2b = new TH2F("bp2b",Form("Z=-%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosN.at(z2))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);
     TH2F* bp3b = new TH2F("bp3b",Form("Z=-%5.2f (m);X(#mum);Y(#mum)",std::get<0>(PosN.at(z3))),
                            1000,averageX_bp-width,averageX_bp+width,1000,averageY_bp-width,averageY_bp+width);

     TEllipse* el1f = new TEllipse(std::get<1>(PosP.at(z1))*mm_to_um,std::get<2>(PosP.at(z1))*mm_to_um,std::get<3>(PosP.at(z1))*mm_to_um,std::get<4>(PosP.at(z1))*mm_to_um);
     TEllipse* el2f = new TEllipse(std::get<1>(PosP.at(z2))*mm_to_um,std::get<2>(PosP.at(z2))*mm_to_um,std::get<3>(PosP.at(z2))*mm_to_um,std::get<4>(PosP.at(z2))*mm_to_um);
     TEllipse* el3f = new TEllipse(std::get<1>(PosP.at(z3))*mm_to_um,std::get<2>(PosP.at(z3))*mm_to_um,std::get<3>(PosP.at(z3))*mm_to_um,std::get<4>(PosP.at(z3))*mm_to_um);
     TEllipse* el1b = new TEllipse(std::get<1>(PosN.at(z1))*mm_to_um,std::get<2>(PosN.at(z1))*mm_to_um,std::get<3>(PosN.at(z1))*mm_to_um,std::get<4>(PosN.at(z1))*mm_to_um);
     TEllipse* el2b = new TEllipse(std::get<1>(PosN.at(z2))*mm_to_um,std::get<2>(PosN.at(z2))*mm_to_um,std::get<3>(PosN.at(z2))*mm_to_um,std::get<4>(PosN.at(z2))*mm_to_um);
     TEllipse* el3b = new TEllipse(std::get<1>(PosN.at(z3))*mm_to_um,std::get<2>(PosN.at(z3))*mm_to_um,std::get<3>(PosN.at(z3))*mm_to_um,std::get<4>(PosN.at(z3))*mm_to_um);
     el1f->SetFillStyle(0); el2f->SetFillStyle(0); el3f->SetFillStyle(0); el1b->SetFillStyle(0); el2b->SetFillStyle(0); el3b->SetFillStyle(0);
     el1f->SetLineWidth(2); el2f->SetLineWidth(2); el3f->SetLineWidth(2); el1b->SetLineWidth(2); el2b->SetLineWidth(2); el3b->SetLineWidth(2);
     bp1f->GetListOfFunctions()->Add(el1f); bp2f->GetListOfFunctions()->Add(el2f); bp3f->GetListOfFunctions()->Add(el3f);
     bp1b->GetListOfFunctions()->Add(el1b); bp2b->GetListOfFunctions()->Add(el2b); bp3b->GetListOfFunctions()->Add(el3b);

     for(int j=0;j<2;j++) {
        std::vector<std::tuple<double,double,double,double,double> >* DetPos;
        TH2F* prof1;TH2F* prof2; TH2F* prof3;
        H_BeamLine* beamline;
        int direction=0;
        switch(j) {
              case 0:  // positive side
                     prof1 = &(*bp1f);prof2 = &(*bp2f);prof3 = &(*bp3f);
                     beamline= &(*m_beamline45);   // PPS1 corresponds to beam2
                     DetPos = &(PosP);
                     direction=1;
                     break;
              case 1: // negative side
                     prof1 = &(*bp1b);prof2 = &(*bp2b); prof3 = &(*bp3b);
                     beamline= &(*m_beamline56);   // PPS2 corresponds to beam1
                     DetPos = &(PosN);
                     direction=-1;
                     break;
        }
        double fVtxMeanZ=0.;
        for(int i=0;i<Nevents;i++) {
           H_BeamParticle h_p; // Hector always gives a positive pz
           PPSTools::LorentzBoost(h_p,direction,"LAB");
           h_p.setPosition(-fBeamXatIP*mm_to_um,fBeamYatIP*mm_to_um,h_p.getTX(),h_p.getTY(),-fVtxMeanZ*cm_to_m);

           m_sigE=1.1e-4;
           h_p.smearPos(m_sigmaSX,m_sigmaSY); h_p.smearAng(m_sigmaSTX,m_sigmaSTY); h_p.smearE(m_sigE);
           h_p.computePath(beamline);
           h_p.propagate(std::get<0>(DetPos->at(z1))); prof1->Fill(h_p.getX(),h_p.getY());
           h_p.propagate(std::get<0>(DetPos->at(z2))); prof2->Fill(h_p.getX(),h_p.getY());
           h_p.propagate(std::get<0>(DetPos->at(z3))); prof3->Fill(h_p.getX(),h_p.getY());
           h_p.resetPath();
        } 
        TEllipse *el1 = new TEllipse(prof1->GetMean(1),prof1->GetMean(2),prof1->GetRMS(1),prof1->GetRMS(2));
        TEllipse *el2 = new TEllipse(prof2->GetMean(1),prof2->GetMean(2),prof2->GetRMS(1),prof2->GetRMS(2));
        TEllipse *el3 = new TEllipse(prof3->GetMean(1),prof3->GetMean(2),prof3->GetRMS(1),prof3->GetRMS(2));
        el1->SetLineWidth(2);el1->SetLineColor(0);el1->SetFillStyle(0);
        el2->SetLineWidth(2);el2->SetLineColor(0);el2->SetFillStyle(0);
        el3->SetLineWidth(2);el3->SetLineColor(0);el3->SetFillStyle(0);
        prof1->GetListOfFunctions()->Add(el1);
        prof2->GetListOfFunctions()->Add(el2);
        prof3->GetListOfFunctions()->Add(el3);
     }
     double _beamX_Det1_f    = bp1f->GetMean(1); double _beamY_Det1_f    = bp1f->GetMean(2);
     double _beamSigX_Det1_f = bp1f->GetRMS(1);  double _beamSigY_Det1_f = bp1f->GetRMS(2);
     double _beamX_Det2_f    = bp2f->GetMean(1); double _beamY_Det2_f    = bp2f->GetMean(2);
     double _beamSigX_Det2_f = bp2f->GetRMS(1);  double _beamSigY_Det2_f = bp2f->GetRMS(2);
     double _beamX_Det3_f    = bp3f->GetMean(1); double _beamY_Det3_f    = bp3f->GetMean(2);
     double _beamSigX_Det3_f = bp3f->GetRMS(1);  double _beamSigY_Det3_f = bp3f->GetRMS(2);
     double _beamX_Det1_b    = bp1b->GetMean(1); double _beamY_Det1_b    = bp1b->GetMean(2);
     double _beamSigX_Det1_b = bp1b->GetRMS(1);  double _beamSigY_Det1_b = bp1b->GetRMS(2);
     double _beamX_Det2_b    = bp2b->GetMean(1); double _beamY_Det2_b    = bp2b->GetMean(2);
     double _beamSigX_Det2_b = bp2b->GetRMS(1);  double _beamSigY_Det2_b = bp2b->GetRMS(2);
     double _beamX_Det3_b    = bp3b->GetMean(1); double _beamY_Det3_b    = bp3b->GetMean(2);
     double _beamSigX_Det3_b = bp3b->GetRMS(1);  double _beamSigY_Det3_b = bp3b->GetRMS(2);

     //LogDebug("PPSHector::BeamProfile")
     std::cout <<std::right<<std::setw(10) << std::setprecision(1) << std::fixed;
     std::cout << "HectorForPPS: BEAM parameters (in um):\n" 
             << "Beam position at Det1 positive side --> " << _beamX_Det1_f << "("<< _beamSigX_Det1_f<<"),\t"<< _beamY_Det1_f << "("<< _beamSigY_Det1_f<<")\n"
             << "Beam position at Det2 positive size --> " << _beamX_Det2_f << "("<< _beamSigX_Det2_f<<"),\t"<< _beamY_Det2_f << "("<< _beamSigY_Det2_f<<")\n"
             << "Beam position at ToF  positive size --> " << _beamX_Det3_f << "("<< _beamSigX_Det3_f<<"),\t"<< _beamY_Det3_f << "("<< _beamSigY_Det3_f<<")\n"
             << "Beam position at Det1 negative side --> " << _beamX_Det1_b << "("<< _beamSigX_Det1_b<<"),\t"<< _beamY_Det1_b << "("<< _beamSigY_Det1_b<<")\n"
                << "Beam position at Det2 negative size --> " << _beamX_Det2_b << "("<< _beamSigX_Det2_b<<"),\t"<< _beamY_Det2_b << "("<< _beamSigY_Det2_b<<")\n"
                << "Beam position at ToF  negative size --> " << _beamX_Det3_b << "("<< _beamSigX_Det3_b<<"),\t"<< _beamY_Det3_b << "("<< _beamSigY_Det3_b<<")\n"
             << "\nBeam positions displacement to the closed orbit (im mm):\n" << std::setprecision(3)
             << "at Det1 positive side  --> X= " << _beamX_Det1_f*um_to_mm-std::get<1>(PosP.at(z1)) << "\tY= "<< _beamY_Det1_f*um_to_mm-std::get<2>(PosP.at(z1))<< "\n"
             << "at Det2 positive side  --> X= " << _beamX_Det2_f*um_to_mm-std::get<1>(PosP.at(z2)) << "\tY= "<<_beamY_Det2_f*um_to_mm-std::get<2>(PosP.at(z2))<< "\n"
             << "at ToF  positive side  --> X= " << _beamX_Det3_f*um_to_mm-std::get<1>(PosP.at(z3)) << "\tY= "<<_beamY_Det3_f*um_to_mm-std::get<2>(PosP.at(z3))<<"\n"
             << "at Det1 negative side  --> X= " << _beamX_Det1_b*um_to_mm-std::get<1>(PosN.at(z1)) << "\tY= "<< _beamY_Det1_b*um_to_mm-std::get<2>(PosN.at(z1))<<"\n"
             << "at Det2 negative side  --> X= " << _beamX_Det2_b*um_to_mm-std::get<1>(PosN.at(z2)) << "\tY= "<< _beamY_Det2_b*um_to_mm-std::get<2>(PosN.at(z2))<<"\n"
             << "at ToF  negative side  --> X= " << _beamX_Det3_b*um_to_mm-std::get<1>(PosN.at(z3)) << "\tY= "<< _beamY_Det3_b*um_to_mm-std::get<2>(PosN.at(z3))<<"\n";
     bp1f->Write();
     bp2f->Write();
     bp3f->Write();
     bp1b->Write();
     bp2b->Write();
     bp3b->Write();
     delete bp1f;delete bp2f;delete bp3f;
     delete bp1b;delete bp2b;delete bp3b;
}

void PPSOpticsCalibrator::ReadEmittance(std::ifstream& tabfile)
{
     if (!tabfile.good()) {
        std::cout << "Bad TWISS file. Could not read emittance."<<std::endl;
        return;
     }
     std::string temp_string;
     std::istringstream curstring;
     while (std::getline(tabfile,temp_string)&&(emittanceX==0||emittanceY==0)) {
            string dummy;
            curstring.clear(); // needed when using istringstream::str(string) several times !
            curstring.str(temp_string);
            if (temp_string.find("@ EX")<temp_string.length()) {
               curstring >> dummy >> dummy >> dummy >> emittanceX;
               continue;
            }
            if (temp_string.find("@ EY")<temp_string.length()) {
               curstring >> dummy >> dummy >> dummy >> emittanceY;
               continue;
            }
     }
     tabfile.clear();
     tabfile.seekg(0);
     return;
}

void PPSOpticsCalibrator::ReadParameterIndex(std::ifstream& tabfile)
{
     if (!tabfile.good()) {
        std::cout << "Bad TWISS file. Could not read emittance."<<std::endl;
        return;
     }
     int N_col=0;
     std::string temp_string;
     std::istringstream curstring;
     while (std::getline(tabfile,temp_string)) {
           if (temp_string[0]=='@') continue;
           if (temp_string[0]=='$') continue;
           if (temp_string.find("NAME")<temp_string.length()) {
              curstring.str(temp_string);
              std::string header;
              while(curstring.good()) { //curstring >> headers[N_col]; if(headers[N_col]!="*") N_col++;}
                   curstring >> header;
                   if (header=="*") continue; // skip first field
                   N_col++;
                   if (header=="NAME") continue;
                   else if (header=="S") {s_idx = N_col-1;continue;}
                   else if (header=="X") {x_idx = N_col-1;continue;}
                   else if (header=="Y") {y_idx = N_col-1;continue;}
                   else if (header=="BETX") {betx_idx = N_col-1;continue;}
                   else if (header=="BETY") {bety_idx = N_col-1;continue;}
              }
           }
           break;
     }
     tabfile.clear();
     tabfile.seekg(0);
}
void PPSOpticsCalibrator::FindIP(std::ifstream& tabfile)
{
    if (!tabfile.good()) {
       std::cout << "Bad input file. Could not find IP position"<<std::endl;
       return;
    }
    double BetaX=0.;
    double BetaY=0.;
    int Ncol=0;
    std::string temp_string;
    std::istringstream curstring;
    while (std::getline(tabfile,temp_string)) {
          if (temp_string.find("IP5")>temp_string.length()) continue;
          curstring.str(temp_string);
          std::string buffer;
          while(curstring>>buffer) {
               if (Ncol==x_idx)    fBeamXatIP=-atof(buffer.c_str())*m_to_mm;
               if (Ncol==y_idx)    fBeamYatIP=atof(buffer.c_str())*m_to_mm;
               if (Ncol==s_idx)    IPposition=atof(buffer.c_str());
               if (Ncol==betx_idx) BetaX=atof(buffer.c_str());
               if (Ncol==bety_idx) BetaY=atof(buffer.c_str());
               if (Ncol>=std::max({s_idx,betx_idx,bety_idx,x_idx,y_idx})) break;
               Ncol++;
          }
          break;
    }
    m_sigmaSX = sqrt(emittanceX*BetaX)*m_to_um;
    m_sigmaSY = sqrt(emittanceY*BetaY)*m_to_um;
    m_sigmaSTX= sqrt(emittanceX/BetaX)*m_to_um;
    m_sigmaSTY= sqrt(emittanceY/BetaY)*m_to_um;

    tabfile.clear();
    tabfile.seekg(0);
}
void PPSOpticsCalibrator::ReadBeamPositionFromOpticsFile(std::ifstream& tabfile)
{
    ReadEmittance(tabfile);
    ReadParameterIndex(tabfile);
    FindIP(tabfile);
    if (!tabfile.good()) {
       std::cout << "Bad input file. Could not find IP position"<<std::endl;
       return;
    }
    std::string temp_string;
    std::istringstream curstring;
    double BetaX=0.;
    double BetaY=0.;
    double X=0.;
    double Y=0.;
    double S=0.;
    while (std::getline(tabfile,temp_string)) {
          int Ncol=0;
          if (temp_string.find("XRPH")>temp_string.length()) continue;
          curstring.clear();
          curstring.str(temp_string);
          std::string buffer;
          while(curstring.good()&&curstring>>buffer) {
               double value=atof(buffer.c_str());
               if (Ncol==s_idx) {S=value-IPposition;}
               if (Ncol==x_idx) {X=value*m_to_mm;}
               if (Ncol==y_idx) {Y=value*m_to_mm;}
               if (Ncol==betx_idx) {BetaX=value;}
               if (Ncol==bety_idx) {BetaY=value;}
               Ncol++;
          }
          double sigx=sqrt(emittanceX*BetaX)*m_to_mm;
          double sigy=sqrt(emittanceY*BetaY)*m_to_mm;
          if (S>0) PosP.push_back(std::make_tuple<double,double,double,double,double>((double)S,(double)X,(double)Y,(double)sigx,(double)sigy));
          else     PosN.insert(PosN.rbegin().base(),std::make_tuple<double,double,double,double,double>((double)-S,(double)X,(double)Y,(double)sigx,(double)sigy));
    }
    std::reverse(PosN.begin(),PosN.end());
    std::cout << "Beam Parameters at RPs (Positive side)"<<std::endl;
    for(int i=0;i<(int)PosP.size();i++) 
       std::cout << "At Z = " << std::get<0>(PosP.at(i)) << " " << std::get<1>(PosP.at(i)) << " "
                 << std::get<2>(PosP.at(i)) << " " << std::get<3>(PosP.at(i)) << " " << std::get<4>(PosP.at(i)) <<std::endl;
    std::cout << "Beam Parameters at RPs (Negative side)"<<std::endl;
    for(int i=0;i<(int)PosN.size();i++) 
       std::cout << "At Z = " << std::get<0>(PosN.at(i)) << " " << std::get<1>(PosN.at(i)) << " "
                 << std::get<2>(PosN.at(i)) << " " << std::get<3>(PosN.at(i)) << " " << std::get<4>(PosN.at(i)) <<std::endl;
    tabfile.clear();
    tabfile.seekg(0);
}
void PPSOpticsCalibrator::AlignObject(H_OpticalElement* opt,H_BeamLine* bline,double init_displacement)
{
     double tolerance=1.;
     for(int i:{'x','y'}) {
         H_BeamParticle h_p;
         h_p.setPosition(-fBeamXatIP*mm_to_um,fBeamYatIP*mm_to_um,0.,0.,0.);
         h_p.computePath(bline);
         h_p.propagate(opt->getS());
         double delta=(i=='x')?abs(opt->getRelX()-h_p.getX()):abs(opt->getRelY()-h_p.getY());
         double delta_min=delta;
         double last_delta=delta;
         double displacement=init_displacement;
         //displacement=delta;
         double TotDisplacement=0.;
         while(delta>tolerance) {
              //double diff = opt->getRelX()-h_p.getX();
              switch (OptElementType(opt->getTypeString())) {
                      case Quadrupole:
                               switch(i) {
                                       case 'x':{ bline->alignElement(opt->getName(),displacement,0.);
                                                 bline->calcMatrix();
                                                 h_p.resetPath();h_p.computePath(bline);h_p.propagate(opt->getS());
                                                 delta=abs(opt->getRelX()-h_p.getX());}
                                                 break;
                                       case 'y': bline->alignElement(opt->getName(),0.,displacement);
                                                 bline->calcMatrix();
                                                 h_p.resetPath();h_p.computePath(bline);h_p.propagate(opt->getS());
                                                 delta=abs(opt->getRelY()-h_p.getY());
                                                 break;
                               }
                               TotDisplacement+=displacement;
                               break;
                               //bline->tiltElement(opt->getName(),displacement,0);TotDisplacement+=displacement;
                      case Dipole:
                      default: return;
               }
               if (delta<delta_min) { delta_min=delta;} // good, go on in this path
               else if (delta>last_delta) {
                    (i=='x')?bline->alignElement(opt->getName(),-displacement,0):bline->alignElement(opt->getName(),0.,-displacement);
                    TotDisplacement-=displacement;
                    displacement/=2; // decrease displacement
                    displacement*=-1;  // change direction
               }
               last_delta=delta;
         }
     }
}
