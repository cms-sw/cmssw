#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsHelper.h"

void L1TMuonBarrelParamsHelper::print(std::ostream& out) const {

  out << "L1 BMTF Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
}

L1TMuonBarrelParamsHelper::L1TMuonBarrelParamsHelper(const L1TMuonBarrelParams& barrelParams) : L1TMuonBarrelParams(barrelParams) //: L1TMuonBarrelParams_PUBLIC(cast_to_L1TMuonBarrelParams_PUBLIC(barrelParams)) //: m_params_helper(barrelParams)
{
//	if (pnodes_.size() != 2) 
//	    pnodes_.resize(2);
  
}

void L1TMuonBarrelParamsHelper::configFromPy(std::map<std::string, int>& allInts, std::map<std::string, bool>& allBools, std::map<std::string, std::vector<std::string> > allMasks, unsigned int fwVersion, const std::string& AssLUTpath)
{
	set_PT_Assignment_nbits_Phi(allInts["PT_Assignment_nbits_Phi"]);
	set_PT_Assignment_nbits_PhiB(allInts["PT_Assignment_nbits_PhiB"]);
	set_PHI_Assignment_nbits_Phi(allInts["PHI_Assignment_nbits_Phi"]);
	set_PHI_Assignment_nbits_PhiB(allInts["PHI_Assignment_nbits_PhiB"]);
	set_Extrapolation_nbits_Phi(allInts["Extrapolation_nbits_Phi"]);
	set_Extrapolation_nbits_PhiB(allInts["Extrapolation_nbits_PhiB"]);
	set_BX_min(allInts["BX_min"]);
	set_BX_max(allInts["BX_max"]);
	set_Extrapolation_Filter(allInts["Extrapolation_Filter"]);
	set_OutOfTime_Filter_Window(allInts["OutOfTime_Filter_Window"]);
	set_OutOfTime_Filter(allBools["OutOfTime_Filter"]);
	set_Open_LUTs(allBools["Open_LUTs"]);
	set_EtaTrackFinder(allBools["EtaTrackFinder"]);
	set_Extrapolation_21(allBools["Extrapolation_21"]);
	setFwVersion(fwVersion);
	set_DisableNewAlgo(allBools["DisableNewAlgo"]);

	setAssLUTPath(AssLUTpath);
	///Read Pt assignment Luts
        std::vector<LUT> pta_lut(0); pta_lut.reserve(19);
	std::vector<int> pta_threshold(10);
	if ( load_pt(pta_lut,pta_threshold, allInts["PT_Assignment_nbits_Phi"], AssLUTpath) != 0 ) {
	  cout << "Can not open files to load pt-assignment look-up tables for L1TMuonBarrelTrackProducer!" << endl;
	}
	setpta_lut(pta_lut);
	setpta_threshold(pta_threshold);

	///Read Phi assignment Luts
	std::vector<LUT> phi_lut(0); phi_lut.reserve(2);
	if ( load_phi(phi_lut, allInts["PHI_Assignment_nbits_Phi"], allInts["PHI_Assignment_nbits_PhiB"], AssLUTpath) != 0 ) {
	  cout << "Can not open files to load phi-assignment look-up tables for L1TMuonBarrelTrackProducer!" << endl;
	}
	setphi_lut(phi_lut);




	l1mudttfparams.reset();  //KK
	l1mudttfqualplut.load(); //KK: Do these LUTs ever change and is it safe to initialize it from the release files like that? 
	l1mudttfetaplut.load();  //KK
        // the data members of the Helper class loaded above are transient, push those to the persistent storage of the base class:
        lutparams_.eta_lut_ = l1mudttfetaplut.m_lut;
        lutparams_.qp_lut_  = l1mudttfqualplut.m_lut;


	for( int wh=-3; wh<4; wh++ ) {
	   int sec = 0;
	   for(char& c : allMasks["mask_phtf_st1"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_inrec_chdis_st1(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_phtf_st2"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_inrec_chdis_st2(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_phtf_st3"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_inrec_chdis_st3(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_phtf_st4"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_inrec_chdis_st4(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_ettf_st1"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_etsoc_chdis_st1(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_ettf_st2"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_etsoc_chdis_st2(wh,sec,mask);
	        sec++;
	    }
	   sec = 0;
	   for(char& c : allMasks["mask_ettf_st3"].at(wh+3) ) {
	        int mask = c - '0';
	        l1mudttfmasks.set_etsoc_chdis_st3(wh,sec,mask);
	        //Not used in BMTF - mask
	        l1mudttfmasks.set_inrec_chdis_csc(wh,sec,true);
	        sec++;
	    }

	}


	///Read Extrapolation Luts
	std::vector<L1TMuonBarrelParams::LUTParams::extLUT> ext_lut(0); ext_lut.reserve(12);
	if ( load_ext(ext_lut, allInts["PHI_Assignment_nbits_Phi"], allInts["PHI_Assignment_nbits_PhiB"]) != 0 ) {
	  cout << "Can not open files to load extrapolation look-up tables for L1TMuonBarrelTrackProducer!" << endl;
	}
	setext_lut(ext_lut);

	//l1mudttfextlut.load();
}

void L1TMuonBarrelParamsHelper::configFromDB(l1t::TriggerSystem& trgSys)
{
	std::map<std::string, std::string> procRole = trgSys.getProcToRoleAssignment();

	//Cleaning the default masking from the prototype
	l1mudttfmasks.reset();

	for(auto it_proc=procRole.begin(); it_proc!=procRole.end(); it_proc++ )
	{

	  std::string procId = it_proc->first;

	  std::map<std::string, l1t::Parameter> settings = trgSys.getParameters(procId.c_str());


          std::vector<std::string> paths = settings["regTable"].getTableColumn<std::string>("register_path");
          std::vector<unsigned int> vals = settings["regTable"].getTableColumn<unsigned int>("register_value");
          for(unsigned int row=0; row<paths.size(); row++)
	  {
	    if (paths[row].find("open_lut") != std::string::npos){
	      //std::cout << "Value is: " << vals[row] << std::endl;
	      set_Open_LUTs(vals[row]);
	    }
	    if (paths[row].find("sel_21") != std::string::npos){
	      //std::cout << "Value is: " << vals[row] << std::endl;
	      set_Extrapolation_21(vals[row]);
	    }

	    if (paths[row].find("dis_newalgo") != std::string::npos){
	      //std::cout << "Value is: " << vals[row] << std::endl;
	      //int fwv = (vals[row]==1) ? 1 : 2;
	      //setFwVersion(fwv);
	      bool disnewalgo = (vals[row]==1);
	      set_DisableNewAlgo(disnewalgo);
	    }

	    string masks[5] = {"mask_ctrl_N2", "mask_ctrl_N1", "mask_ctrl_0", "mask_ctrl_P1", "mask_ctrl_P2"};

	    for(int m=0; m<5; m++)
	    {

	        if (paths[row].find(masks[m]) != std::string::npos){
	          ///Converts the last bit to int
	          int mask = 0x1&vals[row];
	          int mask_all = vals[row];
	          ///All bits must be the same
	          if(!( mask_all==0x111111 || mask_all==0x222222 || mask_all==0x333333 || mask_all==0x444444 ||
	             mask_all==0x555555 || mask_all==0x666666 || mask_all==0x777777) )
	            cerr<<"BMTF: Cannot re-emulate properly. Individual link masking cannot be handled."<<endl;

	          if((mask&1)>0)  {
	             for(int sec=0; sec<12; sec++){
	              if(masks[m]=="mask_ctrl_N2"){
	                                    l1mudttfmasks.set_inrec_chdis_st1(-3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(-3,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_N1"){
	                                    l1mudttfmasks.set_inrec_chdis_st1(-2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(-2,sec,true);
	              }

	              if(masks[m]=="mask_ctrl_0"){
	                                    l1mudttfmasks.set_inrec_chdis_st1(-1,sec,true);
	                                    l1mudttfmasks.set_inrec_chdis_st1(1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(-1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(1,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P1"){
	                                    l1mudttfmasks.set_inrec_chdis_st1(2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(2,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P2"){
	                                    l1mudttfmasks.set_inrec_chdis_st1(3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st1(3,sec,true);
	              }
	            }

	          }

	          if((mask&2)>0)  {
	            for(int sec=0; sec<12; sec++){
	              if(masks[m]=="mask_ctrl_N2"){
	                                    l1mudttfmasks.set_inrec_chdis_st2(-3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(-3,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_N1"){
	                                    l1mudttfmasks.set_inrec_chdis_st2(-2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(-2,sec,true);
	              }

	              if(masks[m]=="mask_ctrl_0"){
	                                    l1mudttfmasks.set_inrec_chdis_st2(-1,sec,true);
	                                    l1mudttfmasks.set_inrec_chdis_st2(1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(-1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(1,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P1"){
	                                    l1mudttfmasks.set_inrec_chdis_st2(2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(2,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P2"){
	                                    l1mudttfmasks.set_inrec_chdis_st2(3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st2(3,sec,true);
	              }
	            }
	          }

	          if((mask&4)>0)  {
	            for(int sec=0; sec<12; sec++){
	              if(masks[m]=="mask_ctrl_N2"){
	                                    l1mudttfmasks.set_inrec_chdis_st3(-3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(-3,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_N1"){
	                                    l1mudttfmasks.set_inrec_chdis_st3(-2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(-2,sec,true);
	              }

	              if(masks[m]=="mask_ctrl_0"){
	                                    l1mudttfmasks.set_inrec_chdis_st3(-1,sec,true);
	                                    l1mudttfmasks.set_inrec_chdis_st3(1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(-1,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(1,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P1"){
	                                    l1mudttfmasks.set_inrec_chdis_st3(2,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(2,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P2"){
	                                    l1mudttfmasks.set_inrec_chdis_st3(3,sec,true);
	                                    //l1mudttfmasks.set_etsoc_chdis_st3(3,sec,true);
	              }
	            }
	          }

	          if((mask&8)>0)  {
	            for(int sec=0; sec<12; sec++){
	              if(masks[m]=="mask_ctrl_N2"){
	                                    l1mudttfmasks.set_inrec_chdis_st4(-3,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_N1"){
	                                    l1mudttfmasks.set_inrec_chdis_st4(-2,sec,true);
	              }

	              if(masks[m]=="mask_ctrl_0"){
	                                    l1mudttfmasks.set_inrec_chdis_st4(-1,sec,true);
	                                    l1mudttfmasks.set_inrec_chdis_st4(1,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P1"){
	                                    l1mudttfmasks.set_inrec_chdis_st4(2,sec,true);
	              }
	              if(masks[m]=="mask_ctrl_P2"){
	                                    l1mudttfmasks.set_inrec_chdis_st4(3,sec,true);
	              }
	            }
	          }
	       }///if register path
	     }///for masks
	  }///for it tRow
	}///for it procRole
}///if configDB

int L1TMuonBarrelParamsHelper::load_pt(std::vector<LUT>& pta_lut,
                                  std::vector<int>& pta_threshold,
                                  unsigned short int nbitphi,
                                  std::string AssLUTpath
                                  ){


// maximal number of pt assignment methods
const int MAX_PTASSMETH = 19;
const int MAX_PTASSMETHA = 12;

// pt assignment methods
enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                   PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H,
                   PB12H,  PB13H,  PB14H,  PB21H,  PB23H,  PB24H, PB34H,
                   NODEF };

  // get directory name
  string pta_str = "";
  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (address) and  pt ...5 bits
  // now convert phi and phib to the required precision
  int nbit_phi = nbitphi;
  int sh_phi  = 12 - nbit_phi;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < MAX_PTASSMETH; pam++ ) {
    switch ( pam ) {
      case PT12L  : { pta_str = "pta12l"; break; }
      case PT12H  : { pta_str = "pta12h"; break; }
      case PT13L  : { pta_str = "pta13l"; break; }
      case PT13H  : { pta_str = "pta13h"; break; }
      case PT14L  : { pta_str = "pta14l"; break; }
      case PT14H  : { pta_str = "pta14h"; break; }
      case PT23L  : { pta_str = "pta23l"; break; }
      case PT23H  : { pta_str = "pta23h"; break; }
      case PT24L  : { pta_str = "pta24l"; break; }
      case PT24H  : { pta_str = "pta24h"; break; }
      case PT34L  : { pta_str = "pta34l"; break; }
      case PT34H  : { pta_str = "pta34h"; break; }
      case PB12H  : { pta_str = "ptb12h_Feb2016"; break; }
      case PB13H  : { pta_str = "ptb13h_Feb2016"; break; }
      case PB14H  : { pta_str = "ptb14h_Feb2016"; break; }
      case PB21H  : { pta_str = "ptb21h_Feb2016"; break; }
      case PB23H  : { pta_str = "ptb23h_Feb2016"; break; }
      case PB24H  : { pta_str = "ptb24h_Feb2016"; break; }
      case PB34H  : { pta_str = "ptb34h_Feb2016"; break; }

    }

    // assemble file name
    const string& lutpath = AssLUTpath;
    edm::FileInPath lut_f = edm::FileInPath(string(lutpath + pta_str + ".lut"));
    string pta_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(pta_file);
    if ( file.open() != 0 ) return -1;

    // get the right shift factor
    int shift = sh_phi;
    int adr_old = -2048 >> shift;
    if (pam >= MAX_PTASSMETHA) adr_old = -512 >> shift;

    LUT tmplut;

    int number = -1;
    int sum_pt = 0;

    if ( file.good() ) {
      int threshold = file.readInteger();
      pta_threshold[pam/2] = threshold;
    }

    // read values and shift to correct precision
    while ( file.good() ) {

      int adr = (file.readInteger()) >> shift;
      int pt  = file.readInteger();

      number++;
      //cout<<pam<<"    "<<number<<"   "<<MAX_PTASSMETHA<<endl;
      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, (sum_pt/number) ));

        adr_old = adr;
        number = 0;
        sum_pt = 0;
      }

      sum_pt += pt;

      if ( !file.good() ) file.close();

    }

    file.close();
    pta_lut.push_back(tmplut);
  }
  return 0;

}




int L1TMuonBarrelParamsHelper::load_phi(std::vector<LUT>& phi_lut,
                                  unsigned short int nbit_phi,
                                  unsigned short int nbit_phib,
                                  std::string AssLUTpath
                                  ) {


  // precision : in the look-up tables the following precision is used :
  // address (phib) ...10 bits, phi ... 12 bits

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  string phi_str;
  // loop over all phi-assignment methods
  for ( int idx = 0; idx < 2; idx++ ) {
    switch ( idx ) {
      case 0 : { phi_str = "phi12"; break; }
      case 1 : { phi_str = "phi42"; break; }
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(AssLUTpath + phi_str + ".lut"));
    string phi_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(phi_file);
    if ( file.open() != 0 ) return -1;

    LUT tmplut;

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_phi = 0;

    // read values
    while ( file.good() ) {

      int adr = (file.readInteger()) >> sh_phib;
      int phi =  file.readInteger();

      number++;

      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, ((sum_phi/number) >> sh_phi) ));

        adr_old = adr;
        number = 0;
        sum_phi  = 0;
      }

      sum_phi += phi;

      if ( !file.good() ) file.close();

    }

    file.close();
    phi_lut.push_back(tmplut);
  }
  return 0;

}

/*
int L1TMuonBarrelParamsESProducer::getPtLutThreshold(int pta_ind, std::vector<int>& pta_threshold) const {

  if ( pta_ind >= 0 && pta_ind < 13/2 ) {
    return pta_threshold[pta_ind];
  }
  else {
    cerr << "PtaLut::getPtLutThreshold : can not find threshold " << pta_ind << endl;
    return 0;
  }

}

*/




//
// load extrapolation look-up tables
//
int L1TMuonBarrelParamsHelper::load_ext(std::vector<L1TMuonBarrelParams::LUTParams::extLUT>& ext_lut,
                                            unsigned short int nbit_phi,
                                            unsigned short int nbit_phib) {

  //max. number of Extrapolations
  const int MAX_EXT = 12;

  // extrapolation types
  enum Extrapolation { EX12, EX13, EX14, EX21, EX23, EX24, EX34,
                     EX15, EX16, EX25, EX26, EX56 };

  // get directory name
  string defaultPath = "L1Trigger/L1TMuon/data/bmtf_luts/";
  string ext_dir = "LUTs_Ext/";
  string ext_str = "";

  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (low, high), phib ...10 bits (address)
  // now convert phi and phib to the required precision

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  // loop over all extrapolations
  for ( int ext = 0; ext < MAX_EXT; ext++ ) {
    switch (ext) {
      case EX12 : ext_str = "ext12"; break;
      case EX13 : ext_str = "ext13"; break;
      case EX14 : ext_str = "ext14"; break;
      case EX21 : ext_str = "ext21"; break;
      case EX23 : ext_str = "ext23"; break;
      case EX24 : ext_str = "ext24"; break;
      case EX34 : ext_str = "ext34"; break;
      case EX15 : ext_str = "ext15"; break;
      case EX16 : ext_str = "ext16"; break;
      case EX25 : ext_str = "ext25"; break;
      case EX26 : ext_str = "ext26"; break;
      case EX56 : ext_str = "ext56"; break;
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + ext_dir + ext_str + ".lut"));
    string ext_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(ext_file);
    if ( file.open() != 0 ) return -1;
    //    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : "
    //                                         << file.getName() << endl;

    L1TMuonBarrelParams::LUTParams::extLUT tmplut;

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_low = 0;
    int sum_high = 0;

    // read values and shift to correct precision
    while ( file.good() ) {

      int adr  = ( file.readInteger() ) >> sh_phib;	// address (phib)
      int low  = ( file.readInteger() );    		// low value (phi)
      int high = ( file.readInteger() );	        // high value (phi)

      number++;

      if ( adr != adr_old ) {

        tmplut.low[adr_old]  = sum_low  >> sh_phi;
        tmplut.high[adr_old] = sum_high >> sh_phi;

        adr_old = adr;
        number = 0;
        sum_low  = 0;
        sum_high = 0;

      }

      if (number == 0) sum_low  = low;
      if (number == 0) sum_high = high;

      if ( !file.good() ) file.close();
    }

    file.close();
    ext_lut.push_back(tmplut);
  }
  return 0;

}
