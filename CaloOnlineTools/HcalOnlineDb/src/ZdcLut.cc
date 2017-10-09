#include "CaloOnlineTools/HcalOnlineDb/interface/ZdcLut.h"
#include <stdlib.h>

/**

   \class ZdcLut
   \brief Generation of ZDC Lookup tables and associate helper methods
   \brief Adopted to CMSSW HCAL LUT manager specs
   \brief by Gena Kukartsev, Brown University, Dec 08, 2009
   \author Elijah Dunn
*/

ZdcLut::ZdcLut()
{
//constants for ADC to GeV conversion
    double ADC_GEV_EM = 13.55, ADC_GEV_HAD = 0.6;
    int LSB_EM = 1, LSB_HAD = 5;
    std::vector <int> fC_TDR;
//fills in fC_TDR with values from table
    fC_TDR.push_back(0);
    fC_TDR.push_back(1);
    fC_TDR.push_back(2);
    fC_TDR.push_back(3);
    fC_TDR.push_back(4);
    fC_TDR.push_back(5);
    fC_TDR.push_back(6);
    fC_TDR.push_back(7);
    fC_TDR.push_back(8);
    fC_TDR.push_back(9);
    fC_TDR.push_back(10);
    fC_TDR.push_back(11);
    fC_TDR.push_back(12);
    fC_TDR.push_back(13);
    fC_TDR.push_back(14);
    fC_TDR.push_back(16);
    fC_TDR.push_back(18);
    fC_TDR.push_back(20);
    fC_TDR.push_back(22);
    fC_TDR.push_back(24);
    fC_TDR.push_back(26);
    fC_TDR.push_back(28);
    fC_TDR.push_back(31);
    fC_TDR.push_back(34);
    fC_TDR.push_back(37);
    fC_TDR.push_back(40);
    fC_TDR.push_back(44);
    fC_TDR.push_back(48);
    fC_TDR.push_back(52);
    fC_TDR.push_back(57);
    fC_TDR.push_back(62);
    fC_TDR.push_back(67);
    fC_TDR.push_back(62);
    fC_TDR.push_back(67);
    fC_TDR.push_back(72);
    fC_TDR.push_back(77);
    fC_TDR.push_back(82);
    fC_TDR.push_back(87);
    fC_TDR.push_back(92);
    fC_TDR.push_back(97);
    fC_TDR.push_back(102);
    fC_TDR.push_back(107);
    fC_TDR.push_back(112);
    fC_TDR.push_back(117);
    fC_TDR.push_back(122);
    fC_TDR.push_back(127);
    fC_TDR.push_back(132);
    fC_TDR.push_back(142);
    fC_TDR.push_back(152);
    fC_TDR.push_back(162);
    fC_TDR.push_back(172);
    fC_TDR.push_back(182);
    fC_TDR.push_back(192);
    fC_TDR.push_back(202);
    fC_TDR.push_back(217);
    fC_TDR.push_back(232);
    fC_TDR.push_back(247);
    fC_TDR.push_back(262);
    fC_TDR.push_back(282);
    fC_TDR.push_back(302);
    fC_TDR.push_back(322);
    fC_TDR.push_back(347);
    fC_TDR.push_back(372);
    fC_TDR.push_back(397);
    fC_TDR.push_back(372);
    fC_TDR.push_back(397);
    fC_TDR.push_back(422);
    fC_TDR.push_back(447);
    fC_TDR.push_back(472);
    fC_TDR.push_back(497);
    fC_TDR.push_back(522);
    fC_TDR.push_back(547);
    fC_TDR.push_back(572);
    fC_TDR.push_back(597);
    fC_TDR.push_back(622);
    fC_TDR.push_back(647);
    fC_TDR.push_back(672);
    fC_TDR.push_back(697);
    fC_TDR.push_back(722);
    fC_TDR.push_back(772);
    fC_TDR.push_back(822);
    fC_TDR.push_back(872);
    fC_TDR.push_back(922);
    fC_TDR.push_back(972);
    fC_TDR.push_back(1022);
    fC_TDR.push_back(1072);
    fC_TDR.push_back(1147);
    fC_TDR.push_back(1222);
    fC_TDR.push_back(1297);
    fC_TDR.push_back(1372);
    fC_TDR.push_back(1472);
    fC_TDR.push_back(1572);
    fC_TDR.push_back(1672);
    fC_TDR.push_back(1797);
    fC_TDR.push_back(1922);
    fC_TDR.push_back(2047);
    fC_TDR.push_back(1922);
    fC_TDR.push_back(2047);
    fC_TDR.push_back(2172);
    fC_TDR.push_back(2297);
    fC_TDR.push_back(2422);
    fC_TDR.push_back(2547);
    fC_TDR.push_back(2672);
    fC_TDR.push_back(1797);
    fC_TDR.push_back(2922);
    fC_TDR.push_back(3047);
    fC_TDR.push_back(3172);
    fC_TDR.push_back(3297);
    fC_TDR.push_back(3422);
    fC_TDR.push_back(2547);
    fC_TDR.push_back(3672);
    fC_TDR.push_back(3922);
    fC_TDR.push_back(4172);
    fC_TDR.push_back(4422);
    fC_TDR.push_back(4672);
    fC_TDR.push_back(4922);
    fC_TDR.push_back(5172);
    fC_TDR.push_back(5422);
    fC_TDR.push_back(5797);
    fC_TDR.push_back(6172);
    fC_TDR.push_back(6547);
    fC_TDR.push_back(6922);
    fC_TDR.push_back(7422);
    fC_TDR.push_back(7922);
    fC_TDR.push_back(8422);
    fC_TDR.push_back(9047);
    fC_TDR.push_back(9672);
    fC_TDR.push_back(10297);

// two vectors containing the LUT; one for Hadronic and one for Electromagnetic
    std::vector <int> HADlut(128);
    std::vector <int> EMlut(128);
//uses the constants to fill each LUT
    for(int zdci = 0; zdci < 128; zdci++)
      { EMlut[zdci] = (int)((fC_TDR[zdci]/ADC_GEV_EM)/LSB_EM + 0.5);}
    for(int zdci = 0; zdci < 128; zdci++)
      { HADlut[zdci] = (int)((fC_TDR[zdci]/ADC_GEV_HAD)/LSB_HAD + 0.5);}

    side.resize(2);
    side[0].fiber.resize(3);
    side[1].fiber.resize(3);
    side[0].fiber[0].channel.resize(3);
    side[0].fiber[1].channel.resize(3);
    side[0].fiber[2].channel.resize(3);
    side[1].fiber[0].channel.resize(3);
    side[1].fiber[1].channel.resize(3);
    side[1].fiber[2].channel.resize(3);
    side[0].fiber[0].channel[0].LUT = EMlut;
    side[0].fiber[0].channel[1].LUT = EMlut;
    side[0].fiber[0].channel[2].LUT = EMlut;
    side[0].fiber[1].channel[0].LUT = EMlut;
    side[0].fiber[1].channel[1].LUT = EMlut;
    side[0].fiber[1].channel[2].LUT = HADlut;
    side[0].fiber[2].channel[0].LUT = HADlut;
    side[0].fiber[2].channel[1].LUT = HADlut;
    side[0].fiber[2].channel[2].LUT = HADlut;
    side[1].fiber[0].channel[0].LUT = EMlut;
    side[1].fiber[0].channel[1].LUT = EMlut;
    side[1].fiber[0].channel[2].LUT = EMlut;
    side[1].fiber[1].channel[0].LUT = EMlut;
    side[1].fiber[1].channel[1].LUT = EMlut;
    side[1].fiber[1].channel[2].LUT = HADlut;
    side[1].fiber[2].channel[0].LUT = HADlut;
    side[1].fiber[2].channel[1].LUT = HADlut;
    side[1].fiber[2].channel[2].LUT = HADlut;
}


ZdcLut::~ZdcLut( void )
{    
}

int ZdcLut::simple_loop()
{
  for(unsigned int zdcs = 0; zdcs < side.size(); zdcs++)
    {
      for (unsigned int zdcf = 0; zdcf < side[zdcs].fiber.size(); zdcf++)
        {
	  for (unsigned int zdcc = 0; zdcc < side[zdcs].fiber[zdcf].channel.size(); zdcc++)
            {
	      for (unsigned int zdcl = 0; zdcl < side[zdcs].fiber[zdcf].channel[zdcc].LUT.size(); zdcl++)
		{ std::cout << side[zdcs].fiber[zdcf].channel[zdcc].LUT[zdcl] << " "; }
	      std::cout << std::endl;
            }
            std::cout << std::endl;
        }
      std::cout << std::endl;
    }
  return 0;
}


std::vector <int> ZdcLut::get_lut(int emap_side,
			          int emap_htr_fiber,
			          int emap_fi_ch){
  int side_num = (1-emap_side)/2;
  int fiber_num = (int)(emap_htr_fiber/4)+(emap_htr_fiber%4);
  int channel_num = emap_fi_ch;
  return side[side_num].fiber[fiber_num].channel[channel_num].LUT;
}


// get LUT by proper ZDC channel
std::vector <int> ZdcLut::get_lut(std::string zdc_section,
			          int zdc_side,
			          int zdc_channel){
  int side_num = (1-zdc_side)/2;
  int fiber_num = -1;
  int channel_num = -1;
  if (zdc_section.find("ZDC EM")!=std::string::npos){
    fiber_num = (int)(zdc_channel/4);
    channel_num = (int)((zdc_channel-1)/3) % 3;
  }
  else if (zdc_section.find("ZDC HAD")!=std::string::npos){
    if (zdc_channel==1){
      fiber_num = 1;
      channel_num = 2;
    }
    else if (zdc_channel==2){
      fiber_num = 2;
      channel_num = 0;
    }
    else if (zdc_channel==3){
      fiber_num = 2;
      channel_num = 1;
    }
    else if (zdc_channel==4){
      fiber_num = 2;
      channel_num = 2;
    }
    else{
      std::cout << zdc_channel << ": unknown ZDC channel, exiting..." << std::endl;
      exit(-1);
    }
  }
  else{
    std::cout << zdc_section << ": unknown ZDC section, exiting..." << std::endl;
    exit(-1);
  }
  // FIXME: add validity check here
  if (1==1){
    return side[side_num].fiber[fiber_num].channel[channel_num].LUT;
  }
  else return side[side_num].fiber[fiber_num].channel[channel_num].LUT;
}
