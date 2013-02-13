#ifndef _CondTools_Ecal_XMLTags_
#define _CondTools_Ecal_XMLTags_

#include <string>

namespace xuti{

  const  std::string iEta_tag("iEta");
  const  std::string iPhi_tag("iPhi");
  const  std::string ix_tag("ix");
  const  std::string iy_tag("iy");
  const  std::string zside_tag("zside");
  const  std::string ixSC_tag("ixSC");
  const  std::string iySC_tag("iySC");
  const  std::string Cell_tag("cell");
  
  const  std::string Header_tag("EcalCondHeader");
  const  std::string Header_methodtag("method");
  const  std::string Header_versiontag("version");
  const  std::string Header_datasourcetag("datasource");
  const  std::string Header_sincetag("since");
  const  std::string Header_tagtag("tag");
  const  std::string Header_datetag("date"); 

  const  std::string WeightGroups_tag("EcalWeightGroups");
  const  std::string WeightGroup_tag("EcalWeightGroup");
  const  std::string EcalTDCId_tag("EcalTDCId");

 
  const  std::string row_tag("row");
  const  std::string id_tag("id");

  const  std::string EcalWeightSet_tag("EcalWeightSet");
  const  std::string wgtBeforeSwitch_tag("WeightBeforeSwitch");
  const  std::string wgtAfterSwitch_tag("WeightAfterSwitch");
  const  std::string wgtChi2BeforeSwitch_tag("Chi2WeightBeforeSwitch");
  const  std::string wgtChi2AfterSwitch_tag("Chi2WeightAfterSwitch");

  const  std::string EcalTBWeights_tag("EcalTBWeights");
  const  std::string EcalTBWeight_tag("EcalTBWeight");
  const  std::string EcalXtalGroupId_tag("EcalXtalGroupId");
 
  const  std::string LinearCorrections_tag("EcalLinearCorrections");
  const  std::string IntercalibConstants_tag("EcalIntercalibConstants");
  const  std::string IntercalibConstant_tag("IntercalibConstant");
  const  std::string IntercalibError_tag("IntercalibError");

  const  std::string GainRatios_tag("EcalGainRatios");
  const std::string Gain6Over1_tag("Gain6Over1");
  const std::string Gain12Over6_tag("Gain12Over6");

  const std::string Pedestals_tag("EcalPedestals");
  const std::string mean12_tag("mean_x12");
  const std::string mean6_tag("mean_x6");
  const std::string mean1_tag("mean_x1");
  const std::string rms12_tag("rms_x12");
  const std::string rms6_tag("rms_x6");
  const std::string rms1_tag("rms_x1");
   
  const std::string ChannelStatus_tag("EcalChannelStatus");
  const std::string ChannelStatusCode_tag("ChannelStatusCode");

  const std::string DAQTowerStatus_tag("EcalDAQTowerStatus");
  const std::string DAQStatusCode_tag("DAQStatusCode");

  const std::string DCSTowerStatus_tag("EcalDCSTowerStatus");
  const std::string DCSStatusCode_tag("DCSStatusCode");

  const std::string TPGTowerStatus_tag("EcalTPGTowerStatus");
  const std::string TPGCrystalStatus_tag("EcalTPGCrystalStatus");
  const std::string TPGStripStatus_tag("EcalTPGStripStatus");

  const  std::string ADCToGeVConstant_tag("EcalADCToGeVConstant");
  const  std::string Barrel_tag("BarrelValue");
  const  std::string Endcap_tag("EndcapValue");


  const  std::string Value_tag("Value");
  const  std::string EcalFloatCondObjectContainer_tag("EcalFloatCondObjectContainer");

  const std::string Laser_tag("EcalLaserAPDPNRatios");
  const std::string Laser_p1_tag("p1");
  const std::string Laser_p2_tag("p2");
  const std::string Laser_p3_tag("p3");
  const std::string Laser_t1_tag("t1");
  const std::string Laser_t2_tag("t2");
  const std::string Laser_t3_tag("t3");

  const std::string Linearization_tag("EcalTPGLinearizationConts");
  const std::string Linearization_m12_tag("mult12");
  const std::string Linearization_m6_tag("mult6");
  const std::string Linearization_m1_tag("mult1");
  const std::string Linearization_s12_tag("shift12");
  const std::string Linearization_s6_tag("shift6");
  const std::string Linearization_s1_tag("shift1");

  const  std::string AlignmentConstant_tag("EcalAlignmentConstant");
  const  std::string subdet_tag("SubDet");
  const  std::string x_tag("x");
  const  std::string y_tag("y");
  const  std::string z_tag("z");
  const  std::string Phi_tag("Phi");
  const  std::string Theta_tag("Theta");
  const  std::string Psi_tag("Psi");

  const  std::string TimeOffsetConstant_tag("EcalTimeOffsetConstant");

}

#endif
