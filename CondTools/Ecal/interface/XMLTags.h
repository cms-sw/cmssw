#ifndef _CondTools_Ecal_XMLTags_
#define _CondTools_Ecal_XMLTags_

#include <string>

namespace xuti{

  const  std::string iEta_tag("iEta");
  const  std::string iPhi_tag("iPhi");
  const  std::string ix_tag("ix");
  const  std::string iy_tag("iy");
  const  std::string zside_tag("zside");
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
 
  const  std::string IntercalibConstants_tag("EcalIntercalibConstants");
  const  std::string IntercalibConstant_tag("IntercalibConstant");
  const  std::string IntercalibError_tag("IntercalibError");

  const  std::string GainRatios_tag("EcalGainRatios");
  const std::string Gain6Over1_tag("Gain6Over1");
  const std::string Gain12Over6_tag("Gain12Over6");

   
  const std::string ChannelStatus_tag("EcalChannelStatus");
  const std::string ChannelStatusCode_tag("ChannelStatusCode");


  const  std::string ADCToGeVConstant_tag("EcalADCToGeVConstant");
  const  std::string Barrel_tag("BarrelValue");
  const  std::string Endcap_tag("EndcapValue");





}

#endif
