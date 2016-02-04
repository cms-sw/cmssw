CREATE OR REPLACE
function HVCHANNELLOGICID (DPEName in varchar2) return varchar2 is

alias varchar2(1000);
superModuleLoc varchar2(100);
superModuleNumber number;
moduleNumber number;
channelNumber number;
invalid_channel_name exception;
currentDate date;

begin
/*
For the HV channels the logic_ids are
 
 1051SS00CC
 
SS = SM number 1-36
CC = channel number 1-34

 source string will by something like 'ECAL_HV/SM_H2/M2/channel17'

 the table 'PVSS_TB_SM_DAT' has a field DP_NAME with strings 
 like 'pcethdcs2:GeneralStands.H4_BEAM.SMNum'
*/
  alias:=getAliasForDevice(DPEName);
  currentDate:=sysdate;

  /* Second field */
  superModuleLoc:=regexp_substr(alias, '[^/]+', 1, 2);

  /* Switch strings */
  if superModuleLoc = 'SM_BEAM' then
     superModuleLoc:='H4_BEAM';
  elsif superModuleLoc = 'SM_COSM' then
     superModuleLoc:='H4_COSM';
  elsif superModuleLoc = 'SM_H2' then
     superModuleLoc:='H2';
  end if;

  /* Get SM Number */
  select to_number(substr(sm, 3)) into superModuleNumber from pvss_tb_sm_dat
   where dp_name like '%'||superModuleLoc||'.SMNum'
     and since <= currentDate and till > currentDate;

  /* Fourth field, all the chars after 'channel' */
  channelNumber:=to_number(substr(regexp_substr(alias, '[^/]+', 1, 4), 8));

  if superModuleNumber is null 
  or channelNumber is null 
  or superModuleNumber>36 
  or channelNumber>34 then
    raise invalid_channel_name;
  end if;

  return 1051000000+10000*superModuleNumber+channelNumber;
end;
/
show errors;
