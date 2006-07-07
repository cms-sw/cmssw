CREATE OR REPLACE
function HVBOARDLOGICID(DPEName in varchar2) return varchar2 is

alias varchar2(1000);
superModuleLoc varchar2(100);
superModuleNumber number;
moduleNumber number;
boardNumber number;
invalid_board_name exception;
currentDate date;

/*
 For the HV boards the logic_ids are

 1061SS00CC

 SS = SM number 1-36
 CC = board number 0, 2, 4 or 6

 source string will by something like 'ECAL_HV/SM_H2/M2/board17'

 the table 'PVSS_TB_SM_DAT' has a field DP_NAME with strings 
 like 'pcethdcs2:GeneralStands.H4_BEAM.SMNum'

*/

begin
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

  /* Third field, all the chars after 'board' */
  boardNumber:=to_number(substr(regexp_substr(alias, '[^/]+', 1, 3), 6));

  if superModuleNumber is null 
  or boardNumber is null
  or superModuleNumber>36 
  or boardNumber>6 then
    raise invalid_board_name;
  end if;

  return 1061000000+10000*superModuleNumber+boardNumber;
end;
/
show errors;
