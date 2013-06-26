CREATE OR REPLACE
function HVCHANNELLOGICID (DPEName in varchar2) return varchar2 is

alias varchar2(1000);
superModuleNumber number;
moduleNumber number;
channelNumber number;
invalid_channel_name exception;

begin
/*
For the HV channels the logic_ids are
 
 1051SS00CC
 
SS = SM number 1-36
CC = channel number 1-34

 source string will by something like ECAL_HV/SM11/M1/channel04
*/
  alias:=getAliasForDevice(DPEName);

  superModuleNumber:=to_number(regexp_substr(alias,'[[:digit:]]+'));
  moduleNumber:=to_number(regexp_substr(alias,'[[:digit:]]+',1,2));
  channelNumber:=to_number(regexp_substr(alias,'[[:digit:]]+',1,3));

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
