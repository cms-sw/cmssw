CREATE OR REPLACE
function HVCHANNELLOGICID (DPEName in varchar2) return varchar2 is
	alias varchar2(1000);
	moduleNumber number;
	moduleNumber2 number;
	channelNumber number;
	invalid_channel_name exception; --there probably isn't much point defining my own exception type since it will be out of scope by the time anything sees it and will just show as a user defined exception, but it's better than throwing some unrelated predefined exception
begin
/*
For the HV channels the logic_ids are
 
 1051SS00CC
 
SS = SM number 1-36
CC = channel number 1-34

 source string will be something like CMS_ECAL_HV_SM11/CMS_ECAL_HV_SM11M1/channel04
*/
	alias:=getAliasForDevice(DPEName);
	moduleNumber:=to_number(regexp_substr(alias,'[[:digit:]]+'));
	moduleNumber2:=to_number(regexp_substr(alias,'[[:digit:]]+',1,2)); --this is just to check that the two numbers are the same, as we expect
	channelNumber:=to_number(regexp_substr(alias,'[[:digit:]]+',1,4));
	if moduleNumber!=moduleNumber2 or channelNumber is null or moduleNumber>36 or channelNumber>34 then --actually the module and channel numbers could be up to 99 and 9999 respectively without mucking up the logic id, but if they're out of the expected range then maybe something else has gone wrong, like readng the wrong number
		raise invalid_channel_name;
	end if;
	return 1051000000+10000*moduleNumber+channelNumber;
end;
/
show errors;
