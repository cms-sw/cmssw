CREATE OR REPLACE
function HVCHANNELLOGICID (DPName in varchar2) return varchar2 is
/*
For the HV channels the logic_ids are
 
 1051SS00CC
 
SS = SM number 1-36
CC = channel number 1-34

 source string will be something like CAEN/supermodule001/board11/channel007 for a channel
*/
	DPNameWithoutSystem varchar2(1000);
	moduleNumber number;
	boardNumber number;
	channelNumber number;
	invalid_channel_name exception; --there probably isn't much point defining my own exception type since it will be out of scope by the time anything sees it and will just show as a user defined exception, but it's better than throwing some unrelated predefined exception
begin
	DPNameWithoutSystem:=substr(DPName,instr(DPName,':')+1);
	moduleNumber:=to_number(regexp_substr(DPNameWithoutSystem,'[[:digit:]]+'));
	boardNumber:=to_number(regexp_substr(DPNameWithoutSystem,'[[:digit:]]+',1,2));
	channelNumber:=to_number(regexp_substr(DPNameWithoutSystem,'[[:digit:]]+',1,3));
	if moduleNumber is null or boardNumber is null or channelNumber is null then
		raise invalid_channel_name;
	end if;
	return 1051000000+10000*moduleNumber+channelNumber;
end HVCHANNELLOGICID;
/
