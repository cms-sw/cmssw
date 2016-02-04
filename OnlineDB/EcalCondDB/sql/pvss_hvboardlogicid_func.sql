CREATE OR REPLACE
function HVBOARDLOGICID(DPEName in varchar2) return varchar2 is

alias varchar2(1000);
superModuleNumber number;
boardNumber number;
invalid_board_name exception;

/*
 For the HV boards the logic_ids are

 1061SS00CC

 SS = SM number 1-36
 CC = board number 0, 2, 4 or 6
 source string will be soemthing like ECAL_HV/SM11/board11  
*/

begin
  alias:=getAliasForDevice(DPEName);

  superModuleNumber:=to_number(regexp_substr(alias,'[[:digit:]]+'));
  boardNumber:=to_number(regexp_substr(alias,'[[:digit:]]+',1,2));

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
