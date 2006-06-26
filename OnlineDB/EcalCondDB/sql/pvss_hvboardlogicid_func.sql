CREATE OR REPLACE
function HVBOARDLOGICID(DPName in varchar2) return varchar2 is
/*
 For the HV boards the logic_ids are

 1061SS00CC

 SS = SM number 1-36
 CC = board number 0, 2, 4 or 6
 source string will be something like CAEN/supermodule001/board11 for a 
board.*/
 DPNameWithoutSystem varchar2(1000);
 moduleNumber number;
 boardNumber number;
 invalid_board_name exception; --there probably isn't much point defining my own exception type since it will be out of scope by the time anything sees it and will just show as a user defined exception, but it's better than throwing some unrelated predefined exception

begin
        DPNameWithoutSystem:=substr(DPName,instr(DPName,':')+1);

moduleNumber:=to_number(regexp_substr(DPNameWithoutSystem,'[[:digit:]]+'));

boardNumber:=to_number(regexp_substr(DPNameWithoutSystem,'[[:digit:]]+',1,2));
        if moduleNumber is null or boardNumber is null then
                raise invalid_board_name;
        end if;
        return 1061000000+10000*moduleNumber+boardNumber;
end HVBOARDLOGICID;
