CREATE OR REPLACE
function getAliasForDevice(DPEName in varchar2) return varchar2 is
	alias aliases.alias%type;
	DPName varchar2(1000);
	dotLocation number;
begin
	--get the name of the DP itself, not an element
	dotLocation:=instr(DPEName,'.');
	if dotLocation>0 then
		DPName:=substr(DPEName,1,dotLocation);
	end if;
	select alias into alias from (select alias from aliases where dpe_name=DPName order by since desc) where rownum=1;
	--perhaps should do something if there is no record... but for now I think we want there to always be one.
	return alias;
end;
/
show errors;
