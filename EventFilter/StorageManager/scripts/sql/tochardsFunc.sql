create or replace function tochards(f_int interval day to second,f_fmt varchar2) return varchar2 is
 -- valid formats are DDD, HHH, HH, MMM, MM, SSS, SS, FF
 ret varchar2(4000);
 f varchar2(4000);
 i interval day(9) to second(9);
 begin
if (f_fmt is null or f_int is null) then
return null;
end if;
 f := upper(f_fmt);
 if (translate(f,'XDHMSF,.:;/- ','X') is not null) then
 raise_application_error(-20001,'Invalid format');
 end if;
 if (extract(day from i)<0) then
 ret:='-';
 i:=f_int*(-1);
 else
 ret:='';
 i:=f_int;
 end if;
 while (f is not null) loop
 if (f like 'DDD%') then
 ret:=ret||to_char(extract(day from i),'FM999999999999999999');
 f:=substr(f,4);
 elsif (f like 'HHH%') then
 ret:=ret||to_char(extract(day from i)*24+extract(hour from i),'FM999999999999999999');
 f:=substr(f,4);
 elsif (f like 'HH%') then
 ret:=ret||to_char(extract(hour from i),'FM999999999999999999');
 f:=substr(f,3);
 elsif (f like 'MMM%') then
 ret:=ret||to_char(extract(day from i)*24*60+extract(hour from i)*60+extract(minute from i),'FM999999999999999999');
 f:=substr(f,4);
 elsif (f like 'MM%') then
 ret:=ret||to_char(extract(minute from i),'FM999999999999999999');
 f:=substr(f,3);
 elsif (f like 'SSS%') then
 ret:=ret||to_char(extract(day from i)*24*60*60+extract(hour from i)*60*60+extract(minute from i)*60+trunc(extract(second from i)),'FM999999999999999999');
 f:=substr(f,4);
 elsif (f like 'SS%') then
 ret:=ret||to_char(trunc(extract(second from i)),'FM999999999999999999');
 f:=substr(f,3);
 elsif (f like 'FF%') then
 ret:=ret||to_char(mod(extract(second from i),1),'FM999999999999999999');
 f:=substr(f,3);
 elsif (substr(f,1,1) in (' ', '-', ':', ';', ',', '.', '/')) then
 ret:=ret||substr(f,1,1);
 f:=substr(f,2);
 else
 raise_application_error(-20001,'Invalid format : '||f_fmt);
 end if;
 end loop;
 return ret;
 end;
 ";
